"""
RVC Trainer — uses RVC WebUI's training scripts via subprocess.

Requires the RVC WebUI repo cloned alongside this project.
Training produces .pth + .index files compatible with RVC inference.
"""

import os
import sys
import json
import time
import shutil
import signal
import subprocess
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, Callable

import config

# Path to the cloned RVC WebUI repo
RVC_WEBUI_DIR = Path(config.BASE_DIR).parent / "rvc-webui"


class TrainingStopped(Exception):
    """Raised inside train() when a stop was requested. Carries the mode
    ('cancel' = discard, 'pause' = keep checkpoints for resume)."""
    def __init__(self, mode="cancel"):
        self.mode = mode
        super().__init__(mode)


class RVCTrainer:
    def __init__(self):
        self.is_training = False
        self.progress = 0.0
        self.status_message = ""
        self.python_cmd = sys.executable

        # Stop/resume support. The training subprocess is launched in its own
        # process group so we can kill it AND its multiprocessing workers together
        # (killing only the parent leaves GPU workers running — observed in prod).
        self._proc = None
        self._stop_mode = None      # None | "cancel" | "pause"
        self.current_voice = None

        if not RVC_WEBUI_DIR.exists():
            print(f"[RVCTrainer] WARNING: RVC WebUI not found at {RVC_WEBUI_DIR}")
            print(f"[RVCTrainer] Clone it: git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git {RVC_WEBUI_DIR}")

    def stop_training(self, mode: str = "cancel") -> bool:
        """
        Stop the current training run.

        mode="cancel": discard — caller deletes the artifacts afterwards.
        mode="pause" : keep logs/<voice> (the G_/D_ checkpoints) so it can resume.

        Kills the whole process group so RVC's multiprocessing GPU workers die too;
        killing only the parent leaves them running on the GPU.
        """
        self._stop_mode = mode if mode in ("cancel", "pause") else "cancel"
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(10):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.3)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                print(f"[RVCTrainer] Stopped training ({self._stop_mode}).")
            except Exception as e:
                print(f"[RVCTrainer] stop_training kill error: {e}")
        return True

    def train(self, voice_name: str, audio_path: str,
              sr: int = 40000, epochs: int = 200, batch_size: int = 8,
              progress_callback: Optional[Callable] = None,
              resume: bool = False) -> dict:

        self.is_training = True
        self.progress = 0.0
        self._stop_mode = None
        self._proc = None
        self.current_voice = voice_name
        start_time = time.time()

        sr_str = {32000: "32k", 40000: "40k", 48000: "48k"}.get(sr, "40k")
        exp_dir = RVC_WEBUI_DIR / "logs" / voice_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        voice_output_dir = config.VOICES_DIR / voice_name
        voice_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # If resuming and the extracted features already exist, skip the whole
            # preprocessing pipeline — it is deterministic and was already done.
            # train.py then auto-resumes from the latest G_/D_ checkpoint in exp_dir.
            features_ready = (
                (exp_dir / "filelist.txt").exists()
                and (exp_dir / "3_feature768").is_dir()
                and any((exp_dir / "3_feature768").glob("*.npy"))
            )
            if resume and features_ready:
                self._update(0.44, "Resuming from last checkpoint...", progress_callback)
            else:
                # Step 1: Copy audio to a training directory
                self._update(0.05, "Preparing audio...", progress_callback)
                trainset_dir = exp_dir / "raw_audio"
                trainset_dir.mkdir(exist_ok=True)
                dst = trainset_dir / Path(audio_path).name
                shutil.copy2(audio_path, dst)

                # Step 2: Preprocess (resample, slice)
                self._update(0.10, "Preprocessing audio (slicing, resampling)...", progress_callback)
                self._run_preprocess(voice_name, str(trainset_dir), sr_str)

                # Step 3: Extract F0 (pitch)
                self._update(0.25, "Extracting pitch (F0 with RMVPE)...", progress_callback)
                self._run_extract_f0(voice_name)

                # Step 4: Extract HuBERT features
                self._update(0.35, "Extracting HuBERT features...", progress_callback)
                self._run_extract_features(voice_name)

                # Step 5: Generate filelist + config
                self._update(0.40, "Generating training config...", progress_callback)
                self._generate_filelist(voice_name, sr_str)

            # Step 6: Train
            self._update(0.45, f"Training model (0/{epochs} epochs)...", progress_callback)
            self._run_train(voice_name, sr_str, epochs, batch_size, progress_callback)

            # Step 7: Build FAISS index
            self._update(0.90, "Building voice index...", progress_callback)
            self._build_index(voice_name)

            # Step 8: Copy final model to voices directory
            self._update(0.95, "Saving voice model...", progress_callback)
            self._copy_final_model(voice_name, voice_output_dir)

            elapsed = time.time() - start_time
            meta = {
                "voice_name": voice_name,
                "epochs": epochs,
                "sample_rate": sr,
                "training_time_min": round(elapsed / 60, 1),
            }
            with open(voice_output_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            self._update(1.0, "Training complete!", progress_callback)
            return {"status": "success", "voice_dir": str(voice_output_dir), "metadata": meta}

        except TrainingStopped as s:
            print(f"[RVCTrainer] Training stopped: {s.mode}")
            return {"status": "stopped", "mode": s.mode,
                    "progress": round(self.progress, 3)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            # A stop kills the subprocess, which surfaces as a non-zero exit; treat
            # that as a stop, not a failure.
            if self._stop_mode is not None:
                return {"status": "stopped", "mode": self._stop_mode,
                        "progress": round(self.progress, 3)}
            self._update(0, f"Training failed: {str(e)}", progress_callback)
            return {"status": "error", "message": str(e)}
        finally:
            self.is_training = False
            self._proc = None
            self.current_voice = None

    def _run_cmd(self, cmd, step_name):
        """
        Run a preprocessing/extraction subprocess in the RVC WebUI directory.

        Launched in its own process group and tracked in self._proc so
        stop_training() can kill it (and its workers) mid-step. Without this, a
        Cancel during preprocessing/F0/feature extraction could not interrupt the
        current step — the run would hang until it finished on its own.
        """
        print(f"[RVCTrainer] Running: {cmd}")
        if self._stop_mode is not None:
            raise TrainingStopped(self._stop_mode)

        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(RVC_WEBUI_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        self._proc = proc
        out, _ = proc.communicate()
        self._proc = None

        if self._stop_mode is not None:
            raise TrainingStopped(self._stop_mode)
        if proc.returncode != 0:
            print(f"[RVCTrainer] {step_name} OUTPUT: {(out or '')[-2000:]}")
            raise RuntimeError(f"{step_name} failed (exit code {proc.returncode}): {(out or '')[-500:]}")
        if out:
            print(f"[RVCTrainer] {step_name}: {out[-500:]}")
        return proc

    def _run_preprocess(self, voice_name, trainset_dir, sr_str):
        """Step 1: Preprocess audio — resample and slice into segments."""
        sr_val = {"32k": 32000, "40k": 40000, "48k": 48000}[sr_str]
        n_p = os.cpu_count() or 4
        exp_dir = f"{RVC_WEBUI_DIR}/logs/{voice_name}"

        cmd = (
            f'"{self.python_cmd}" infer/modules/train/preprocess.py '
            f'"{trainset_dir}" {sr_val} {n_p} "{exp_dir}" False 3.7'
        )
        self._run_cmd(cmd, "Preprocess")

    def _run_extract_f0(self, voice_name):
        """Step 2: Extract F0 using RMVPE."""
        exp_dir = f"{RVC_WEBUI_DIR}/logs/{voice_name}"
        n_p = os.cpu_count() or 4

        # Use the standard F0 extraction (not RMVPE GPU variant for simplicity)
        cmd = (
            f'"{self.python_cmd}" infer/modules/train/extract/extract_f0_print.py '
            f'"{exp_dir}" {n_p} rmvpe'
        )
        self._run_cmd(cmd, "Extract F0")

    def _run_extract_features(self, voice_name):
        """Step 3: Extract HuBERT content features."""
        exp_dir = f"{RVC_WEBUI_DIR}/logs/{voice_name}"

        cmd = (
            f'"{self.python_cmd}" infer/modules/train/extract_feature_print.py '
            f'cuda:0 1 0 0 "{exp_dir}" v2 True'
        )
        self._run_cmd(cmd, "Extract Features")

    def _generate_filelist(self, voice_name, sr_str):
        """Step 4: Generate the training filelist and config."""
        exp_dir = RVC_WEBUI_DIR / "logs" / voice_name
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        feature_dir = exp_dir / "3_feature768"
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"

        # Get names that exist in all directories
        names = (
            set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
            & set(n.split(".")[0] for n in os.listdir(feature_dir))
            & set(n.split(".")[0] for n in os.listdir(f0_dir))
            & set(n.split(".")[0] for n in os.listdir(f0nsf_dir))
        )

        if not names:
            raise RuntimeError("No valid training samples found after preprocessing")

        opt = []
        spk_id = 0
        for name in names:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
                f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )

        # Add mute samples if they exist
        mute_dir = RVC_WEBUI_DIR / "logs" / "mute"
        if (mute_dir / "0_gt_wavs").exists():
            for _ in range(2):
                opt.append(
                    f"{mute_dir}/0_gt_wavs/mute{sr_str}.wav|"
                    f"{mute_dir}/3_feature768/mute.npy|"
                    f"{mute_dir}/2a_f0/mute.wav.npy|"
                    f"{mute_dir}/2b-f0nsf/mute.wav.npy|{spk_id}"
                )

        import random
        random.shuffle(opt)
        with open(exp_dir / "filelist.txt", "w") as f:
            f.write("\n".join(opt))

        # Write config
        config_path = f"v2/{sr_str}.json"
        config_src = RVC_WEBUI_DIR / "configs" / config_path
        config_dst = exp_dir / "config.json"
        if not config_dst.exists():
            if not config_src.exists():
                raise RuntimeError(
                    f"RVC config not found: {config_src}. "
                    f"Available: {list((RVC_WEBUI_DIR / 'configs' / 'v2').glob('*.json'))}"
                )
            shutil.copy2(config_src, config_dst)

        print(f"[RVCTrainer] Filelist: {len(opt)} entries, config: {config_path}")

    def _run_train(self, voice_name, sr_str, epochs, batch_size, progress_callback):
        """Step 5: Run the actual training."""
        # Find pretrained models.
        #
        # These are NOT optional. RVC fine-tunes from a base model trained on
        # thousands of hours of speech; the per-voice run only adapts it. Training
        # without them starts from random weights, and ~200 epochs on a few minutes
        # of audio cannot learn to synthesise speech — it yields a checkpoint that
        # looks valid (right shape, "200epoch" tag) but sounds badly degraded.
        #
        # This previously fell back to "" and trained anyway, silently producing a
        # broken voice with no error. Fail loudly instead.
        pretrained_G = str(RVC_WEBUI_DIR / "assets" / "pretrained_v2" / f"f0G{sr_str}.pth")
        pretrained_D = str(RVC_WEBUI_DIR / "assets" / "pretrained_v2" / f"f0D{sr_str}.pth")
        missing = [p for p in (pretrained_G, pretrained_D) if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                "Pretrained base models are missing, so training would start from "
                "random weights and produce an unusable voice. Missing: "
                + ", ".join(missing)
                + ". Download f0G{sr}.pth and f0D{sr}.pth into "
                  "assets/pretrained_v2/ before training.".format(sr=sr_str)
            )

        # Build command — train.py uses argparse via get_hparams()
        cmd_parts = [
            f'"{self.python_cmd}"', 'infer/modules/train/train.py',
            '-e', f'"{voice_name}"',
            '-sr', f'"{sr_str}"',
            '-f0', '1',
            '-bs', str(batch_size),
            '-g', '0',
            '-te', str(epochs),
            '-se', '50',
            '-l', '1',
            '-c', '0',
            '-sw', '1',
            '-v', 'v2',
        ]
        if pretrained_G:
            cmd_parts.extend(['-pg', f'"{pretrained_G}"'])
        if pretrained_D:
            cmd_parts.extend(['-pd', f'"{pretrained_D}"'])

        cmd = " ".join(cmd_parts)
        print(f"[RVCTrainer] Training command: {cmd}")

        # Bail out immediately if a stop was requested before we even launched
        # (e.g. cancelled during preprocessing).
        if self._stop_mode is not None:
            raise TrainingStopped(self._stop_mode)

        # start_new_session=True puts the shell, train.py and its multiprocessing
        # workers in one process group, so stop_training() can kill them together.
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(RVC_WEBUI_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        self._proc = proc

        for line in proc.stdout:
            line = line.strip()
            if line:
                print(f"[RVC-Train] {line}")
                if "Epoch" in line or "epoch" in line or "E " in line:
                    try:
                        import re
                        match = re.search(r'[Ee](?:poch)?[:\s]*(\d+)', line)
                        if match:
                            current_epoch = int(match.group(1))
                            if current_epoch <= epochs:
                                progress = 0.45 + 0.45 * current_epoch / epochs
                                self._update(
                                    progress,
                                    f"Training epoch {current_epoch}/{epochs}",
                                    progress_callback,
                                )
                    except:
                        pass

        proc.wait()
        if self._stop_mode is not None:
            raise TrainingStopped(self._stop_mode)
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    def _build_index(self, voice_name):
        """Step 6: Build FAISS index for voice retrieval."""
        exp_dir = RVC_WEBUI_DIR / "logs" / voice_name
        feature_dir = exp_dir / "3_feature768"

        if not feature_dir.exists():
            print("[RVCTrainer] No feature directory, skipping index")
            return

        npys = []
        for name in sorted(os.listdir(feature_dir)):
            if name.endswith(".npy"):
                phone = np.load(feature_dir / name)
                npys.append(phone)

        if not npys:
            print("[RVCTrainer] No features found, skipping index")
            return

        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        np.save(exp_dir / "total_fea.npy", big_npy)

        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        n_ivf = max(n_ivf, 1)

        index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
        index.train(big_npy)
        index.add(big_npy)

        index_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{voice_name}_v2.index"
        faiss.write_index(index, str(index_path))
        print(f"[RVCTrainer] Index built: {index_path}")

    def _copy_final_model(self, voice_name, voice_output_dir):
        """Step 7: Copy trained .pth and .index to voices directory."""
        exp_dir = RVC_WEBUI_DIR / "logs" / voice_name
        weights_dir = RVC_WEBUI_DIR / "assets" / "weights"

        # Find the .pth file — RVC saves to assets/weights/
        pth_found = False
        for search_dir in [weights_dir, exp_dir]:
            if search_dir.exists():
                for pth in sorted(search_dir.glob(f"{voice_name}*.pth"),
                                  key=os.path.getmtime, reverse=True):
                    dst = voice_output_dir / f"{voice_name}.pth"
                    shutil.copy2(pth, dst)
                    print(f"[RVCTrainer] Model copied: {pth} → {dst}")
                    pth_found = True
                    break
            if pth_found:
                break

        if not pth_found:
            # Check for G_*.pth pattern and extract
            for pth in sorted(exp_dir.glob("G_*.pth"),
                              key=os.path.getmtime, reverse=True):
                # This is the raw checkpoint, need to extract
                try:
                    from rvc.lib.train.process_ckpt import extract_small_model
                    dst = voice_output_dir / f"{voice_name}.pth"
                    extract_small_model(str(pth), str(dst), "40k", True, voice_name, "v2")
                    print(f"[RVCTrainer] Model extracted: {pth} → {dst}")
                    pth_found = True
                except Exception as e:
                    print(f"[RVCTrainer] Could not extract model: {e}")
                    # Fallback: just copy the raw checkpoint
                    dst = voice_output_dir / f"{voice_name}.pth"
                    shutil.copy2(pth, dst)
                    pth_found = True
                break

        if not pth_found:
            raise FileNotFoundError(f"No trained model found for {voice_name}")

        # Copy index file
        for idx_file in exp_dir.glob("added_*.index"):
            dst = voice_output_dir / f"{voice_name}.index"
            shutil.copy2(idx_file, dst)
            print(f"[RVCTrainer] Index copied: {idx_file} → {dst}")
            break

    def _update(self, progress, message, callback=None):
        self.progress = progress
        self.status_message = message
        print(f"[RVCTrainer] {progress*100:.0f}% — {message}")
        if callback:
            callback(progress, message)