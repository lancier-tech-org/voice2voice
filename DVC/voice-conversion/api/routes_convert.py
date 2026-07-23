"""
Real-time voice conversion WebSocket — RVC backend.
Includes auto pitch detection for cross-gender conversion.
"""

import json
import time
import numpy as np
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import config

router = APIRouter()


def detect_median_f0(audio_np, sr=48000):
    """Detect median fundamental frequency from speech segments only."""
    try:
        import parselmouth

        # Only analyze segments with sufficient energy (actual speech)
        # Split audio into frames and find speech frames
        frame_len = int(0.03 * sr)  # 30ms frames
        rms_threshold = 0.01

        speech_segments = []
        for i in range(0, len(audio_np) - frame_len, frame_len):
            frame = audio_np[i:i + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > rms_threshold:
                speech_segments.append(frame)

        if len(speech_segments) < 10:  # Less than 300ms of speech
            return None

        speech_audio = np.concatenate(speech_segments)
        snd = parselmouth.Sound(speech_audio.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch_ac(pitch_floor=80, pitch_ceiling=600)
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        if len(voiced) > 5:
            median = float(np.median(voiced))
            print(f"[AutoPitch] Analyzed {len(speech_segments)} speech frames, median F0: {median:.0f}Hz")
            return median
    except Exception as e:
        print(f"[AutoPitch] F0 detection error: {e}")
    return None


def load_target_f0(voice_name):
    """Load target voice median F0 from metadata or compute from training data."""
    # Check metadata
    voice_dir = config.VOICES_DIR / voice_name
    meta_path = voice_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "median_f0" in meta:
            return meta["median_f0"]

    # Compute from training segments if available
    rvc_webui = Path(config.BASE_DIR).parent / "rvc-webui"
    gt_dir = rvc_webui / "logs" / voice_name / "0_gt_wavs"
    if gt_dir.exists():
        import librosa
        all_f0 = []
        wavs = sorted(gt_dir.glob("*.wav"))[:15]  # Sample 15 segments
        for wav in wavs:
            try:
                audio, sr = librosa.load(str(wav), sr=48000)
                f0 = detect_median_f0(audio, sr)
                if f0:
                    all_f0.append(f0)
            except:
                pass
        if all_f0:
            median = float(np.median(all_f0))
            # Save to metadata for next time
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {}
            meta["median_f0"] = median
            voice_dir.mkdir(exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[AutoPitch] Computed target F0: {median:.0f}Hz for {voice_name}")
            return median

    return None


@router.websocket("/stream")
async def stream_convert(websocket: WebSocket):
    await websocket.accept()
    state = websocket.app.state.app_state
    processor = state.stream_processor
    frame_count = 0

    try:
        config_msg = await websocket.receive_text()
        cfg = json.loads(config_msg)

        voice_name = cfg.get("voice")
        raw_pitch = cfg.get("pitch_shift", 0)
        auto_pitch = (raw_pitch == "auto")
        pitch_shift = 0.0 if auto_pitch else float(raw_pitch)
        sample_rate = int(cfg.get("sample_rate", 48000))

        if not voice_name:
            await websocket.send_json({"error": "voice name required"})
            await websocket.close()
            return

        # Find voice model
        voice_dir = config.VOICES_DIR / voice_name
        voice_path = None
        if voice_dir.is_dir():
            pth_files = list(voice_dir.glob("*.pth"))
            if pth_files:
                voice_path = str(pth_files[0])
        if not voice_path:
            loose = config.VOICES_DIR / f"{voice_name}.pth"
            if loose.exists():
                voice_path = str(loose)

        if not voice_path:
            await websocket.send_json({"error": f"Voice '{voice_name}' not found"})
            await websocket.close()
            return

        # Auto pitch: load target F0
        target_f0 = None
        auto_pitch_calibrated = False
        auto_pitch_buffer = []
        if auto_pitch:
            target_f0 = load_target_f0(voice_name)
            if target_f0:
                print(f"[AutoPitch] Target voice F0: {target_f0:.0f}Hz")
            else:
                print("[AutoPitch] Could not determine target F0, using pitch_shift=0")
                auto_pitch = False

        processor.start(
            voice_path=voice_path,
            pitch_shift=pitch_shift,
            input_sr=sample_rate,
        )

        await websocket.send_json({
            "status": "ready",
            "voice": voice_name,
            "latency_ms": processor.get_latency_ms(),
            "auto_pitch": auto_pitch,
            "target_f0": target_f0,
        })

        total_latency = 0

        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break

            if "text" in message:
                try:
                    cmd = json.loads(message["text"])
                    if cmd.get("action") == "stop":
                        break
                    elif cmd.get("action") == "switch_voice":
                        new_voice = cmd.get("voice")
                        new_dir = config.VOICES_DIR / new_voice
                        new_path = None
                        if new_dir.is_dir():
                            pths = list(new_dir.glob("*.pth"))
                            if pths:
                                new_path = str(pths[0])
                        if not new_path:
                            loose = config.VOICES_DIR / f"{new_voice}.pth"
                            if loose.exists():
                                new_path = str(loose)
                        if new_path:
                            processor.start(
                                voice_path=new_path,
                                pitch_shift=float(cmd.get("pitch_shift", pitch_shift)),
                                input_sr=sample_rate,
                            )
                            await websocket.send_json({
                                "status": "voice_switched", "voice": new_voice,
                            })
                        else:
                            await websocket.send_json({
                                "error": f"Voice '{new_voice}' not found",
                            })
                        continue
                    elif cmd.get("action") == "set_pitch":
                        new_pitch = float(cmd.get("pitch_shift", 0))
                        processor.pitch_shift = new_pitch
                        print(f"[WebSocket] Pitch shift updated to {new_pitch}")
                        continue
                except json.JSONDecodeError:
                    continue

            if "bytes" in message:
                audio_bytes = message["bytes"]

                # Auto pitch calibration from first ~2.5 seconds
                if auto_pitch and not auto_pitch_calibrated and target_f0:
                    chunk_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    auto_pitch_buffer.extend(chunk_np.tolist())

                    if len(auto_pitch_buffer) >= sample_rate * 4:
                        buf_np = np.array(auto_pitch_buffer, dtype=np.float32)
                        source_f0 = detect_median_f0(buf_np, sample_rate)

                        if source_f0 and source_f0 > 0:
                            pitch_shift = 12 * np.log2(target_f0 / source_f0)
                            pitch_shift = float(np.clip(pitch_shift, -24, 24))
                            pitch_shift = round(pitch_shift)
                            processor.pitch_shift = pitch_shift
                            print(f"[AutoPitch] Source: {source_f0:.0f}Hz → "
                                  f"Target: {target_f0:.0f}Hz → "
                                  f"Shift: {pitch_shift:+d} semitones")
                            await websocket.send_json({
                                "auto_pitch_result": {
                                    "source_f0": round(source_f0),
                                    "target_f0": round(target_f0),
                                    "shift": pitch_shift,
                                }
                            })
                        else:
                            print("[AutoPitch] Could not detect source F0")

                        auto_pitch_calibrated = True
                        auto_pitch_buffer = []

                t_start = time.perf_counter()
                output_bytes = processor.process(audio_bytes)
                t_elapsed = (time.perf_counter() - t_start) * 1000
                frame_count += 1
                total_latency += t_elapsed

                if output_bytes is not None:
                    await websocket.send_bytes(output_bytes)

                    if frame_count % 50 == 0:
                        avg_latency = total_latency / frame_count
                        await websocket.send_json({
                            "stats": {
                                "avg_processing_ms": round(avg_latency, 1),
                                "total_latency_ms": round(
                                    processor.get_latency_ms() + avg_latency, 1
                                ),
                                "frames_processed": frame_count,
                            }
                        })

    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            pass  # Normal disconnect
        else:
            import traceback
            traceback.print_exc()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        processor.stop()
        print(f"[WebSocket] Closed. Processed {frame_count} frames.")