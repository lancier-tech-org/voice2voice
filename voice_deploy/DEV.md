# Making changes — the everyday workflow

Everything below runs from the project folder:

```bash
cd ~/voice2voice/voice_deploy
```

---

## 1. Edit the file

| What you want to change | File |
|---|---|
| Delay before a phrase is converted, VAD sensitivity | `backend/config.py` |
| Conversion logic | `backend/tts_engine.py` |
| Phrase detection / segmentation | `backend/vad_engine.py` |
| Session handling, stats | `backend/pipeline.py` |
| API routes, voice upload | `backend/main.py` |
| The web page (buttons, layout, mic handling) | `frontend/index.html` |
| Add a target voice | drop a `.wav` into `backend/voices/` |

## 2. Apply the change

| What you edited | Command | Time |
|---|---|---|
| Any `.py` file | `docker compose restart` | ~20 s |
| `frontend/index.html` | *nothing* — just refresh the browser (Ctrl-Shift-R) | instant |
| A voice `.wav` | *nothing* — it appears in the dropdown | instant |
| `requirements.txt` | `docker compose up -d --build` | ~30 min |
| `Dockerfile` | `docker compose up -d --build` | ~5–35 min |

The `.py` and `index.html` files are mounted live into the container, which is
why they don't need a rebuild. See the comments in `docker-compose.yml`.

## 3. Check it came up

```bash
curl -s http://localhost:8131/api/health
```

Expected:

```json
{"vc":true,"vc_sr":24000,"voices":["pavan.wav","vineesha.wav"]}
```

If something looks wrong, watch the logs live:

```bash
docker logs -f voice-vc          # Ctrl-C to stop watching
```

A conversion prints a line like:

```
VC done: 1.84s in -> 1.91s out (0.86s processing, 0.47x realtime)
```

## 4. Test it

Open **https://v2v.lancieretech.com/** (or https://vc.lancieretech.com/).

- **Wear headphones.** There is no echo cancellation — speakers feed the
  converted audio back into the mic and it re-converts in a loop.
- **You hear nothing while speaking.** It waits for you to pause, then sends
  the whole phrase at once. That is by design, not a fault.
- **One person at a time.** The model holds a single global target voice;
  two people on different voices corrupt each other's audio.

## 5. Save your work

```bash
git add -A voice_deploy/
git commit -m "describe what you changed"
git push
```

---

## The main tuning dial

In `backend/config.py`:

```python
VAD_MIN_SILENCE_MS = 400   # how long you must pause before it converts
```

Lower = snappier reply, but it may cut you off mid-sentence.
Higher = waits longer, but keeps whole sentences together.

Change it, `docker compose restart`, listen. Under a minute per attempt.

Other dials in the same file:

- `VAD_THRESHOLD` (0.45) — lower catches quiet speech but lets noise through
- `VAD_MIN_SPEECH_MS` (500) — phrases shorter than this are ignored
- `DEFAULT_VOICE_REF` (None) — set to a filename to preload it at startup

---

## If something breaks

```bash
docker compose restart          # first thing to try
docker logs --tail 50 voice-vc  # what went wrong
docker compose up -d --force-recreate   # rebuild the container from the image
```

To go back to the last version that worked:

```bash
git log --oneline -10           # find a good commit
git checkout <commit> -- voice_deploy/backend/
docker compose restart
```

## Bringing RVC back

Nothing was deleted — the models and volumes are still on disk.

```bash
sudo cp /etc/nginx/sites-available/v2v.conf.rvc-backup /etc/nginx/sites-available/v2v.conf
sudo nginx -t && sudo systemctl reload nginx
cd ~/voice2voice/docker && docker compose up -d
```

That puts RVC back on https://v2v.lancieretech.com/. Chatterbox stays reachable
at https://vc.lancieretech.com/ throughout.
