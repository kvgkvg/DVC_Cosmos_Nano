# dvc_cosmos_nano

Modular **Cosmos3-Nano** timestamped video captioning: a reusable inference
core, a CLI over it, and a web app over that.

```
core/                    lower layer — reusable inference library, no CLI/web deps
  config.py                 pipeline defaults (PipelineConfig)
  ffmpeg_utils.py            ffmpeg/ffprobe wrappers
  timing.py                  pure logic: chunk planning, line parsing, SRT rendering
  model.py                   model loading + generation
  pipeline.py                CosmosCaptioner — load once, .caption(...) many times
infer.py                   CLI over core/ — batch (manifest -> predictions JSONL)
                          and single-video (one .mp4 -> one .srt) modes
server/                  higher layer — web app over core/
  app.py                     FastAPI server: upload a video, poll for captions
  jobs.py                    sequential in-process job queue (one GPU, one worker)
  static/index.html          upload form + video player with live captions
system_prompt.txt        output-format contract, sent as the system prompt
requirements.txt
```

The pipeline itself (global-context pass + overlapping chunked captioning,
stitched into one SRT) is unchanged from `demo_cosmos/infer_cosmos.py` — this
project just splits it into a library (`core/`) that both the CLI and the
server call, instead of duplicating the logic per entry point.

---

## Setup

**1. Environment** — needs a CUDA GPU, `torch`, `transformers>=5.11` (for
`Cosmos3OmniForConditionalGeneration`), plus `fastapi`/`uvicorn` for the web
server:

```bash
conda create -n cosmos python=3.13 -y && conda activate cosmos
pip install -r requirements.txt
```

**2. ffmpeg** — `ffmpeg` and `ffprobe` must be on `PATH`:

```bash
sudo apt install ffmpeg
```

**3. Model** — [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano),
public, ~33 GB. Downloads on first run, or pre-fetch:

```bash
hf download nvidia/Cosmos3-Nano
```

**4. Check the install** — no GPU, model or ffmpeg needed:

```bash
python infer.py --selfcheck
# core.timing selfcheck OK
```

---

## CLI (lower layer)

**Single video:**

```bash
python infer.py caption --video clip.mp4 --instruction "fold the towel" --out clip.srt
# or print the SRT to stdout
python infer.py caption --video clip.mp4
```

**Batch, over a manifest** (same manifest/output format and flags as
`demo_cosmos/infer_cosmos.py`, resumable — episode_uids already in `--out` are
skipped):

```bash
python infer.py batch --manifest val.jsonl --out preds/cosmos3-nano_val.jsonl
```

Manifest is JSONL, one episode per line:

```json
{"episode_uid": "...", "video": "videos/ep024.mp4", "instruction": "...", "duration_s": 10.0}
```

Both subcommands share the pipeline flags: `--prompt`, `--model`,
`--chunk-seconds`, `--overlap-seconds`, `--chunk-fps`, `--global-num-frames`,
`--global-max-tokens`, `--chunk-max-tokens`. Run `python infer.py caption -h` /
`batch -h` for the full list.

---

## Web app (higher layer)

Loads the model once at startup, then serves uploads through a single
sequential worker (one GPU — no concurrent `model.generate()` calls):

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — upload a video, optionally add an instruction,
submit, and once the job finishes the page shows the video with captions
running live against playback, plus the full cue list below.

**API**, if you want to drive it from a script instead of the page:

| | |
|---|---|
| `POST /api/jobs` | multipart form: `video` (file), `instruction` (optional text) → `{job_id, status}` |
| `GET /api/jobs/{job_id}` | `{status: queued\|running\|done\|error, result, error, video_url}` — poll until `done`/`error` |
| `GET /api/jobs/{job_id}/video` | the uploaded video, for playback |

`result` (once `status == "done"`) is `{"pred": "<SRT text>", "n_events": N,
"n_chunks": M, "global_caption": "...", "duration_s": ...}`.

Uploaded videos are kept under `server/uploads/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command failed: ffprobe ...` | `ffmpeg`/`ffprobe` not on `PATH` |
| `ImportError: Cosmos3OmniForConditionalGeneration` | `transformers` older than 5.11 |
| `no parseable timestamped lines in any chunk` | model answered in prose; check `--prompt` points at `system_prompt.txt` |
| CUDA OOM | lower `--chunk-seconds` or `--chunk-fps` |
| server hangs on first request | expected — model load (~33GB) happens at `uvicorn` startup, before it accepts traffic |
