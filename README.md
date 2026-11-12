# Serve

A small control plane for vLLM on a Kaggle GPU kernel. It serves an OpenAI
compatible API through an [opentunnel](https://github.com/anomalyco/opentunnel)
URL and gives you a dashboard to download, load, unload and switch models
without touching the kernel.

```
start kernel ──▶ tunnel URL ──▶ dashboard ──▶ load model ──▶ REST API
                                    │
                          idle 10 min └──▶ kernel exits
```

## What it does

- Starts a dashboard on `:8000` and opens a public `https://…opentunnel.xyz`
  URL from inside the kernel.
- Runs exactly one model at a time. Loading another unloads the first, which
  is what you want when both GPUs are shared by one model.
- Downloads weights from Hugging Face with a disk preflight (Kaggle's working
  disk is small), or uses weights mounted from a Kaggle Dataset.
- Installs `vllm` or `vllm-omni` on demand and streams the install log to the
  dashboard.
- Proxies `/v1/*` to the active engine, including streaming responses.
- Ends the kernel after 10 minutes without inference traffic.

Everything lives under `/kaggle/working/serve-state` while the run lasts.

## Read this before loading the 27B

Two T4s give 32 GB of VRAM and **no bf16 or FP8**. Models must run in float16,
so the 27B needs a 4-bit checkpoint. The preset uses
`btbtyler09/Qwen3.8-27B-GPTQ-4bit` with tensor parallelism 2. GPTQ is
supported on Turing (compute capability 7.5), and this checkpoint uses group
size 32 without act-order, so vLLM can pick a compatible GPTQ kernel.

That checkpoint is **21 GB on disk**, and Kaggle's working disk is about
**20 GB**. It cannot be downloaded inside the kernel. Mount it as a Kaggle
Dataset instead: `/kaggle/input` is read-only and does not consume the working
disk. `scripts/upload_model_dataset.sh` prepares and uploads such a dataset
from a machine with enough disk.

The kernel metadata pins that dataset, so it is mounted automatically at boot.
The gateway detects it under `/kaggle/input` by folder name and loads it
without a download.

If you just want to check that the tunnel and API work, load `qwen3-8b-awq`
(5.6 GB). It fits comfortably and downloads inside the kernel.

## Image generation

Image models are served by the same gateway through vLLM-Omni (`--omni`) and
the same `/v1` surface (`/v1/images/generations`, `/v1/images/edits`).

vLLM-Omni lists compute capability 7.5 as supported, but diffusion memory
requirements vary a lot:

| Preset | Approx. VRAM | On one T4 (16 GB) |
| --- | --- | --- |
| `sdxl-base` | ~12 GB | should fit — start here |
| `sd35-medium` | ~14 GB | tight |
| `flux2-klein-4b` | ~16 GB | likely too tight |
| `z-image-turbo` | ~20 GB | no |

`vllm-omni==0.32.0` is installed together with `vllm==0.32.0`. If a diffusion
pipeline fails at startup on Turing, the error and the full traceback are in
the engine log tab; the text models are unaffected.

## Quick start

1. Push this repository to GitHub. The kernel clones it at boot.

   ```bash
   git add -A && git commit -m "serve gateway"
   git push origin main
   ```

2. Push the kernel. The id, T4 shape and the 27B dataset source are already
   hardcoded in `.kaggle/`:

   ```bash
   kaggle kernels push -p .kaggle/
   ```

3. Open the kernel on Kaggle and watch the log. When the tunnel is ready it
   prints:

   ```
   tunnel: dashboard: https://serve.<tunnel-id>.opentunnel.xyz
   ```

   The same line is in `/kaggle/working/serve-state/logs/tunnel.log`.

4. Open the URL. Click **install** on vLLM (or just **load** a model and it
   installs first), then **download** and **load**. The dashboard shows the
   exact `curl` command for the active model.

### Keep the same URL between kernels

A fresh kernel gets a fresh tunnel identity and therefore a new URL. To keep
the URL stable:

1. In the dashboard header, click **identity**. This copies a base64 bundle of
   the tunnel key, token and certificate.
2. In Kaggle, open the kernel → **Add-ons → Secrets** and add a secret named
   `OPENTUNNEL_IDENTITY_B64` with that value.
3. The next run restores the identity and reuses the same hostname.

The bundle is a private key. Keep the kernel private and don't paste it
anywhere public.

### Optional secrets

| Secret | Purpose |
| --- | --- |
| `HUGGINGFACE_TOKEN` | gated/private models and higher download limits |
| `OPENTUNNEL_IDENTITY_B64` | stable tunnel URL |
| `SERVE_IDLE_TIMEOUT` | idle seconds, e.g. `1800` |

## Using the API

The model name is the preset key (`qwen3.8-27b-gptq`, `sdxl-base`, …). When no
model is loaded, `/v1/*` returns a JSON 503 instead of hanging.

```bash
BASE=https://serve.<tunnel-id>.opentunnel.xyz

curl $BASE/v1/models

curl $BASE/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b-gptq",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'

curl $BASE/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{
    "model": "sdxl-base",
    "prompt": "a ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 20
  }'
```

Any OpenAI SDK works by pointing it at `$BASE/v1`.

### Control API

| Method | Path | Meaning |
| --- | --- | --- |
| `GET` | `/api/state` | everything the dashboard renders |
| `POST` | `/api/models/{key}/load` | load (body `{"replace": true}` to swap) |
| `POST` | `/api/models/unload` | stop the engine |
| `POST` | `/api/models/{key}/download` | download in the background |
| `DELETE` | `/api/models/{key}/cache` | delete downloaded weights |
| `POST` | `/api/models` | add a custom Hugging Face model |
| `DELETE` | `/api/models/{key}` | remove a custom model from the list |
| `POST` | `/api/runtime/{engine}/install` | install `vllm` or `vllm-omni` |
| `PUT` | `/api/settings` | `{"idle_timeout": seconds}` |
| `POST` | `/api/shutdown` | stop the engine and end the kernel run |
| `GET` | `/api/logs?source=engine` | `engine`, `install`, `tunnel`, `app` |

### Custom models

Add one from the dashboard or with a request:

```bash
curl -X POST $BASE/api/models \
  -H 'content-type: application/json' \
  -d '{
    "repo": "owner/model-AWQ",
    "kind": "chat",
    "engine": "vllm",
    "tensor_parallel": 2,
    "max_model_len": 16384,
    "extra_args": ["--enforce-eager"],
    "local_paths": ["/kaggle/input/my-model"]
  }'
```

The key is derived from the repo name. `local_paths` is checked before
Hugging Face, so a dataset mount avoids the download.

## Idle shutdown

After 10 minutes without **inference traffic or control actions**, the gateway
stops the engine, closes the tunnel and exits the kernel process. Kaggle then
ends the run and stops metering GPU time.

- Dashboard polling (`/api/state`, logs) does not count.
- In-flight requests, downloads, installs and loads pause the timer.
- A request that arrives before the deadline resets it.
- Change it in the dashboard, with `SERVE_IDLE_TIMEOUT`, or via
  `PUT /api/settings`.

Locally (`SERVE_TUNNEL=0`) the process does not exit, so you can develop
without killing your shell; set `SERVE_ALLOW_EXIT=1` to test the real
behaviour.

## Why a Kaggle Dataset for the 27B

```bash
# on a machine with ~25 GB free; needs the Kaggle CLI and huggingface_hub
./scripts/upload_model_dataset.sh btbtyler09/Qwen3.8-27B-GPTQ-4bit
```

The script:

1. checks free disk against the size reported by the Hugging Face API,
2. downloads the snapshot into `./.model-datasets/<slug>` (override with
   `--workdir`),
3. writes `dataset-metadata.json` using your Kaggle username and the slug the
   preset expects (`qwen3-8-27b-gptq-4bit`), and
4. runs `kaggle datasets create --dir-mode zip`, or `datasets version` when the
   dataset already exists, then waits for it to become ready.

Use `--yes` for unattended runs, `--clean` to delete the staging copy
afterwards, and `--skip-space-check` only when the size lookup fails. A
`HUGGINGFACE_TOKEN` or `HF_TOKEN` environment variable is used for gated
repos.

The 27B dataset is already pinned in `.kaggle/kernel-metadata.json`, so the
kernel mounts it automatically:

```json
"dataset_sources": ["katherinameyerkat/qwen3-8-27b-gptq-4bit"]
```

The gateway finds it under `/kaggle/input` and uses it in place. To use a
different model, upload it, update `dataset_sources`, and keep the derived slug
in the model's `local_paths`.

## Layout

```
serve/
  config.py     paths, env, ports
  catalog.py    presets and vLLM argv construction
  manager.py    load/unload, downloads, state machine
  engine.py     vLLM child process (own session, health check, logs)
  runtime.py    uv-based install of vllm / vllm-omni
  tunnel.py     opentunnel CLI, route, identity bundle
  idle.py       idle watchdog
  app.py        FastAPI routes, /v1 streaming proxy
  static/       dashboard (no build step, no CDN)
scripts/
  upload_model_dataset.sh  HF snapshot -> private Kaggle Dataset
.kaggle/
  kernel.py             launcher: clone repo, inject secrets, exec serve
  kernel-metadata.json  T4 x2, internet on, private
tests/                  behavior tests for the catalog, state machine, proxy
```

## Local development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q

SERVE_TUNNEL=0 SERVE_ALLOW_EXIT=0 .venv/bin/python -m serve.main
# http://127.0.0.1:8000
```

Without vLLM installed the dashboard still works; a load fails with a clear
error and an install button.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| tunnel says `error` | No outbound access, or provisioning was rate limited. Click the pill to retry; the tunnel log has the details. |
| `route add` hangs | Certificate issuance can take a few minutes on first use. The identity secret removes provisioning from later runs. |
| Load fails with CUDA OOM | Lower `max_model_len`, lower `gpu_memory_utilization`, or add `--enforce-eager`. The full traceback is in the engine log. |
| 21 GB model refuses to download | Expected: it does not fit on the working disk. Mount it as a Kaggle Dataset. |
| Image model 404s | `/v1/images/*` only exists on an `image`/omni model. Chat models expose `/v1/chat/completions`. |
| `vllm-omni` install fails | It needs Python 3.11–3.13 and the `vllm==0.32.0` console script; the install log shows the resolver error. |
| Kernel still running after closing the tab | The idle timer only sees API traffic. Wait 10 minutes or press **end session**. |
| Stale code after a push | The launcher resets the clone to `origin/main` on every boot; re-push the kernel to start a new run. |

## Security

The dashboard and API have no authentication. The tunnel hostname is random
and never appears in certificate transparency logs, so the URL itself is the
secret. Keep the kernel private, share the URL deliberately, and remember that
anyone with it can unload models or end the session.
