# serve

Inference on Kaggle 2x T4, reached through a Raspberry Pi on the local network.

```
pi agent harness ── HTTP ──▶ 192.168.1.142:8080   (Pi, local network)
                                    │
                                    │ WebSocket relay (multiplexed HTTP)
                                    ▼
                    wss://<pi>.opentunnel.xyz/kaggle
                                    │
                             Kaggle kernel
                                    │
                             llama-server :8001
                                    │
                              Qwen3.8 27B GGUF
```

The Kaggle kernel has no public port and no dashboard. On boot it installs
llama.cpp, downloads the pinned GGUF, starts the engine, and dials the Pi. The
Pi keeps the only stable address, so the harness never sees a changing Kaggle
tunnel URL. There is no API key: the harness talks to the Pi over the LAN.

## Pi

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m serve.pi
```

That starts:

| Listener | Default | Purpose |
| --- | --- | --- |
| HTTP | `0.0.0.0:8080` | OpenAI-compatible endpoint for the harness |
| WebSocket | `0.0.0.0:8787` | Kaggle node connects here |

`serve.pi` also installs and starts opentunnel for port `8787` (set
`SERVE_PI_TUNNEL=0` to manage opentunnel yourself). It prints the URL to paste
into the Kaggle launcher and saves it to `serve-state/pi-tunnel-url.txt`:

```
kaggle relay url: wss://something.opentunnel.xyz/kaggle
```

Harness configuration:

```
OPENAI_BASE_URL=http://192.168.1.142:8080/v1
OPENAI_API_KEY=unused
```

## Kaggle

Edit `RASPI_RELAY_URL` in `.kaggle/kernel.py`, or create a Kaggle Secret named
`SERVE_RELAY_URL` with the same value. Then push:

```bash
kaggle kernels push -p .kaggle/
```

The kernel clones this repo from GitHub, so commit and push the `serve/` changes
before launching it.

The kernel clones this repo at `/kaggle/working/serve`, installs the Python
dependencies, and runs `python -m serve.main`. It automatically:

1. installs the pinned llama.cpp CUDA build,
2. downloads `unsloth/Qwen3.8-27B-GGUF` (`Q4_K_M` weights + `mmproj`),
3. starts `llama-server` on `127.0.0.1:8001` with both T4s,
4. warms the model once,
5. connects to the Pi relay and starts serving.

No dataset is attached and no model needs to be uploaded to Kaggle. After an
idle timeout it tears down the engine and ends the run so GPU time stops.

## Environment

| Variable | Default | Side |
| --- | --- | --- |
| `SERVE_MODEL` | `qwen3.8-27b-gguf` | Kaggle |
| `SERVE_RELAY_URL` | `ws://127.0.0.1:8787/kaggle` | Kaggle |
| `SERVE_RELAY_TOKEN` | unset | both |
| `SERVE_IDLE_TIMEOUT` | `600` | Kaggle |
| `SERVE_ENGINE_PORT` | `8001` | Kaggle |
| `SERVE_LLAMACPP_BUILD` | `b12318` | Kaggle |
| `SERVE_LLAMACPP_CUDA` | `cuda-12.8` | Kaggle |
| `SERVE_PI_HTTP_HOST` | `0.0.0.0` | Pi |
| `SERVE_PI_HTTP_PORT` | `8080` | Pi |
| `SERVE_PI_WS_HOST` | `0.0.0.0` | Pi |
| `SERVE_PI_WS_PORT` | `8787` | Pi |
| `SERVE_PI_WS_PATH` | `/kaggle` | Pi |
| `SERVE_PI_TUNNEL` | `1` | Pi |
| `SERVE_PI_TUNNEL_NAME` | `serve-relay` | Pi |

`SERVE_RELAY_TOKEN` is optional. It protects the public WebSocket from someone
who finds the opentunnel URL. It is not the harness API key.

## Layout

| Path | Role |
| --- | --- |
| `serve/main.py` | Kaggle node: runtime, model, engine, relay client |
| `serve/relay_client.py` | Kaggle side of the WebSocket relay |
| `serve/pi.py` | Pi HTTP server, WebSocket server, opentunnel |
| `serve/relay.py` | Frame format and header filtering |
| `serve/llamacpp.py` | llama.cpp CUDA download and verification |
| `serve/manager.py` | Model download/load lifecycle |
| `serve/catalog.py` | Model presets and engine argv |
