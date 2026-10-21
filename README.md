# strandsdecider — Docker

Container images for [strandsdecider](https://github.com/strands-labs/strands-decider), a System One decision model that picks between options (`choice`), answers yes/no (`noul`), or rates on a scale (`score`) with calibrated confidence — served over HTTP at `POST /v1/systemone`.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the runtime image (Python 3.11, CPU-only torch, `strands-decider` CLI) |
| `docker-compose.yml` | Runs the server with a named volume backing `HF_HOME` so model checkpoints persist |
| `README.md` | This file |

## Requirements

- Docker (Compose v2, `docker compose`)
- At least ~6 GB free disk: the image is ~400 MB, and each Hugging Face checkpoint (e.g. the Qwen3.5-2B base at ~4.5 GB) is downloaded into the `hf_cache` volume on first run.

## Build

```bash
docker compose build
```

### Optional build args

```bash
# Install the vision extra (Pillow) for image input
docker build --build-arg STRANDS_DECIDER_EXTRAS="[vision]" -t strandsdecider:vision .

# Install from git main (needed for Gemma 4, not yet on PyPI)
docker build --build-arg STRANDS_DECIDER_GIT=1 -t strandsdecider:git .

# Install bitsandbytes for 8-bit / 4-bit quantized loading
docker build --build-arg INSTALL_BITSANDBYTES=1 -t strandsdecider:q8 .

# Install IPEX for CPU optimizations (AVX2/VNNI, BF16, graph optimization)
docker build --build-arg INSTALL_IPEX=1 -t strandsdecider:ipex .

# Install IPEX + Intel Neural Compressor for CPU quantization (INT8)
docker build --build-arg INSTALL_IPEX=1 --build-arg INSTALL_IPEX_QUANT=1 -t strandsdecider:ipex-quant .
```

### Running with IPEX CPU optimizations

The standard `strands-decider` CLI doesn't expose IPEX flags. Instead, use the **custom entrypoint** (`serve_ipex.py`) which applies IPEX optimizations at model load time. This is done via the `ipex` compose profile:

```bash
# Build and run IPEX-optimized service on port 8003
docker compose --profile ipex up -d --build strandsdecider-ipex

# Or manually with docker run
docker run -p 8000:8000 \
  --entrypoint python3 /app/serve_ipex.py \
  strandsdecider:ipex \
  StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --port 8000 --host 0.0.0.0 --ipex-optimize
```

### Running with IPEX CPU quantization (INT8)

Similarly, quantization is applied programmatically via the custom entrypoint. Use the `ipex-quant` compose profile:

```bash
# Build and run IPEX + INT8 quantized service on port 8002
docker compose --profile ipex-quant up -d --build strandsdecider-ipex-quant

# Or manually
docker run -p 8000:8000 \
  --entrypoint python3 /app/serve_ipex.py \
  strandsdecider:ipex-quant \
  StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --port 8000 --host 0.0.0.0 --ipex-optimize --ipex-quantize int8_dynamic
```

> **Note:** IPEX quantization uses Intel Neural Compressor for INT8 static/dynamic quantization on CPU. This can provide 2-4x memory reduction and 1.5-3x speedup on Intel CPUs with AVX2/VNNI. Static quantization requires calibration data; dynamic quantization works out-of-the-box.

### Running with bitsandbytes quantization (CUDA)

Pass `--load-in-8bit` or `--load-in-4bit` via `STRANDS_ARGS` (requires the `INSTALL_BITSANDBYTES=1` build):

```bash
docker run -p 8000:8000 -e STRANDS_ARGS="--port 8000 --host 0.0.0.0 --load-in-8bit" strandsdecider:q8
```

Or via `docker-compose.yml` / `.env`:

```
STRANDS_ARGS=--port 8000 --host 0.0.0.0 --load-in-4bit
```

> **Note:** bitsandbytes quantization is primarily designed for CUDA GPUs. The default image uses CPU-only torch, so quantized CPU inference is limited and slower. For best results with `--load-in-8bit`/`--load-in-4bit`, use a CUDA-capable image or run on a machine with a GPU.

## Run the server

### Default CPU service (port 8000)

```bash
docker compose up -d
```

### IPEX-optimized service (port 8003)

```bash
docker compose --profile ipex up -d
```

### IPEX + INT8 quantized service (port 8002)

```bash
docker compose --profile ipex-quant up -d
```

### GPU service (port 8001, requires NVIDIA GPU)

```bash
docker compose --profile cuda up -d
```

The service listens on `http://localhost:8000` (or the respective port). Query it:

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
      "urgent": { "type": "noul", "instructions": "Does this convey urgency?" }
    }
  }'
```

Response shape:

```json
{
  "model": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
  "answers": { "urgent": { "type": "noul", "noul": 0.8581 } },
  "usage": { "input_tokens": 12, "output_tokens": 1 },
  "latency_ms": 42
}
```

### Configuration (via `.env` or `docker compose -e`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_ID` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | Hugging Face checkpoint to serve |
| `STRANDS_ARGS` | `--port 8000 --host 0.0.0.0` | Extra CLI flags appended after the model id |

Example `.env`:

```
MODEL_ID=StrandsAgents/strands-decider-E4B-gemma4-v1-2610
STRANDS_ARGS=--vision --port 8080
```

Then `docker compose up -d` picks them up automatically.

## Persistence

`HF_HOME` is set to `/root/.cache/huggingface` inside the container, backed by the named Docker volume `hf_cache`. Model checkpoints downloaded once survive container recreation — run `docker compose down` (without `-v`) to keep the cache; `docker compose down -v` wipes it.

## One-off `ask` (no server)

Use the CLI directly without starting the server:

```bash
docker run --rm strandsdecider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

With image input (vision build):

```bash
docker run --rm strandsdecider:vision ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "" --image page.png \
  --noul "Is the signature block filled in?"
```

## Stop

```bash
docker compose down
```

## Troubleshooting

- **`Missing argument 'checkpoint'`** — `MODEL_ID` / `STRANDS_ARGS` were blank. Compose substitutes `${VAR}` from the host environment/`.env`, so provide defaults via `.env` or `docker compose -e MODEL_ID=...`.
- **Disk-full during model load** — checkpoints are multiple GB. Ensure at least ~6 GB free, or point `hf_cache` at a larger disk by overriding the volume in `docker-compose.yml`.
- **Healthcheck failing** — the `/v1/systemone` endpoint requires a valid POST body, so the healthcheck probes the listen port directly. It may take a minute to become healthy while the model downloads/loads.

## License

Apache-2.0 — see the upstream [strands-decider](https://github.com/strands-labs/strands-decider) repository.
