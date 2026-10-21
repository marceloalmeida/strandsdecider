#!/usr/bin/env python3
"""
Custom entrypoint for strandsdecider with IPEX optimizations and quantization.
This bypasses the CLI to apply IPEX/quantization at model load time.
"""
import os
import sys
import argparse
import torch
import uvicorn

# Ensure we can import strands_decider
sys.path.insert(0, "/app")

from strands_decider.server import SystemOneEngine, EngineConfig
from strands_decider.modeling import StrandsDeciderModel


# Global flag to track IPEX availability
IPEX_AVAILABLE = False


def check_ipex_available():
    """Check if IPEX is actually usable (not just importable)."""
    global IPEX_AVAILABLE
    try:
        import intel_extension_for_pytorch as ipex
        # Try a simple operation to verify the shared library works
        test_tensor = torch.randn(2, 2)
        _ = ipex.optimize(torch.nn.Linear(2, 2), dtype=torch.float32)
        IPEX_AVAILABLE = True
        print("IPEX is available and functional")
        return True
    except Exception as e:
        print(f"IPEX not usable: {e}")
        IPEX_AVAILABLE = False
        return False


def apply_ipex_optimizations(model, dtype=torch.float32):
    """Apply Intel Extension for PyTorch optimizations to a model.

    Note: torch.compile is incompatible with PEFT/LoRA layers, so we skip compilation
    if the model has LoRA adapters. IPEX optimize works fine with LoRA.
    """
    # Check if model has LoRA (PEFT) adapters
    has_lora = hasattr(model, 'peft_config') or any(
        'lora' in type(m).__name__.lower() for m in model.modules()
    )

    if not IPEX_AVAILABLE:
        if has_lora:
            print("IPEX not available, model has LoRA - skipping torch.compile (incompatible with LoRA)")
            return model
        print("IPEX not available, falling back to torch.compile...")
        # Use torch.compile as fallback (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Applied torch.compile optimization")
        except Exception as e:
            print(f"torch.compile failed: {e}")
        return model

    try:
        import intel_extension_for_pytorch as ipex
        print("Applying IPEX optimizations...")
        model = ipex.optimize(model, dtype=dtype, inplace=True)
        print("IPEX optimizations applied successfully")
    except Exception as e:
        print(f"IPEX optimization failed: {e}")
        if has_lora:
            print("Model has LoRA - skipping torch.compile fallback (incompatible with LoRA)")
        else:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("Applied torch.compile optimization as fallback")
            except Exception as e2:
                print(f"torch.compile fallback also failed: {e2}")
    return model


def apply_torch_quantization(model, quant_mode="int8_dynamic"):
    """Apply INT8 quantization using native PyTorch quantization."""
    try:
        print(f"Applying native PyTorch {quant_mode} quantization...")

        if quant_mode == "int8_dynamic":
            # Dynamic quantization - quantize weights to int8, activations computed in fp32
            # Only quantize Linear layers, skip Embedding to avoid the float_qparams_weight_only_qconfig issue
            from torch.ao.quantization import quantize_dynamic
            model = quantize_dynamic(
                model,
                {torch.nn.Linear},  # Only quantize Linear layers
                dtype=torch.qint8,
                inplace=False
            )
            print("Dynamic INT8 quantization applied successfully (Linear layers only)")
        else:
            print(f"Static quantization not implemented in fallback, skipping")
            return model

        return model

    except Exception as e:
        print(f"Native PyTorch quantization failed: {e}, continuing without quantization")
        return model


def apply_quantization(model, quant_mode="int8_dynamic", calibration_dataloader=None):
    """Apply INT8 quantization using Intel Neural Compressor if available, else PyTorch native."""
    # Try Intel Neural Compressor first (requires IPEX backend)
    if IPEX_AVAILABLE:
        try:
            from neural_compressor import PostTrainingQuantConfig, quantize
            print(f"Applying {quant_mode} quantization with Intel Neural Compressor...")

            if quant_mode == "int8_dynamic":
                config = PostTrainingQuantConfig(
                    approach="dynamic",
                    backend="ipex",
                    dtype="int8",
                )
            elif quant_mode == "int8_static":
                if calibration_dataloader is None:
                    raise ValueError("Static quantization requires calibration_dataloader")
                config = PostTrainingQuantConfig(
                    approach="static",
                    backend="ipex",
                    dtype="int8",
                    calibration_sampling_size=[100],
                )
            else:
                raise ValueError(f"Unknown quant_mode: {quant_mode}")

            q_model = quantize(model, config, calibration_dataloader=calibration_dataloader)
            print(f"{quant_mode} quantization applied successfully")
            return q_model

        except Exception as e:
            print(f"Intel Neural Compressor failed: {e}, falling back to PyTorch native quantization")

    # Fallback to PyTorch native quantization
    return apply_torch_quantization(model, quant_mode)


def _model_has_lora(model):
    """Check if model has PEFT/LoRA adapters."""
    return hasattr(model, 'peft_config') or any(
        'lora' in type(m).__name__.lower() for m in model.modules()
    )


def load_model_with_optimizations(
    checkpoint: str,
    device: str = "cpu",
    ipex_optimize: bool = False,
    ipex_quantize: str | None = None,
    attn_implementation: str | None = None,
):
    """Load StrandsDeciderModel with optional optimizations."""
    print(f"Loading model from {checkpoint} on {device}...")

    # Check IPEX availability at startup
    if ipex_optimize or ipex_quantize:
        check_ipex_available()

    # Load the model using the standard method first
    model = StrandsDeciderModel.load(
        checkpoint,
        device_map=device if device != "cpu" else None,
        attn_implementation=attn_implementation,
    )

    # Move to device
    model = model.to(device)

    # Check for LoRA
    has_lora = _model_has_lora(model)
    if has_lora:
        print("Model has LoRA adapters - quantization will be skipped (incompatible)")

    # Apply optimizations to the torso (the main model body)
    if ipex_optimize:
        try:
            model.torso = apply_ipex_optimizations(model.torso)
        except Exception as e:
            print(f"Failed to optimize torso: {e}")

    # Apply quantization (skip if LoRA present)
    if ipex_quantize:
        if has_lora:
            print("Skipping quantization - incompatible with LoRA adapters")
        else:
            try:
                model.torso = apply_quantization(model.torso, ipex_quantize)
            except Exception as e:
                print(f"Quantization failed: {e}, continuing without quantization")

    # Also optimize head if possible - skip torch.compile for head (can have LoRA issues)
    if ipex_optimize:
        try:
            if IPEX_AVAILABLE:
                import intel_extension_for_pytorch as ipex
                print("Applying IPEX optimizations to head...")
                model.head = ipex.optimize(model.head, dtype=torch.float32, inplace=True)
                print("IPEX optimizations applied to head successfully")
            else:
                print("IPEX not available, skipping head optimization (torch.compile incompatible with LoRA)")
        except Exception as e:
            print(f"Failed to optimize head: {e}")

    print(f"Model loaded successfully on {device}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Serve strands-decider with IPEX/quantization")
    parser.add_argument("checkpoint", help="Model checkpoint path or HF repo ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu|cuda|mps)")
    parser.add_argument("--ipex-optimize", action="store_true", help="Enable IPEX graph optimizations")
    parser.add_argument("--ipex-quantize", choices=["int8_dynamic", "int8_static"], help="INT8 quantization mode")
    parser.add_argument("--model-name", help="Model name for responses")
    parser.add_argument("--attn-implementation", help="Attention implementation (eager|sdpa|flash_attention_2)")
    parser.add_argument("--no-prefix-cache", action="store_true", help="Disable prefix cache")

    args = parser.parse_args()

    # Load model with optimizations
    model = load_model_with_optimizations(
        checkpoint=args.checkpoint,
        device=args.device,
        ipex_optimize=args.ipex_optimize,
        ipex_quantize=args.ipex_quantize,
        attn_implementation=args.attn_implementation,
    )

    # Create engine with the pre-loaded model
    resolved_name = args.model_name or os.path.basename(args.checkpoint.rstrip("/")) or args.checkpoint
    engine = SystemOneEngine(
        model,
        EngineConfig(
            device=args.device,
            use_prefix_cache=not args.no_prefix_cache,
            model_name=resolved_name,
        ),
    )

    # Create FastAPI app with pre-loaded engine
    from strands_decider.server import FastAPI, JSONResponse, HTTPException, SystemOneRequest, SystemOneResponse
    import time

    # We need to set the global engine
    import strands_decider.server as server_module
    server_module._engine = engine

    app = FastAPI(
        title="strands-decider System One (Optimized)",
        version="0.1.0",
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    @app.get("/health")
    def health() -> dict:
        eng = server_module.get_engine()
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": args.checkpoint,
            "base_model": eng.model.config.base_model,
            "num_slots": eng.model.config.num_slots,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = server_module.get_engine()
        started = time.perf_counter()
        try:
            response = eng.evaluate(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    print(f"Starting server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()