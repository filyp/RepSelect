"""Quantization attack: quantize the (unlearned) model's weights and dequantize back.

Unlearning updates that are small relative to the quantization step can be rounded
away, restoring the original behaviour (Zhang et al., 2024, "Catastrophic Failure of
LLM Unlearning via Quantization"). We apply the quantization error in place and keep
bf16 compute, so the model stays an ordinary HF model that the Trainer and all
evaluators accept; only the weight values change.

  nf4:  bitsandbytes 4-bit NormalFloat, blocksize 64 (the BitsAndBytesConfig default)
  int8: symmetric per-output-row absmax int8 (LLM.int8 without outlier decomposition)

All nn.Linear layers except lm_head, which bitsandbytes also leaves unquantized.
"""

import logging

import torch as pt

logger = logging.getLogger(__name__)


@pt.no_grad()
def quantize_roundtrip(model, kind):
    import bitsandbytes.functional as bnbF

    n = 0
    for name, module in model.named_modules():
        if not isinstance(module, pt.nn.Linear) or "lm_head" in name:
            continue
        w = module.weight.data
        wc = w.to("cuda")
        if kind == "nf4":
            q, state = bnbF.quantize_4bit(wc, quant_type="nf4", blocksize=64)
            wq = bnbF.dequantize_4bit(q, state)
        elif kind == "int8":
            wf = wc.float()
            scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
            wq = (wf / scale).round().clamp(-127, 127) * scale
        else:
            raise ValueError(f"unknown quantization kind: {kind}")
        module.weight.data = wq.to(device=w.device, dtype=w.dtype)
        n += 1
    logger.info(f"quantize_roundtrip({kind}): {n} linear layers")
