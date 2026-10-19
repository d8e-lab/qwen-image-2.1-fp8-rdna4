#!/usr/bin/env python3
"""Persist quantised fp8 weights to disk, so a run does not re-quantise every time.

Why this needs its own serializer
=================================
`Fp8Weight` deliberately stores its payload in **non-persistent** buffers (`qweight`,
`scale`, `bias`) and is *not* an `nn.Linear` subclass, so `save_pretrained` writes only the
empty bf16-shaped placeholder and a reload silently reconstructs empty `nn.Linear` layers.
That is not a hypothetical: an earlier revision of `scripts/run_qwen_image.py` did exactly
that, and the only thing that caught it was a layer-count assertion (NOTES.md, phase 1).

So the fp8 payload is written as its own safetensors file, keyed by module path, plus a small
manifest that says which modules were quantised and with what shapes. Loading walks the
manifest, finds the matching `nn.Linear`, and swaps in an `Fp8Weight` holding the bytes.

What it buys
============
Quantising the 7.1B image transformer takes 34-37 s of CPU on this box. The resulting fp8
bytes are 6.63 GiB, and reading 6.63 GiB back takes ~4 s warm. So a run that hits the cache
saves ~30 s, which matters for the one-shot CLI where the whole render is ~3 min.

It also makes the quantised artefact inspectable and reproducible, and (unlike the bf16
checkpoint, which is 13.25 GiB) small enough to keep next to the model.

Layout
======
    <cache-dir>/
        fp8_weights.safetensors    qweight (uint8) / scale (fp32) / bias, flat keys
        fp8_manifest.json          version, source fingerprint, per-module shapes
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn

from fp8_quant import FP8_MAX, Fp8Weight  # noqa: F401  (FP8_MAX documented the format)

VERSION = 1
WEIGHTS = "fp8_weights.safetensors"
MANIFEST = "fp8_manifest.json"


def _walk_fp8(model: nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, Fp8Weight) and name:
            yield name, mod


def fingerprint(model_dir: str) -> dict:
    """Cheap identity for the checkpoint this cache was built from.

    A full hash of 13 GiB of safetensors would cost more than the quantisation it guards, so
    this uses the transformer sub-config (layer counts, widths) plus the shard file sizes. It
    catches "wrong model / half-downloaded / re-quantised with different settings", which is
    what actually goes wrong in practice.
    """
    cfg_path = os.path.join(model_dir, "transformer", "config.json")
    entry = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            entry["config"] = json.load(fh)
    sizes = {}
    for root, _dirs, files in os.walk(os.path.join(model_dir, "transformer")):
        for f in sorted(files):
            if f.endswith(".safetensors"):
                sizes[f] = os.path.getsize(os.path.join(root, f))
    entry["shards"] = sizes
    return entry


def exists(cache_dir: str) -> bool:
    return os.path.isfile(os.path.join(cache_dir, WEIGHTS)) and \
        os.path.isfile(os.path.join(cache_dir, MANIFEST))


def save_fp8(model: nn.Module, cache_dir: str, model_dir: str | None = None,
             settings: dict | None = None) -> dict:
    """Write every Fp8Weight in ``model`` to ``cache_dir``. Returns a small report."""
    from safetensors.torch import save_file

    layers = list(_walk_fp8(model))
    if not layers:
        raise ValueError("save_fp8: model has no Fp8Weight modules - was it quantised?")
    os.makedirs(cache_dir, exist_ok=True)

    flat: dict[str, torch.Tensor] = {}
    manifest_layers = []
    for path, mod in layers:
        q = mod.qweight.detach().to("cpu")
        if q.dtype != torch.uint8:
            q = q.view(torch.uint8)
        s = mod.scale.detach().float().to("cpu")
        flat[f"{path}.qweight"] = q.contiguous()
        flat[f"{path}.scale"] = s.contiguous()
        entry = {"path": path, "out_features": int(q.shape[0]),
                 "in_features": int(q.shape[1]), "bias": mod.bias is not None}
        if mod.bias is not None:
            b = mod.bias.detach().to("cpu").contiguous()
            flat[f"{path}.bias"] = b
            entry["bias_dtype"] = str(b.dtype).split(".")[-1]
        manifest_layers.append(entry)

    t0 = time.perf_counter()
    save_file(flat, os.path.join(cache_dir, WEIGHTS), metadata={"version": str(VERSION)})
    manifest = {
        "version": VERSION,
        "created": time.strftime("%F %T"),
        "source": fingerprint(model_dir) if model_dir else {},
        "settings": settings or {},
        "layers": manifest_layers,
        "bytes": sum(t.numel() * t.element_size() for t in flat.values()),
    }
    tmp = os.path.join(cache_dir, MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, os.path.join(cache_dir, MANIFEST))

    gib = manifest["bytes"] / 2 ** 30
    print(f"[fp8-ckpt] saved {len(layers)} layers, {gib:.2f} GiB -> {cache_dir} "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)
    return manifest


def load_fp8(model: nn.Module, cache_dir: str, model_dir: str | None = None,
             cache_dequant: bool = False, settings: dict | None = None) -> dict:
    """Replace the quantisable ``nn.Linear`` modules of ``model`` with the cached Fp8Weights.

    The model is expected to be the *unquantised* module graph (freshly loaded on CPU), so the
    layer paths line up with what was saved. Shape mismatches raise instead of silently
    producing a model with holes in it.
    """
    from safetensors import safe_open

    if not exists(cache_dir):
        raise FileNotFoundError(f"no fp8 cache in {cache_dir}")
    with open(os.path.join(cache_dir, MANIFEST), encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("version") != VERSION:
        raise ValueError(f"fp8 cache version {manifest.get('version')} != {VERSION}")
    if model_dir:
        want = fingerprint(model_dir)
        if manifest.get("source") and manifest["source"] != want:
            raise ValueError("fp8 cache was built from a different checkpoint")
    if settings is not None and manifest.get("settings") != settings:
        raise ValueError(f"fp8 cache was built with {manifest.get('settings')}, "
                         f"not {settings}")

    t0 = time.perf_counter()
    replaced = 0
    with safe_open(os.path.join(cache_dir, WEIGHTS), framework="pt") as fh:
        keys = set(fh.keys())
        by_path = {m["path"]: m for m in manifest["layers"]}
        for path, meta in by_path.items():
            parent_name, _, child_name = path.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            child = getattr(parent, child_name, None)
            if not isinstance(child, nn.Linear):
                raise ValueError(f"{path} is {type(child).__name__}, expected nn.Linear")
            if (child.weight.shape[0], child.weight.shape[1]) != \
                    (meta["out_features"], meta["in_features"]):
                raise ValueError(f"{path}: shape {tuple(child.weight.shape)} != manifest "
                                 f"{(meta['out_features'], meta['in_features'])}")
            q = fh.get_tensor(f"{path}.qweight")
            s = fh.get_tensor(f"{path}.scale")
            b = fh.get_tensor(f"{path}.bias") if meta.get("bias") else None
            if q.dtype == torch.uint8:
                q = q.view(torch.uint8)
            new = Fp8Weight(q, s, b, cache_dequant=cache_dequant)
            new = new.to(device=child.weight.device)
            setattr(parent, child_name, new)
            replaced += 1
            del child
    if replaced != len(manifest["layers"]):
        raise ValueError(f"loaded {replaced} layers, manifest lists {len(manifest['layers'])}")
    dt = time.perf_counter() - t0
    gib = manifest["bytes"] / 2 ** 30
    print(f"[fp8-ckpt] loaded {replaced} layers, {gib:.2f} GiB from {cache_dir} "
          f"in {dt:.1f}s ({gib/dt:.2f} GiB/s)", flush=True)
    return {"layers": replaced, "seconds": dt, "GiB": gib}
