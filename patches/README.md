# Patches applied to the pinned diffusers checkout

`scripts/env.sh` puts `diffusers-src/` on `PYTHONPATH`, so these edits live in that
checkout and are **not** part of the installed diffusers. They are recorded here because
`diffusers-src/` is git-ignored (it is a full clone of upstream), and a re-clone would
silently lose them.

Apply with:

```bash
cd diffusers-src
git apply ../patches/0001-qwenimage21-image-pad-mask.patch
```

---

## 0001 — `QwenImage21Pipeline.__call__` cannot accept a precomputed `image_pad_mask`

**Symptom** (image editing only; text-to-image is unaffected):

```
ValueError: Pass `image_pad_mask` alongside `prompt_embeds` when the embeddings cover
condition images, so the transformer knows which positions hold image tokens.
```

**Cause.** `encode_prompt` requires `image_pad_mask` when `prompt_embeds` is supplied *and*
`image` is not None. But `__call__` has no such parameter, so there is **no way to satisfy
that requirement** when passing precomputed embeddings for an edit.

This matters for this project's memory strategy specifically: we encode the prompt (and the
reference, via the VLM) on CPU, free the 16.33 GiB text encoder, and only then call the
pipeline. That flow *must* pass `prompt_embeds`, so it necessarily hits this error.

Note the asymmetry: for text-to-image a precomputed-embeddings call is fine, because
`encode_prompt` then synthesises an all-False `image_pad_mask` itself. Only the
image-conditioned path is unreachable.

**Fix.** Add an optional `image_pad_mask` parameter to `__call__` and forward it, matching
the existing pattern for the other precomputed tensors (`prompt_embeds`,
`prompt_embeds_mask`, `negative_prompt_embeds`, `negative_prompt_embeds_mask`).

Two lines changed:

```diff
         prompt_embeds: torch.Tensor | None = None,
         prompt_embeds_mask: torch.Tensor | None = None,
+        image_pad_mask: torch.Tensor | None = None,
         negative_prompt_embeds: torch.Tensor | None = None,
```
```diff
             prompt_embeds=prompt_embeds,
             prompt_embeds_mask=prompt_embeds_mask,
+            image_pad_mask=image_pad_mask,
```

**Upstream status.** Not reported here; this was found locally. Worth filing upstream —
the parameter exists one level down and is simply not plumbed through.
