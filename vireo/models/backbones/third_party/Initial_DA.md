# Initial Modifications for Depth Anything

## Depth Anything V2

In **Depth-Anything-V2/depth_anything_v2/dinov2.py**, locate the DINOv2 function and replace its definition with the following version (changing img_size to 512 and patch_size to 16):
```
def DINOv2(model_name):
  model_zoo = {
    "vits": vit_small,
    "vitb": vit_base,
    "vitl": vit_large,
    "vitg": vit_giant2,
	}

  return model_zoo[model_name](
	img_size=512,
	patch_size=16,
	init_values=1.0,
	ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
	block_chunks=0,
	num_register_tokens=0,
	interpolate_antialias=False,
	interpolate_offset=0.1,
  )
```

## PromptDA

In **PromptDA/promptda/promptda.py**, open the `PromptDA` class and change its `patch_size` parameter to `16`.

Then in **PromptDA/torchhub/facebookresearch_dinov2_main/hubconf.py**, locate the `_make_dinov2_model` function signature and update the defaults from
```
img_size: int = 518,
patch_size: int = 14,
```
to
```
img_size: int = 512,
patch_size: int = 16,
```