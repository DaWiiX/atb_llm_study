"""
CPU-side image preprocessing for Qwen3VL Embedding models.

Pure Python implementation — does NOT depend on transformers Processor.
Takes raw NCHW uint8 image tensors as input.

Pipeline:
    image (C,H,W) uint8 → smart_resize → rescale → normalize → patch extraction
    → pixel_values (N, C*tp*p*p), grid_thw (1, 3)
"""
import math
import torch
import torch.nn.functional as F


# Qwen3-VL-Embedding-2B normalization constants (from preprocessor_config.json)
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def smart_resize(height: int, width: int,
                 factor: int = 32,
                 min_pixels: int = 256 * 256,
                 max_pixels: int = 1800 * 32 * 32) -> tuple[int, int]:
    """Resize image dimensions to be divisible by factor, within pixel limits.

    Strategy:
        1. Round to nearest multiple of factor.
        2. If area exceeds max_pixels: scale down proportionally.
        3. If area is below min_pixels: scale up proportionally.

    Returns (height, width).
    """
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(image: torch.Tensor,
                     patch_size: int = 16,
                     temporal_patch_size: int = 2,
                     merge_size: int = 2,
                     min_pixels: int = 4096,
                     max_pixels: int = 1310720) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess a single image into Qwen3VL VisionModel input format.

    Defaults match Qwen3-VL-Embedding-2B:
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), rescale=1/255.

    Args:
        image:  (C, H, W) uint8 tensor, pixel values in [0, 255].
        patch_size:          spatial patch size (default 16).
        temporal_patch_size: temporal patch size (default 2).
        merge_size:          spatial merge size in VisionModel (default 2).
        min_pixels:          minimum total pixels after resize.
        max_pixels:          maximum total pixels after resize.

    Returns:
        pixel_values: (N, C * temporal_patch_size * patch_size^2) float32.
        grid_thw:     (1, 3) LongTensor [grid_t, grid_h, grid_w].

        For single images, grid_t = 1.
    """
    image = image.float()
    _, h, w = image.shape
    factor = patch_size * merge_size

    # Resize to factor-aligned dimensions within pixel budget
    new_h, new_w = smart_resize(h, w, factor=factor,
                                min_pixels=min_pixels, max_pixels=max_pixels)
    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                          mode='bicubic', align_corners=False).squeeze(0)

    # Rescale to [0, 1] and normalize
    image = image / 255.0
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    image = (image - mean) / std

    # Pad to temporal_patch_size if needed (single image → 2 frames)
    frames = image.unsqueeze(0)  # (1, C, H, W)
    if frames.shape[0] % temporal_patch_size != 0:
        pad_n = temporal_patch_size - (frames.shape[0] % temporal_patch_size)
        frames = torch.cat([frames, frames[-1:].repeat(pad_n, 1, 1, 1)], dim=0)

    total_frames = frames.shape[0]
    C = frames.shape[1]
    grid_t = total_frames // temporal_patch_size
    grid_h = new_h // patch_size
    grid_w = new_w // patch_size

    # Patch extraction with spatial merge ordering.
    # Reshape to 9D and permute to group patches in merge_size×merge_size blocks.
    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size)

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    return flatten_patches.contiguous(), grid_thw
