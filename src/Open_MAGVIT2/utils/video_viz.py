from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _tensor_to_frame_uint8(frame: torch.Tensor) -> np.ndarray:
    """frame: (3, H, W) in [-1, 1] -> uint8 HWC"""
    x = frame.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
    return x


def make_comparison_grid(
    rows: List[Tuple[str, torch.Tensor]],
    frame_indices: List[int] = None,
    cell_size: int = 128,
    label_width: int = 140,
    header_height: int = 28,
) -> Image.Image:
    """
    Build PNG grid: rows = [(name, video(C,T,H,W)), ...], columns = frames.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    video = rows[0][1]
    if video.ndim == 5:
        _, _, t_len, _, _ = video.shape
    elif video.ndim == 4:
        _, t_len, _, _ = video.shape
    else:
        raise ValueError(f"Expected video (C,T,H,W) or (B,C,T,H,W), got shape {tuple(video.shape)}")
    if frame_indices is None:
        frame_indices = list(range(t_len))

    n_cols = len(frame_indices)
    n_rows = len(rows)
    grid_w = label_width + n_cols * cell_size
    grid_h = header_height + n_rows * cell_size
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    for col, t_idx in enumerate(frame_indices):
        x0 = label_width + col * cell_size + cell_size // 2 - 16
        draw.text((x0, 4), f"t={t_idx}", fill=(0, 0, 0), font=font_sm)

    for row_idx, (name, video) in enumerate(rows):
        y0 = header_height + row_idx * cell_size
        draw.text((8, y0 + cell_size // 2 - 8), name, fill=(0, 0, 0), font=font)
        for col, t_idx in enumerate(frame_indices):
            vid = rows[row_idx][1]
            frame = vid[:, t_idx] if vid.ndim == 4 else vid[0, :, t_idx]
            arr = _tensor_to_frame_uint8(frame)
            img = Image.fromarray(arr).resize((cell_size, cell_size), Image.BILINEAR)
            canvas.paste(img, (label_width + col * cell_size, y0))

    return canvas


def predictor_rows_to_grid(
    log: Dict[str, torch.Tensor],
    t_context: int,
    t_total: int,
    cell_size: int = 128,
    label_width: int = 140,
    header_height: int = 28,
) -> Image.Image:
    """Horizon-focused grid for predictor decode paths (one batch index)."""
    rows = [
        ("ground_truth", log["inputs"]),
        ("vae_recon", log["vae_recon"]),
        ("pred_train", log["pred_train"]),
        ("pred_parallel", log["pred_parallel"]),
        ("pred_ar", log["pred_ar"]),
    ]
    frame_indices = list(range(t_context, t_total))
    sliced = []
    for name, video in rows:
        if video.ndim == 5:
            video = video[0]
        sliced.append((name, video))
    return make_comparison_grid(
        sliced,
        frame_indices=frame_indices,
        cell_size=cell_size,
        label_width=label_width,
        header_height=header_height,
    )


def videos_to_row_dict(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    progressive: Dict[str, torch.Tensor] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Build row list for one batch index (caller slices batch dim)."""
    rows = [
        ("ground_truth", inputs),
        ("reconstruction", reconstructions),
    ]
    if progressive:
        for key in sorted(progressive.keys()):
            rows.append((key, progressive[key]))
    return rows
