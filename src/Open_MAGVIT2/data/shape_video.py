import hashlib
import json
import os
import random
import time
from enum import Enum
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from shapekit import Random2DShapeCreator, Random3DShapeCreator, Scene, SceneType
from torch.utils.data import Dataset
from torchvision import transforms


class TemporalPattern(Enum):
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    OSCILLATION = "oscillation"
    INTERRUPTION = "interruption"


def validate_video_sequence_length(sequence_length: int) -> None:
    """Open-MAGVIT2's causal 3D encoder/decoder requires T ≡ 1 (mod 4)."""
    if sequence_length % 4 != 1:
        nearest = max(1, round((sequence_length - 1) / 4) * 4 + 1)
        raise ValueError(
            f"sequence_length={sequence_length} is incompatible with the video "
            f"tokenizer (encoder/decoder output length must match input). "
            f"Use T where T % 4 == 1, e.g. {nearest} instead of {sequence_length}."
        )


class ShapeVideoDataset(Dataset):
    """
    Synthetic rotating 2D/3D shape sequences rendered with shapekit.

    Returns dicts compatible with Open-MAGVIT2 video training:
      video: (C, T, H, W) in [-1, 1]
      label: int (unused placeholder)
    """

    SCALE_FACTOR = 1000

    def __init__(self, config=None):
        super().__init__()
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        self.mode = self.config.get("mode", "train")
        self.sequence_length = self.config["sequence_length"]
        validate_video_sequence_length(self.sequence_length)
        self.resolution = self.config["size"]
        self.dataset_size = self.config.get("dataset_size", 10000)
        self.cache = self.config.get("cache", True)
        self.grayscale = self.config.get("shape_grayscale", False)

        self.scene_type = SceneType[self.config.get("shape_scene_type", "DIM_3")]
        self.min_cubes = self.config.get("shape_min_cubes", 2)
        self.max_cubes = self.config.get("shape_max_cubes", 6)
        self.angle_min = self.config.get("shape_angle_min", 15)
        self.angle_max = self.config.get("shape_angle_max", 45)

        pattern_names = self.config.get("shape_temporal_patterns", [])
        self.temporal_patterns = [TemporalPattern(p) for p in pattern_names]
        self.pattern_combining = self.config.get("shape_pattern_combining", False)
        self.accel_min = self.config.get("shape_accel_min", 3)
        self.accel_max = self.config.get("shape_accel_max", 6)
        self.oscillation_period_min = self.config.get("shape_oscillation_period_min", 1)
        self.oscillation_period_max = self.config.get("shape_oscillation_period_max", 4)
        self.interruption_period_min = self.config.get("shape_interruption_period_min", 1)
        self.interruption_period_max = self.config.get("shape_interruption_period_max", 4)

        self.render_size = self.resolution
        self.transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
        ])

        cache_root = self.config.get("shape_cache_dir", "../../data/shape_cache")
        config_hash = self._config_hash()
        self.cache_dir = os.path.join(cache_root, config_hash)
        self._samples = None

        if not self.cache:
            self._samples = self._generate_in_memory()
        elif self._cache_valid():
            print(
                f"[ShapeVideoDataset:{self.mode}] Using cached dataset at "
                f"{self.cache_dir} ({self.dataset_size} samples)"
            )
        else:
            self._generate_and_save()

    def _config_hash(self):
        cfg = dict(
            mode=self.mode,
            size=self.dataset_size,
            sequence_length=self.sequence_length,
            resolution=self.resolution,
            scene_type=self.scene_type.name,
            min_cubes=self.min_cubes,
            max_cubes=self.max_cubes,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            patterns=[p.value for p in self.temporal_patterns],
            pattern_combining=self.pattern_combining,
            accel_min=self.accel_min,
            accel_max=self.accel_max,
            osc_min=self.oscillation_period_min,
            osc_max=self.oscillation_period_max,
            int_min=self.interruption_period_min,
            int_max=self.interruption_period_max,
            render_size=self.render_size,
            grayscale=self.grayscale,
        )
        return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

    def _cache_valid(self):
        meta_path = os.path.join(self.cache_dir, "meta.json")
        if not os.path.isfile(meta_path):
            return False
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("size") == self.dataset_size and meta.get("done", False)

    def _build_scene(self, is_2d, creator_2d, creator_3d):
        if is_2d:
            init_fig = creator_2d.create_equilateral_triangle()
        else:
            init_fig, _ = creator_3d.create_connected_cubes(self.min_cubes)

        scene = Scene(
            init_fig,
            self.scene_type,
            self.render_size,
            bg_color="white",
            mesh_color="black" if is_2d else "gray",
            show_edges=False,
            lighting=not is_2d,
            line_width=4.0,
            distance_factor=1.0 if is_2d else 2.5,
            fixed_camera_distance=2.7 if is_2d else None,
            axis="z",
        )
        scene.plotter.render()
        return scene

    def _generate_and_save(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        print(
            f"[ShapeVideoDataset:{self.mode}] Generating {self.dataset_size} samples "
            f"to {self.cache_dir} ..."
        )

        is_2d = self.scene_type == SceneType.DIM_2
        creator_2d = Random2DShapeCreator()
        creator_3d = None if is_2d else Random3DShapeCreator(self.max_cubes, include_reflections=False)
        scene = self._build_scene(is_2d, creator_2d, creator_3d)

        t0 = time.time()
        for idx in range(self.dataset_size):
            frames = self._render_with_scene(scene, creator_2d, creator_3d, is_2d)
            np.save(os.path.join(self.cache_dir, f"{idx}.npy"), frames)
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = (self.dataset_size - idx - 1) / rate
                print(f"  [{idx + 1}/{self.dataset_size}] {rate:.1f} samples/s, ~{remaining:.0f}s remaining")

        scene.plotter.close()
        with open(os.path.join(self.cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"size": self.dataset_size, "done": True}, f)
        elapsed = time.time() - t0
        print(
            f"[ShapeVideoDataset:{self.mode}] Done in {elapsed:.1f}s "
            f"({self.dataset_size / elapsed:.1f} samples/s)"
        )

    def _generate_in_memory(self):
        is_2d = self.scene_type == SceneType.DIM_2
        creator_2d = Random2DShapeCreator()
        creator_3d = None if is_2d else Random3DShapeCreator(self.max_cubes, include_reflections=False)
        scene = self._build_scene(is_2d, creator_2d, creator_3d)
        samples = [
            self._render_with_scene(scene, creator_2d, creator_3d, is_2d)
            for _ in range(self.dataset_size)
        ]
        scene.plotter.close()
        return samples

    def _render_with_scene(self, scene, creator_2d, creator_3d, is_2d):
        if is_2d:
            base = random.randint(self._scale(0.5), self._scale(1)) / self.SCALE_FACTOR
            shift = random.randint(self._scale(-0.2), self._scale(base + 0.2)) / self.SCALE_FACTOR
            height = random.randint(self._scale(0.5), self._scale(1)) / self.SCALE_FACTOR
            figure = creator_2d.create_triangle(base, shift, height)
        else:
            num_blocks = random.randint(self.min_cubes, self.max_cubes)
            figure, _ = creator_3d.create_connected_cubes(num_blocks)

        scene.prepare_scene(
            figure,
            bg_color="white",
            mesh_color="black" if is_2d else "gray",
            lighting=not is_2d,
            show_edges=False,
            line_width=4.0,
            distance_factor=1.0 if is_2d else 2.5,
            fixed_camera_distance=2.7 if is_2d else None,
            axis="z",
        )
        scene.plotter.render()

        step = np.zeros(3, dtype=float)
        if is_2d:
            step[2] = self._angle_gen()
        else:
            step[2] = self._angle_gen()

        selected_patterns = self._select_patterns()
        acceleration = 1.0
        oscillation_period = 0
        interruption_period = 0
        step_swap = np.array([0.0, 0.0, 0.0])

        if TemporalPattern.OSCILLATION in selected_patterns:
            oscillation_period = random.randint(
                self.oscillation_period_min, self.oscillation_period_max
            )
        elif TemporalPattern.INTERRUPTION in selected_patterns:
            interruption_period = random.randint(
                self.interruption_period_min, self.interruption_period_max
            )

        if TemporalPattern.ACCELERATION in selected_patterns:
            step /= 1.5
            acceleration = 1.0 + random.randint(
                self._scale(self.accel_min / 2), self._scale(self.accel_max / 2)
            ) / (self.SCALE_FACTOR * 100)
        elif TemporalPattern.DECELERATION in selected_patterns:
            step *= 1.5
            dec = 1.0 + random.randint(
                self._scale(self.accel_min * 2), self._scale(self.accel_max * 2)
            ) / (self.SCALE_FACTOR * 100)
            acceleration = 1.0 / dec

        frames = np.empty(
            (self.sequence_length, self.render_size, self.render_size, 3), dtype=np.uint8
        )
        for i in range(self.sequence_length):
            if step[0] != 0.0:
                figure.rotate_x(step[0], point=scene.center_of_mass, inplace=True)
            if step[1] != 0.0:
                figure.rotate_y(step[1], point=scene.center_of_mass, inplace=True)
            if step[2] != 0.0:
                figure.rotate_z(step[2], point=scene.center_of_mass, inplace=True)
            scene.plotter.render()
            frames[i] = np.array(scene.plotter.screenshot())

            if TemporalPattern.OSCILLATION in selected_patterns and oscillation_period > 0:
                if (i + 1) % oscillation_period == 0:
                    step = step * -1.0
            elif TemporalPattern.INTERRUPTION in selected_patterns and interruption_period > 0:
                if (i + 1) % interruption_period == 0:
                    step, step_swap = step_swap, step

            if (
                TemporalPattern.ACCELERATION in selected_patterns
                or TemporalPattern.DECELERATION in selected_patterns
            ):
                step = step * acceleration

        return frames

    @staticmethod
    def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        return np.round(gray).astype(np.uint8)

    def _frames_to_video(self, frames_uint8):
        frame_tensors = []
        for i in range(frames_uint8.shape[0]):
            frame = frames_uint8[i]
            if self.grayscale:
                frame = self._rgb2gray(frame)
                pil_image = Image.fromarray(frame, mode="L")
                tensor = self.transform(pil_image)
                tensor = tensor.repeat(3, 1, 1)
            else:
                pil_image = Image.fromarray(frame)
                tensor = self.transform(pil_image)
            frame_tensors.append(tensor)

        # (T, C, H, W) -> (C, T, H, W), scale to [-1, 1] like UCF101
        video = torch.stack(frame_tensors).permute(1, 0, 2, 3).contiguous()
        return video * 2.0 - 1.0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self._samples is not None:
            frames_uint8 = self._samples[idx]
        else:
            frames_uint8 = np.load(os.path.join(self.cache_dir, f"{idx}.npy"))
        return dict(video=self._frames_to_video(frames_uint8), label=0)

    def _scale(self, v):
        return int(v * self.SCALE_FACTOR)

    def _angle_gen(self):
        return random.choice((-1, 1)) * random.randint(
            self._scale(self.angle_min), self._scale(self.angle_max)
        ) / self.SCALE_FACTOR

    def _select_patterns(self):
        if not self.temporal_patterns:
            return []
        selected = self.temporal_patterns.copy()
        if TemporalPattern.ACCELERATION in selected and TemporalPattern.DECELERATION in selected:
            selected.remove(random.choice((TemporalPattern.ACCELERATION, TemporalPattern.DECELERATION)))
        if TemporalPattern.OSCILLATION in selected and TemporalPattern.INTERRUPTION in selected:
            selected.remove(random.choice((TemporalPattern.OSCILLATION, TemporalPattern.INTERRUPTION)))
        if selected:
            k = random.randint(1, len(selected) if self.pattern_combining else 1)
            selected = list(np.random.choice(selected, size=k, replace=False))
        return selected


def build_shape_hparams(config):
    """Utility for scripts that expect hierarchical-vq-transformers-style hparams."""
    if not isinstance(config, dict):
        config = OmegaConf.to_container(config)
    return SimpleNamespace(
        context_length=config["sequence_length"],
        image_dims=[config["size"], config["size"]],
        shape_scene_type=config.get("shape_scene_type", "DIM_3"),
        shape_min_cubes=config.get("shape_min_cubes", 2),
        shape_max_cubes=config.get("shape_max_cubes", 6),
        shape_angle_min=config.get("shape_angle_min", 15),
        shape_angle_max=config.get("shape_angle_max", 45),
        shape_temporal_patterns=config.get("shape_temporal_patterns", []),
        shape_pattern_combining=config.get("shape_pattern_combining", False),
        shape_accel_min=config.get("shape_accel_min", 3),
        shape_accel_max=config.get("shape_accel_max", 6),
        shape_oscillation_period_min=config.get("shape_oscillation_period_min", 1),
        shape_oscillation_period_max=config.get("shape_oscillation_period_max", 4),
        shape_interruption_period_min=config.get("shape_interruption_period_min", 1),
        shape_interruption_period_max=config.get("shape_interruption_period_max", 4),
        shape_cache_dir=config.get("shape_cache_dir", "../../data/shape_cache"),
        shape_no_imagenet_norm=True,
        shape_grayscale=config.get("shape_grayscale", False),
    )
