import os
from typing import Optional

import lightning as L

from src.Open_MAGVIT2.utils.video_viz import make_comparison_grid, videos_to_row_dict


class VideoReconstructionLogger(L.Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        max_samples: int = 2,
        log_wandb: bool = True,
        save_dir: str = "reconstructions",
        cell_size: int = 96,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.log_wandb = log_wandb
        self.save_dir = save_dir
        self.cell_size = cell_size
        self._fixed_batch = None

    def _get_fixed_batch(self, trainer, pl_module):
        if self._fixed_batch is not None:
            return self._fixed_batch
        loader = trainer.datamodule.val_dataloader()
        batch = next(iter(loader))
        if isinstance(batch, dict):
            self._fixed_batch = batch
        else:
            self._fixed_batch = batch
        return self._fixed_batch

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return

        batch = self._get_fixed_batch(trainer, pl_module)
        pl_module.eval()
        if getattr(pl_module, "use_ema", False):
            cm = pl_module.ema_scope()
        else:
            from contextlib import nullcontext
            cm = nullcontext()

        with cm:
            log = pl_module.log_images(batch)

        x = log["inputs"].detach()
        n = min(self.max_samples, x.shape[0])
        out_dir = os.path.join(trainer.default_root_dir, self.save_dir)
        os.makedirs(out_dir, exist_ok=True)

        wandb_log = {}
        for b in range(n):
            progressive = {}
            for k, v in log.items():
                if k.startswith("progressive_"):
                    progressive[k] = v[b]
            rows = videos_to_row_dict(x[b], log["reconstructions"][b], progressive or None)
            png = make_comparison_grid(rows, cell_size=self.cell_size)
            path = os.path.join(
                out_dir, f"epoch{trainer.current_epoch:04d}_sample{b}.png"
            )
            png.save(path)

            if self._log_to_wandb(trainer):
                try:
                    import wandb
                    wandb_log[f"recon/sample{b}"] = wandb.Image(path)
                except ImportError:
                    pass

            if self._log_to_tensorboard(trainer):
                import numpy as np
                arr = np.array(png).transpose(2, 0, 1)
                trainer.logger.experiment.add_image(
                    f"recon/sample{b}", arr, global_step=trainer.global_step
                )

        if wandb_log and self._log_to_wandb(trainer):
            import wandb
            wandb.log(wandb_log, step=trainer.global_step)

    @staticmethod
    def _log_to_wandb(trainer) -> bool:
        if trainer.logger is None:
            return False
        return trainer.logger.__class__.__name__ == "WandbLogger"

    @staticmethod
    def _log_to_tensorboard(trainer) -> bool:
        if trainer.logger is None:
            return False
        name = trainer.logger.__class__.__name__
        return name == "TensorBoardLogger"
