import os
from contextlib import nullcontext

import lightning as L

from src.Open_MAGVIT2.utils.video_viz import predictor_rows_to_grid


class PredictorHorizonLogger(L.Callback):
    """Save horizon PNG grids on validation epochs (predictor decode paths)."""

    def __init__(
        self,
        every_n_epochs: int = 1,
        max_samples: int = 4,
        log_wandb: bool = True,
        save_dir: str = "predictions",
        cell_size: int = 96,
    ):
        super().__init__()
        self.every_n_epochs = max(1, every_n_epochs)
        self.max_samples = max_samples
        self.log_wandb = log_wandb
        self.save_dir = save_dir
        self.cell_size = cell_size
        self._fixed_batch = None

    def _get_fixed_batch(self, trainer, pl_module):
        if self._fixed_batch is not None:
            return self._fixed_batch
        loader = trainer.datamodule.val_dataloader()
        self._fixed_batch = next(iter(loader))
        return self._fixed_batch

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return

        batch = self._get_fixed_batch(trainer, pl_module)
        pl_module.eval()
        cm = nullcontext()

        with cm:
            log = pl_module.log_images(batch)

        n = min(self.max_samples, log["inputs"].shape[0])
        out_dir = os.path.join(trainer.default_root_dir, self.save_dir)
        os.makedirs(out_dir, exist_ok=True)

        wandb_log = {}
        t_context = getattr(pl_module, "t_context", 0)
        t_total = getattr(pl_module, "t_total", log["inputs"].shape[2])

        for b in range(n):
            sample_log = {k: v[b : b + 1] for k, v in log.items()}
            png = predictor_rows_to_grid(
                sample_log,
                t_context=t_context,
                t_total=t_total,
                cell_size=self.cell_size,
            )
            path = os.path.join(
                out_dir, f"epoch{trainer.current_epoch:04d}_sample{b}.png"
            )
            png.save(path)

            if self.log_wandb and self._log_to_wandb(trainer):
                try:
                    import wandb
                    wandb_log[f"pred/sample{b}"] = wandb.Image(path)
                except ImportError:
                    pass

            if self._log_to_tensorboard(trainer):
                import numpy as np
                arr = np.array(png).transpose(2, 0, 1)
                trainer.logger.experiment.add_image(
                    f"pred/sample{b}", arr, global_step=trainer.global_step
                )

        if wandb_log and self.log_wandb and self._log_to_wandb(trainer):
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
