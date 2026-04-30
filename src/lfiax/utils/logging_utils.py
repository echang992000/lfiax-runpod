from pathlib import Path
import csv
import wandb


class CSVLogger:
    """Lightweight CSV writer mirroring your original logic."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class WandBLogger:
    """Wrap WandB so the rest of the code depends *only* on this class."""

    def __init__(self, cfg):
        self.enabled = cfg.wandb.use_wandb
        if self.enabled:
            wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                       config=cfg)

    def log(self, data: dict):
        if self.enabled:
            wandb.log(data)

    def image(self, key: str, path: Path):
        if self.enabled:
            wandb.log({key: wandb.Image(str(path))})