"""
日志工具 — wandb / 控制台
"""
from typing import Optional


class Logger:
    """统一日志接口"""

    def __init__(self, use_wandb: bool = False, project: str = "xinhe", run_name: Optional[str] = None):
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=project, name=run_name)

    def log(self, metrics: dict, step: int):
        """记录指标"""
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self):
        if self.use_wandb:
            import wandb
            wandb.finish()
