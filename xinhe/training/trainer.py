"""
Trainer — 训练循环

核心特性:
- state 跨 segment (对话轮次) 传递
- 截断 BPTT: 每 tbptt_steps 轮做 detach + backward + step
- 只训练 StatePlugin + LoRA 参数
"""
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.xinhe_model import XinheModel
from ..model.config import XinheConfig


class Trainer:
    """
    心核训练器。

    episode 循环:
        for each episode (多轮对话):
            state = model.init_state()
            for seg_idx, segment in enumerate(episode):
                1. forward(segment, state) → logits, state_next
                2. 累积 loss
                3. 每 tbptt_steps 做一次 backward + optimizer step + state.detach()
                state = state_next
    """

    def __init__(
        self,
        model: XinheModel,
        config: XinheConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # 设备和精度
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype, torch.float32)

        # 只优化可训练参数 (StatePlugin + LoRA)
        trainable_params = model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 学习率调度: cosine with warmup
        self.scheduler = self._build_scheduler()

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _build_scheduler(self):
        """Cosine schedule with linear warmup"""
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(max_steps - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """主训练循环"""
        self.model.to(self.device)
        self.model.train()

        total_params = self.model.get_total_param_count()
        trainable_params = self.model.get_trainable_param_count()
        print(f"总参数: {total_params:,} | 可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1
            epoch_loss = self._train_epoch()

            if self.val_dataloader and self.global_step % self.config.eval_every == 0:
                val_loss = self._validate()
                print(f"[Epoch {epoch}] val_loss={val_loss:.4f}")

        print(f"训练完成, 共 {self.global_step} 步")

    def _train_epoch(self) -> float:
        """训练一个 epoch (遍历所有 episode)"""
        total_loss = 0
        num_episodes = 0

        for episode_segments in self.train_dataloader:
            if self.global_step >= self.config.max_steps:
                break

            loss = self._train_episode(episode_segments)
            total_loss += loss
            num_episodes += 1

        return total_loss / max(num_episodes, 1)

    def _train_episode(self, episode_segments: list[torch.Tensor]) -> float:
        """
        训练一个 episode (多轮对话)。

        episode_segments: segment 列表，每个 (B, T)
        """
        B = episode_segments[0].shape[0]
        state = self.model.init_state(B).to(self.device)
        accumulated_loss = torch.tensor(0.0, device=self.device)
        episode_total_loss = 0.0
        steps_in_window = 0

        for seg_idx, segment in enumerate(episode_segments):
            segment = segment.to(self.device)

            # 截断 BPTT: 每 tbptt_steps 切断梯度
            if seg_idx > 0 and seg_idx % self.config.tbptt_steps == 0:
                # 先做 backward + step
                if steps_in_window > 0:
                    avg_loss = accumulated_loss / steps_in_window
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_params(),
                        self.config.grad_clip,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    episode_total_loss += avg_loss.item()

                    if self.global_step % self.config.log_every == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        scale = torch.sigmoid(self.model.plugin.state_scale).item()
                        print(f"  [Step {self.global_step}] loss={avg_loss.item():.4f} lr={lr:.2e} scale={scale:.4f}")

                    # 保存 checkpoint
                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()

                # 切断梯度
                state = state.detach()
                accumulated_loss = torch.tensor(0.0, device=self.device)
                steps_in_window = 0

            # Forward
            with torch.amp.autocast("cuda", dtype=self.dtype):
                result = self.model(segment, state, labels=segment)

            state = result["state_next"]
            accumulated_loss = accumulated_loss + result["loss"]
            steps_in_window += 1

        # 处理剩余的 accumulated loss
        if steps_in_window > 0:
            avg_loss = accumulated_loss / steps_in_window
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_params(),
                self.config.grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            episode_total_loss += avg_loss.item()

        return episode_total_loss

    @torch.no_grad()
    def _validate(self) -> float:
        """验证集评估"""
        self.model.eval()
        total_loss = 0
        num_episodes = 0

        for episode_segments in self.val_dataloader:
            B = episode_segments[0].shape[0]
            state = self.model.init_state(B).to(self.device)

            for seg_idx, segment in enumerate(episode_segments):
                segment = segment.to(self.device)

                with torch.amp.autocast("cuda", dtype=self.dtype):
                    result = self.model(segment, state, labels=segment)

                state = result["state_next"]
                total_loss += result["loss"].item()

            num_episodes += 1

        self.model.train()
        return total_loss / max(num_episodes, 1)

    def _save_checkpoint(self, path: Optional[str] = None):
        """保存 checkpoint (只保存可训练参数 + 优化器状态)"""
        if path is None:
            path = f"checkpoints/xinhe_step_{self.global_step}.pt"

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # 收集 StatePlugin 和 LoRA 的 state_dict
        plugin_state = self.model.plugin.state_dict()

        # LoRA 参数
        lora_state = {}
        for name, module in self.model.backbone.named_modules():
            from ..model.lora import LoRALinear
            if isinstance(module, LoRALinear):
                lora_state[f"{name}.lora_A"] = module.lora_A.data
                lora_state[f"{name}.lora_B"] = module.lora_B.data

        checkpoint = {
            "global_step": self.global_step,
            "plugin_state": plugin_state,
            "lora_state": lora_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"  [Checkpoint] 保存到 {path}")

    def load_checkpoint(self, path: str):
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.plugin.load_state_dict(checkpoint["plugin_state"])
        self.global_step = checkpoint["global_step"]

        # 恢复 LoRA 参数
        from ..model.lora import LoRALinear
        lora_state = checkpoint.get("lora_state", {})
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A.data = lora_state[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state:
                    module.lora_B.data = lora_state[f"{name}.lora_B"]

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"[Checkpoint] 从 {path} 恢复, step={self.global_step}")
