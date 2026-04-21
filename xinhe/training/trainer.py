"""
Trainer — 训练循环

核心特性:
- state 跨 segment (对话轮次) 传递
- 截断 BPTT: 每 tbptt_steps 轮做 detach + backward + step
- 只训练 StateInterface + LoRA 参数
"""
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 预加载 torch 子模块 (如果在函数内 import, Python 会把 torch 误作 local 变量,
# 导致 UnboundLocalError)
try:
    import torch._dynamo
    import torch._logging
except Exception:
    pass

from ..model.xinhe_model import XinheModel
from ..model.config import XinheConfig
from ..model.lora import get_lora_params


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
        pad_token_id: Optional[int] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.pad_token_id = pad_token_id

        # 设备和精度
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype, torch.float32)

        # 分组优化: Plugin / LoRA 两组独立学习率
        self._apply_freezes(config)
        self.optimizer = self._build_optimizer(config)

        # 学习率调度: cosine with warmup
        self.scheduler = self._build_scheduler()

        # 课程阶段名 (用于 checkpoint 标记)
        self.current_stage_name = ""

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float("inf")
        self._accum_count = 0  # 梯度累积计数器

        # 早停状态 (滑动窗口: 连续 N 步 loss 低 + acc 高才收敛)
        self._recent_losses = []
        self._recent_accs = []
        self._early_stopped = False

        # EMA 用于日志显示 (alpha≈0.04, ~50步窗口)
        self._ema_loss = None
        self._ema_acc = None

    def _apply_freezes(self, config: XinheConfig):
        """按配置冻结/解冻 LoRA 和 plugin 参数 (v5a: 仅支持 freeze_lora 一个开关)"""
        freeze_lora = getattr(config, "freeze_lora", False)
        for p in get_lora_params(self.model.backbone):
            p.requires_grad = not freeze_lora
        # plugin_lr_multiplier=0 表示冻结整个 plugin
        freeze_plugin = getattr(config, "plugin_lr_multiplier", 1.0) == 0
        for p in self.model.state_interface.parameters():
            p.requires_grad = not freeze_plugin

    def _build_optimizer(self, config: XinheConfig) -> torch.optim.AdamW:
        """构建 optimizer: Plugin / LoRA 两组独立学习率 (Adam 默认 eps)。"""
        lr = config.learning_rate
        plugin_mult = getattr(config, "plugin_lr_multiplier", 1.0)

        plugin_params = [p for p in self.model.state_interface.parameters() if p.requires_grad]

        param_groups = []
        if plugin_params:
            param_groups.append({"params": plugin_params, "lr": lr * plugin_mult})
        if not getattr(config, "freeze_lora", False):
            lora_params = [p for p in get_lora_params(self.model.backbone) if p.requires_grad]
            if lora_params:
                param_groups.append({"params": lora_params, "lr": lr})
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    def _build_scheduler(self):
        """Cosine schedule with linear warmup + min LR clamp (5% of peak)。

        Min clamp 避免 LR→0 时 Adam 的 v_hat 长期低估 + 小梯度 spike 产生
        巨大 update step (grad / sqrt(v_hat)) 导致训练末期崩盘 (observed v5e stage 2)。
        """
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        min_mult = 0.1  # LR 最低降到 peak 的 10% (配合 Adam eps=1e-6 防末期爆炸)

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(max_steps - warmup, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_mult, cosine)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """主训练循环"""
        self.model.setup_device(self.device)
        self.model.train()

        # TF32 加速 (float32 矩阵乘法用 TensorFloat32)
        torch.set_float32_matmul_precision('high')

        # torch.compile: 只编译 backbone (transformer blocks)，plugin 的 state 操作不编译
        # 配套: reset_for_new_stage() 里显式 dynamo.reset() 清掉 cache, 避免跨阶段 OOM
        import sys
        import warnings
        if sys.platform == "linux" and not getattr(self, "_compiled", False):
            # 屏蔽 Dynamo 追踪 FLA 库时的良性告警
            warnings.filterwarnings("ignore", message=".*Dynamo.*", category=UserWarning)
            warnings.filterwarnings("ignore", module=r"torch\._dynamo.*", category=UserWarning)
            try:
                torch._logging.set_logs(dynamo=40)  # ERROR only
            except Exception:
                pass
            try:
                self.model.backbone.forward_blocks = torch.compile(
                    self.model.backbone.forward_blocks)
                self._compiled = True
                print("[torch.compile] backbone 已编译")
            except Exception as e:
                print(f"[torch.compile] 跳过: {e}")

        total_params = self.model.get_total_param_count()
        trainable_params = self.model.get_trainable_param_count()
        print(f"总参数: {total_params:,} | 可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        if self.config.grad_accum_steps > 1:
            print(f"梯度累积: {self.config.grad_accum_steps} 步")

        self.optimizer.zero_grad()
        self._last_eval_step = -1  # 避免同步骤重复 eval
        epoch = 0
        while self.global_step < self.config.max_steps:
            if self._early_stopped:
                break
            epoch += 1
            self._train_epoch()
            # 注: val 已改到 _maybe_optimizer_step 里每 eval_every 步触发, 不再只在 epoch 末

        scale = torch.sigmoid(self.model.state_interface.read_scale).item()
        if self._early_stopped:
            print(f"训练已收敛, 共 {self.global_step} 步, scale={scale:.4f}")
        else:
            print(f"训练完成, 共 {self.global_step} 步, scale={scale:.4f}")

    def _train_epoch(self) -> float:
        """训练一个 epoch (遍历所有 episode)"""
        total_loss = 0
        num_episodes = 0

        for episode_segments in self.train_dataloader:
            if self.global_step >= self.config.max_steps or self._early_stopped:
                break

            loss = self._train_episode(episode_segments)
            total_loss += loss
            num_episodes += 1

        return total_loss / max(num_episodes, 1)

    def _train_episode(self, episode_segments) -> float:
        """
        训练一个 episode (多轮对话)。

        episode_segments: segment 列表，每个是 (input_ids, labels, weights) tuple，shape (B, T)
        """
        B = episode_segments[0][0].shape[0]
        state = self.model.init_state(B).to(self.device)
        accumulated_loss = torch.tensor(0.0, device=self.device)
        episode_total_loss = 0.0
        episode_correct = 0
        episode_total = 0
        loss_segments = 0  # 只计数有真实 loss 的 segment

        for seg_idx, batch in enumerate(episode_segments):
            segment, labels, weights = batch
            segment = segment.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            # 截断 BPTT: 每 tbptt_steps 切断梯度
            if seg_idx > 0 and seg_idx % self.config.tbptt_steps == 0:
                # 先做 backward (+ 可能的 optimizer step)
                if loss_segments > 0:
                    avg_loss = accumulated_loss / loss_segments
                    (avg_loss / self.config.grad_accum_steps).backward()
                    episode_total_loss += avg_loss.item()
                    self._maybe_optimizer_step(avg_loss.item())
                elif seg_idx > 0:
                    pass  # 没有 loss segment，跳过

                # 切断梯度
                state = state.detach()
                accumulated_loss = torch.tensor(0.0, device=self.device)
                loss_segments = 0

            # Forward
            with torch.amp.autocast("cuda", dtype=self.dtype):
                result = self.model(segment, state, labels=labels, pad_token_id=self.pad_token_id, weights=weights)

            state = result["state_next"]
            seg_loss = result["loss"]
            accumulated_loss = accumulated_loss + seg_loss
            correct = result.get("correct", 0)
            total = result.get("total", 0)
            episode_correct += correct.item() if hasattr(correct, 'item') else correct
            episode_total += total.item() if hasattr(total, 'item') else total
            # 只有真正有 loss 的 segment 才计数
            has_valid_labels = (labels != -100).any()
            if has_valid_labels:
                loss_segments += 1

        # 处理剩余的 accumulated loss
        if loss_segments > 0:
            avg_loss = accumulated_loss / loss_segments
            (avg_loss / self.config.grad_accum_steps).backward()
            episode_total_loss += avg_loss.item()
            acc = episode_correct / max(episode_total, 1)
            self._maybe_optimizer_step(avg_loss.item(), acc)

        return episode_total_loss

    @torch.no_grad()
    def _validate(self) -> None:
        """验证集评估: VALUE/FRAME/TELL 分解 (替代无信息量的 val_loss)"""
        self.model.eval()
        try:
            from scripts.eval_value_breakdown import eval_value_breakdown_fast
            from pathlib import Path as _Path
            if _Path(self.config.val_path).exists():
                tokenizer = getattr(self, "_fast_eval_tokenizer", None)
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(_Path(self.config.backbone_model_path).resolve()),
                        trust_remote_code=True,
                    )
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    self._fast_eval_tokenizer = tokenizer
                breakdown = eval_value_breakdown_fast(
                    self.model, tokenizer, self.config.val_path, self.device,
                    seg_len=self.config.segment_length, max_episodes=50,
                )
                print(f"  [val breakdown] VALUE={breakdown['VALUE']:.2%} "
                      f"FRAME={breakdown['FRAME']:.2%} TELL={breakdown['TELL']:.2%}")
        except Exception as e:
            print(f"  [val breakdown] 跳过: {e}")
        self.model.train()

    def _maybe_optimizer_step(self, last_loss: float, last_acc: float = 1.0):
        """梯度累积: 累积够 grad_accum_steps 次后执行一次 optimizer step"""
        self._accum_count += 1
        if self._accum_count < self.config.grad_accum_steps:
            return

        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_params(),
            self.config.grad_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        self._accum_count = 0

        # 更新 EMA (窗口 ≈ log_every 步)
        alpha = 2.0 / (self.config.log_every + 1)
        if self._ema_loss is None:
            self._ema_loss = last_loss
            self._ema_acc = last_acc
        else:
            self._ema_loss = alpha * last_loss + (1 - alpha) * self._ema_loss
            self._ema_acc = alpha * last_acc + (1 - alpha) * self._ema_acc

        if self.global_step % self.config.log_every == 0:
            lr = self.scheduler.get_last_lr()[0]
            scale = torch.sigmoid(self.model.state_interface.read_scale).item()
            print(f"  [Step {self.global_step}] loss={last_loss:.6f} acc={last_acc:.2%} ema_loss={self._ema_loss:.4f} ema_acc={self._ema_acc:.2%} lr={lr:.2e} scale={scale:.4f}")

        if self.global_step % self.config.save_every == 0:
            self._save_checkpoint()

        # 每 eval_every 步跑一次 val breakdown (不再依赖 epoch 边界)
        if (self.val_dataloader is not None
                and self.global_step % self.config.eval_every == 0
                and self.global_step != self._last_eval_step):
            self._last_eval_step = self.global_step
            self._validate()

        # 早停检查 (滑动窗口: 最近 patience 步全部满足才收敛)
        early_stop_loss = getattr(self.config, 'early_stop_loss', 0)
        early_stop_patience = getattr(self.config, 'early_stop_patience', 0)
        if early_stop_loss > 0 and early_stop_patience > 0:
            self._recent_losses.append(last_loss)
            self._recent_accs.append(last_acc)
            # 只保留最近 patience 步
            if len(self._recent_losses) > early_stop_patience:
                self._recent_losses.pop(0)
                self._recent_accs.pop(0)
            # 条件: 窗口填满 + 所有 loss < 阈值 + 所有 acc > 95%
            if (len(self._recent_losses) >= early_stop_patience
                    and max(self._recent_losses) < early_stop_loss
                    and min(self._recent_accs) > 0.95):
                self._early_stopped = True
                avg_loss = sum(self._recent_losses) / len(self._recent_losses)
                avg_acc = sum(self._recent_accs) / len(self._recent_accs)
                print(f"  [已收敛] 连续{early_stop_patience}步: loss<{early_stop_loss} acc={avg_acc:.2%}")

    def reset_for_new_stage(self, config: XinheConfig, train_dataloader: DataLoader,
                            val_dataloader: Optional[DataLoader] = None):
        """课程学习：切换到新阶段，保留模型权重，重建 optimizer/scheduler"""
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.global_step = 0
        self._accum_count = 0
        self._recent_losses = []
        self._recent_accs = []
        self._early_stopped = False
        self._ema_loss = None
        self._ema_acc = None

        # 动态更新 runtime 超参 (不重建模型, 只改 StateInterface 上的标量)
        new_wi = getattr(config, "write_iterations", 1)
        if new_wi != getattr(self.model.state_interface, "write_iterations", 1):
            self.model.state_interface.write_iterations = new_wi
            print(f"[write_iterations] 阶段切换 → {new_wi}")

        # 清掉 Dynamo 编译缓存: 新 stage 的 episode_length/batch_size 变化会触发重编译,
        # 若不清, 旧阶段的编译 graph 还占着显存 → 跨阶段 OOM
        if getattr(self, "_compiled", False):
            try:
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                print("[torch.compile] 清空 Dynamo cache (阶段切换)")
            except Exception as e:
                print(f"[torch.compile] reset 失败: {e}")

        self._apply_freezes(config)
        self.optimizer = self._build_optimizer(config)
        self.scheduler = self._build_scheduler()
        self.optimizer.zero_grad()

    def _save_checkpoint(self, path: Optional[str] = None):
        """保存 checkpoint (只保存可训练参数 + 优化器状态)"""
        if path is None:
            path = f"checkpoints/xinhe_step_{self.global_step}.pt"

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # 收集 StateInterface 和 LoRA 的 state_dict
        plugin_state = self.model.state_interface.state_dict()

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
            "curriculum_stage": self.current_stage_name,
        }

        torch.save(checkpoint, path)
        print(f"  [Checkpoint] 保存到 {path}")

    def load_checkpoint(self, path: str):
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        result = self.model.state_interface.load_state_dict(checkpoint["plugin_state"], strict=False)
        if result.missing_keys:
            print(f"  注意: checkpoint 缺少 {result.missing_keys}，使用默认初始化")
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
            # 将 optimizer state 搬到正确设备
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"[Checkpoint] 从 {path} 恢复, step={self.global_step}")
