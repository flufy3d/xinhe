"""
Trainer (v7) — 训练循环

核心特性:
- state 跨 segment (对话轮次) 传递，state 是单张量 W: (B, H, d_v, d_k)
- 截断 BPTT: 每 tbptt_steps 轮做 detach + backward + step
- 只训练 Hippocampus + LoRA 参数
"""
import math
from pathlib import Path
from typing import Optional

import torch
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
        self.model.config = config
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

        # 早停状态
        self._recent_losses = []
        self._recent_accs = []
        self._early_stopped = False

        # EMA 用于日志显示 (alpha≈0.04, ~50步窗口)
        self._ema_loss = None
        self._ema_acc = None

    def _apply_freezes(self, config: XinheConfig):
        """按配置冻结/解冻 LoRA / Hippocampus 参数。

        v7:
        - freeze_lora: Stage 0a/0b bootstrap 用，强迫 Hippocampus 独立承担
        - freeze_time_shift: 实验用，冻结 Linear(hidden, H) 的 γ 偏移层
        - plugin_lr_multiplier=0 也等效冻结整个 Hippocampus
        """
        freeze_lora = getattr(config, "freeze_lora", False)
        for p in get_lora_params(self.model.backbone):
            p.requires_grad = not freeze_lora

        freeze_plugin = getattr(config, "plugin_lr_multiplier", 1.0) == 0
        for p in self.model.hippocampus.parameters():
            p.requires_grad = not freeze_plugin

        # 额外：freeze_time_shift 单独冻 Δγ 层 + reset 到零初值
        # 语义："我不要内容驱动的 γ，只用静态先验"。如果前一阶段学偏了，必须 reset 才符合语义。
        if getattr(config, "freeze_time_shift", False):
            import torch.nn as nn
            nn.init.zeros_(self.model.hippocampus.time_shift.weight)
            nn.init.zeros_(self.model.hippocampus.time_shift.bias)
            for p in self.model.hippocampus.time_shift.parameters():
                p.requires_grad = False
            print("  [freeze_time_shift] time_shift 已 reset 到零 + 冻结（γ 仅用静态先验 σ(θ_h)）")

        # freeze_beta_weight：reset beta_proj.weight 到零 + 冻结，保留 bias 可训
        # 让 β = σ(bias) 纯 per-head 静态先验。防止 β 在 W 空态死锁中被压到 0。
        if getattr(config, "freeze_beta_weight", False):
            import torch.nn as nn
            nn.init.zeros_(self.model.hippocampus.beta_proj.weight)
            self.model.hippocampus.beta_proj.weight.requires_grad = False
            # bias 保留可训（让模型调节整体写强度）
            print("  [freeze_beta_weight] beta_proj.weight reset 到零 + 冻结（β=σ(bias) per-head 静态）")

        # freeze_read_scale_at: 强制 read_scale = logit(x) 并冻结。破 chicken-and-egg
        scale_val = getattr(config, "freeze_read_scale_at", 0.0)
        if scale_val > 0.0 and scale_val < 1.0:
            import math as _math
            logit = _math.log(scale_val / (1.0 - scale_val))
            with torch.no_grad():
                self.model.hippocampus.read_scale.data.fill_(logit)
            self.model.hippocampus.read_scale.requires_grad = False
            print(f"  [freeze_read_scale_at] read_scale = {scale_val:.3f} (logit={logit:.3f}) 冻结")

    def _build_optimizer(self, config: XinheConfig) -> torch.optim.AdamW:
        """构建 optimizer: Hippocampus / LoRA 两组独立学习率。"""
        lr = config.learning_rate
        plugin_mult = getattr(config, "plugin_lr_multiplier", 1.0)

        plugin_params = [p for p in self.model.hippocampus.parameters() if p.requires_grad]

        param_groups = []
        if plugin_params:
            param_groups.append({"params": plugin_params, "lr": lr * plugin_mult})

        if not getattr(config, "freeze_lora", False):
            lora_params = [p for p in get_lora_params(self.model.backbone) if p.requires_grad]
            if lora_params:
                param_groups.append({"params": lora_params, "lr": lr})
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    def _build_scheduler(self):
        """Cosine schedule with linear warmup + min LR clamp (1% of peak)。"""
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        min_mult = 0.01

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

        # TF32 加速
        torch.set_float32_matmul_precision('high')

        # torch.compile: 只编译 backbone (transformer blocks)，plugin 的 state 操作不编译
        import sys
        import warnings
        if sys.platform == "linux" and not getattr(self, "_compiled", False):
            warnings.filterwarnings("ignore", message=".*Dynamo.*", category=UserWarning)
            warnings.filterwarnings("ignore", module=r"torch\._dynamo.*", category=UserWarning)
            try:
                torch._logging.set_logs(dynamo=40)
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
        self._last_eval_step = -1
        epoch = 0
        while self.global_step < self.config.max_steps:
            if self._early_stopped:
                break
            epoch += 1
            self._train_epoch()

        # 阶段末尾最终 val（若和上一次 eval 不同 step）
        if (self.val_dataloader is not None
                and self.global_step != self._last_eval_step):
            self._last_eval_step = self.global_step
            print(f"  [最终 val @ step {self.global_step}]")
            self._validate()

        read_scale = torch.sigmoid(self.model.hippocampus.read_scale).item()
        if self._early_stopped:
            print(f"训练已收敛, 共 {self.global_step} 步, read_scale={read_scale:.4f}")
        else:
            print(f"训练完成, 共 {self.global_step} 步, read_scale={read_scale:.4f}")

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
        """训练一个 episode (多轮对话)。

        episode_segments: segment 列表，每个是 (input_ids, labels, weights) tuple，shape (B, T)
        """
        B = episode_segments[0][0].shape[0]
        state = self.model.init_state(B).to(self.device)
        accumulated_loss = torch.tensor(0.0, device=self.device)
        episode_total_loss = 0.0
        episode_correct = 0
        episode_total = 0
        loss_segments = 0

        for seg_idx, batch in enumerate(episode_segments):
            segment, labels, weights = batch
            segment = segment.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            if seg_idx > 0 and seg_idx % self.config.tbptt_steps == 0:
                if loss_segments > 0:
                    avg_loss = accumulated_loss / loss_segments
                    (avg_loss / self.config.grad_accum_steps).backward()
                    episode_total_loss += avg_loss.item()
                    self._maybe_optimizer_step(avg_loss.item())

                state = state.detach()
                accumulated_loss = torch.tensor(0.0, device=self.device)
                loss_segments = 0

            with torch.amp.autocast("cuda", dtype=self.dtype):
                result = self.model(segment, state, labels=labels, pad_token_id=self.pad_token_id, weights=weights)

            state = result["state_next"]
            seg_loss = result["loss"]
            accumulated_loss = accumulated_loss + seg_loss
            correct = result.get("correct", 0)
            total = result.get("total", 0)
            episode_correct += correct.item() if hasattr(correct, 'item') else correct
            episode_total += total.item() if hasattr(total, 'item') else total
            has_valid_labels = (labels != -100).any()
            if has_valid_labels:
                loss_segments += 1

        if loss_segments > 0:
            avg_loss = accumulated_loss / loss_segments
            (avg_loss / self.config.grad_accum_steps).backward()
            episode_total_loss += avg_loss.item()
            acc = episode_correct / max(episode_total, 1)
            self._maybe_optimizer_step(avg_loss.item(), acc)

        return episode_total_loss

    @torch.no_grad()
    def _validate(self) -> None:
        """验证集评估。
        默认: VALUE/FRAME/TELL breakdown，VALUE ≥ early_stop_value 触发早停。
        use_joint_early_stop=True 时: 额外跑 WorldQA/Refusal/Compositional/Decay/RapidOW 5 指标。
        """
        self.model.eval()
        try:
            from scripts.eval_value_breakdown import eval_value_breakdown_fast
            from pathlib import Path as _Path
            if not _Path(self.config.val_path).exists():
                self.model.train()
                return

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
            value_acc = breakdown["VALUE"]
            print(f"  [val breakdown] VALUE={value_acc:.2%} "
                  f"FRAME={breakdown['FRAME']:.2%} TELL={breakdown['TELL']:.2%}")

            use_joint = getattr(self.config, "use_joint_early_stop", False)

            if not use_joint:
                early_stop_value = getattr(self.config, "early_stop_value", 0.0)
                if early_stop_value > 0 and value_acc >= early_stop_value:
                    self._early_stopped = True
                    print(f"  [已收敛] VALUE={value_acc:.2%} ≥ {early_stop_value:.2%}，提前进下一阶段")
                self.model.train()
                return

            # 联合早停 (v8): 通用循环，从 config.early_stop dict 或 early_stop_<key> 字段读阈值
            from xinhe.evaluation.event_eval import eval_joint_v8
            joint = eval_joint_v8(
                self.model, tokenizer, self.config, device=self.device,
                max_episodes=50,
            )
            joint["VALUE"] = value_acc  # 内置 VALUE 指标始终参与

            def _fmt(x): return f"{x:.2%}"
            line = " ".join(f"{k}={_fmt(v)}" for k, v in joint.items() if v > 0)
            print(f"  [joint] {line}")

            # 阈值来源优先级: config.early_stop dict > early_stop_<key> 字段
            thresholds: dict[str, float] = {}
            es_dict = getattr(self.config, "early_stop", None)
            if isinstance(es_dict, dict):
                for k, v in es_dict.items():
                    if isinstance(v, (int, float)) and v > 0:
                        thresholds[k] = float(v)
            for attr in dir(self.config):
                if attr.startswith("early_stop_") and attr != "early_stop_value":
                    val = getattr(self.config, attr, 0.0)
                    if isinstance(val, (int, float)) and val > 0:
                        thresholds[attr[len("early_stop_"):]] = float(val)
            es_value = getattr(self.config, "early_stop_value", 0.0)
            if es_value and es_value > 0:
                thresholds["VALUE"] = float(es_value)

            checks = [(k, joint.get(k, 0.0), thr) for k, thr in thresholds.items()]
            active = [c for c in checks if c[2] > 0]
            if not active:
                self.model.train()
                return

            missed = [c for c in active if c[1] < c[2]]
            if not missed:
                self._early_stopped = True
                passed = " ".join(f"{name}≥{thr:.0%}" for name, _, thr in active)
                print(f"  [已收敛] {len(active)} 个 active 指标全部达标：{passed}")
            else:
                summary = ", ".join(f"{name}({val:.2%}<{thr:.0%})" for name, val, thr in missed)
                print(f"  [未达标] {summary}")

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

        # 更新 EMA
        alpha = 2.0 / (self.config.log_every + 1)
        if self._ema_loss is None:
            self._ema_loss = last_loss
            self._ema_acc = last_acc
        else:
            self._ema_loss = alpha * last_loss + (1 - alpha) * self._ema_loss
            self._ema_acc = alpha * last_acc + (1 - alpha) * self._ema_acc

        if self.global_step % self.config.log_every == 0:
            lr = self.scheduler.get_last_lr()[0]
            hippo = self.model.hippocampus
            read_scale = torch.sigmoid(hippo.read_scale).item()
            # v7 γ 诊断
            gamma_prior = torch.sigmoid(hippo.head_decay_logits)
            gp_min = gamma_prior.min().item()
            gp_max = gamma_prior.max().item()
            diag = hippo.get_gamma_diagnostics()
            gamma_str = f"γ_prior=[{gp_min:.3f},{gp_max:.3f}]"
            if diag is not None:
                gamma_str += (
                    f" γ_tok={diag['gamma_token_mean']:.3f}±{diag['gamma_token_std']:.3f}"
                    f" γ_min={diag['gamma_token_min']:.3f}"
                )
            print(
                f"  [Step {self.global_step}] ema_loss={self._ema_loss:.4f} "
                f"ema_acc={self._ema_acc:.2%} lr={lr:.2e} "
                f"read_scale={read_scale:.4f} {gamma_str}"
            )

        if self.global_step % self.config.save_every == 0:
            self._save_checkpoint()

        if (self.val_dataloader is not None
                and self.global_step % self.config.eval_every == 0
                and self.global_step != self._last_eval_step):
            self._last_eval_step = self.global_step
            self._validate()

    def reset_for_new_stage(self, config: XinheConfig, train_dataloader: DataLoader,
                            val_dataloader: Optional[DataLoader] = None):
        """课程学习：切换到新阶段，保留模型权重，重建 optimizer/scheduler"""
        self.config = config
        self.model.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.global_step = 0
        self._accum_count = 0
        self._recent_losses = []
        self._recent_accs = []
        self._early_stopped = False
        self._ema_loss = None
        self._ema_acc = None

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

        hippocampus_state = self.model.hippocampus.state_dict()

        # LoRA 参数
        lora_state = {}
        for name, module in self.model.backbone.named_modules():
            from ..model.lora import LoRALinear
            if isinstance(module, LoRALinear):
                lora_state[f"{name}.lora_A"] = module.lora_A.data
                lora_state[f"{name}.lora_B"] = module.lora_B.data

        checkpoint = {
            "global_step": self.global_step,
            "hippocampus_state": hippocampus_state,
            "lora_state": lora_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "curriculum_stage": self.current_stage_name,
        }

        torch.save(checkpoint, path)
        print(f"  [Checkpoint] 保存到 {path}")

    def load_checkpoint(self, path: str):
        """加载 checkpoint（v7 strict=True，不兼容 v5c/v6 旧格式）"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "hippocampus_state" not in checkpoint:
            raise RuntimeError(
                "checkpoint 缺少 'hippocampus_state' 键。v7 不兼容 v5c/v6 旧格式，"
                "请从零重训。"
            )
        self.model.hippocampus.load_state_dict(checkpoint["hippocampus_state"], strict=True)
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
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"[Checkpoint] 从 {path} 恢复, step={self.global_step}")
