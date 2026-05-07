"""
Trainer (v9) — 训练循环

核心特性:
- state 跨 turn 传递,state 是 XinheMemoryState(per-layer LayerMemState 字典)
- 截断 BPTT: 每 tbptt_turns 轮做 detach + backward + step
- 只训练 NeuralMemoryPair 参数(backbone 全冻,无 LoRA)
"""
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    import torch._dynamo
    import torch._logging
except Exception:
    pass

from ..model.xinhe_model import XinheModel
from ..model.config import XinheConfig


class Trainer:
    """
    心核训练器。

    episode 循环:
        for each episode (多轮对话):
            state = model.init_state()
            for turn_idx, turn_tensor in enumerate(episode):
                1. forward(turn_tensor, state) → logits, state_next
                2. 累积 loss
                3. 每 tbptt_turns 做一次 backward + optimizer step + state.detach()
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
        """按配置冻结 NeuralMemoryPair 内的 alpha / gate_q。
        plugin_lr_multiplier=0 等效冻结整个 memory。"""
        freeze_plugin = getattr(config, "plugin_lr_multiplier", 1.0) == 0
        for p in self.model.memory.parameters():
            p.requires_grad = not freeze_plugin

        if getattr(config, "freeze_alpha", False):
            for pair in self.model.memory.values():
                pair.alpha_logit.requires_grad = False
            print("  [freeze_alpha] alpha_logit 全部冻结")

        if getattr(config, "freeze_gate_q", False):
            for pair in self.model.memory.values():
                for p in pair.gate_q.parameters():
                    p.requires_grad = False
            print("  [freeze_gate_q] gate_q.weight 全部冻结")

    def _build_optimizer(self, config: XinheConfig) -> torch.optim.AdamW:
        """构建 optimizer:NeuralMemoryPair 单组 lr × plugin_lr_multiplier。"""
        lr = config.learning_rate
        plugin_mult = getattr(config, "plugin_lr_multiplier", 1.0)

        plugin_params = self.model.get_trainable_params()
        param_groups = []
        if plugin_params:
            param_groups.append({"params": plugin_params, "lr": lr * plugin_mult})
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

        # NOTE: 不开 torch.compile。NeuralMemoryPair 内 store_memories 走 vmap(grad)
        # 做 test-time SGD,Dynamo 编译路径会强制 disable_saved_tensors_hooks,
        # 但 vmap(grad) 自己也用同一机制 → 冲突直接 InternalTorchDynamoError。
        # 而且 backbone 是 frozen 的,只 fwd 不 bwd,compile 收益只剩 ~20%,不值。

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

        # 取一个 layer 的 alpha 平均当作整体记忆开度指标
        alphas = [torch.sigmoid(p.alpha_logit).item() for p in self.model.memory.values()]
        alpha_mean = sum(alphas) / max(len(alphas), 1)
        if self._early_stopped:
            print(f"训练已收敛, 共 {self.global_step} 步, alpha_mean={alpha_mean:.4f}")
        else:
            print(f"训练完成, 共 {self.global_step} 步, alpha_mean={alpha_mean:.4f}")

    def _train_epoch(self) -> float:
        """训练一个 epoch (遍历所有 episode)"""
        total_loss = 0
        num_episodes = 0

        for episode_turns in self.train_dataloader:
            if self.global_step >= self.config.max_steps or self._early_stopped:
                break

            loss = self._train_episode(episode_turns)
            total_loss += loss
            num_episodes += 1

        return total_loss / max(num_episodes, 1)

    def _train_episode(self, episode_turns) -> float:
        """训练一个 episode (多轮对话)。

        episode_turns: turn tensor 列表，每个是 (input_ids, labels, weights) tuple，shape (B, T)
                       其中 T = turn_max_tokens
        """
        B = episode_turns[0][0].shape[0]
        state = self.model.init_state(B).to(self.device)
        accumulated_loss = torch.tensor(0.0, device=self.device)
        episode_total_loss = 0.0
        episode_correct = 0
        episode_total = 0
        loss_turns = 0

        for turn_idx, batch in enumerate(episode_turns):
            turn_ids, labels, weights = batch
            turn_ids = turn_ids.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            if turn_idx > 0 and turn_idx % self.config.tbptt_turns == 0:
                if loss_turns > 0:
                    avg_loss = accumulated_loss / loss_turns
                    (avg_loss / self.config.grad_accum_steps).backward()
                    episode_total_loss += avg_loss.item()
                    # 传 running acc:混合数据时 grad_accum cycle 可能落在 mid-ep call,
                    # EMA 必须得到真实 partial accuracy,否则会被 default 1.0 污染成假 100%
                    running_acc = episode_correct / max(episode_total, 1)
                    self._maybe_optimizer_step(avg_loss.item(), running_acc)

                state = state.detach()
                accumulated_loss = torch.tensor(0.0, device=self.device)
                loss_turns = 0

            with torch.amp.autocast("cuda", dtype=self.dtype):
                # model.forward 内部仍叫 segment（纯实现细节，与业务 turn 解耦）
                result = self.model(turn_ids, state, labels=labels, pad_token_id=self.pad_token_id, weights=weights)

            state = result["state_next"]
            turn_loss = result["loss"]
            accumulated_loss = accumulated_loss + turn_loss
            correct = result.get("correct", 0)
            total = result.get("total", 0)
            episode_correct += correct.item() if hasattr(correct, 'item') else correct
            episode_total += total.item() if hasattr(total, 'item') else total
            has_valid_labels = (labels != -100).any()
            if has_valid_labels:
                loss_turns += 1

        if loss_turns > 0:
            avg_loss = accumulated_loss / loss_turns
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
                seg_len=self.config.turn_max_tokens, max_episodes=50,
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

            # 联合早停: 通用循环,从 config.early_stop dict 或 early_stop_<key> 字段读阈值
            from xinhe.evaluation.event_eval import eval_joint
            joint = eval_joint(
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

    def _capture_mac_grads(self) -> dict:
        """诊断:捕获 MAC 三组 + Hippo to_decay_factor 的梯度 norm。
        在 optimizer.step() 前调用(grad 已 clip)。grad 为 None 表示该参数本步无梯度。
        """
        out = {}
        for name in ("persistent_mem", "mem_token_init", "mac_inject_logit"):
            p = getattr(self.model, name, None)
            out[name] = (
                p.grad.detach().float().norm().item()
                if (p is not None and p.grad is not None) else None
            )
        # to_decay_factor 是 per-layer Hippo NeuralMemory 内部参数,聚合 norm
        decay_norms = []
        for pair in self.model.memory.values():
            for p in pair.hippocampus.to_decay_factor.parameters():
                if p.grad is not None:
                    decay_norms.append(p.grad.detach().float().norm().item())
        out["to_decay_factor"] = (
            sum(decay_norms) / len(decay_norms) if decay_norms else None
        )
        return out

    def _maybe_optimizer_step(self, last_loss: float, last_acc: float):
        """梯度累积: 累积够 grad_accum_steps 次后执行一次 optimizer step.

        last_acc 必须显式传(running 或 episode-level),不给默认值 ——
        默认值会在 mixed-source grad_accum cycle 落在 mid-ep call 时把 EMA 污染。
        """
        self._accum_count += 1
        if self._accum_count < self.config.grad_accum_steps:
            return

        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_params(),
            self.config.grad_clip,
        )
        # 诊断:在 step 前捕获 grad norm(随后 zero_grad 会清掉)
        self._last_mac_grads = self._capture_mac_grads()
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
            # pure MAC 通路效度指标:persistent_mem / mem_token_init norm + 注入开度
            pmem_norm = (
                self.model.persistent_mem.detach().float().norm().item()
                if getattr(self.model, "persistent_mem", None) is not None else 0.0
            )
            mtok_norm = (
                self.model.mem_token_init.detach().float().norm().item()
                if getattr(self.model, "mem_token_init", None) is not None else 0.0
            )
            inject = (
                torch.sigmoid(self.model.mac_inject_logit).item()
                if getattr(self.model, "mac_inject_logit", None) is not None else 0.0
            )
            # 诊断:MAC + decay 梯度 norm + mem_out norm(NM 输出强度)
            grads = getattr(self, "_last_mac_grads", None) or {}
            def _fg(k): return grads.get(k)
            def _fmt(v): return f"{v:.1e}" if isinstance(v, float) else "—"
            mem_out_norms = []
            for pair in self.model.memory.values():
                t = pair.last_mem_out_norm
                if torch.is_tensor(t):
                    v = t.item()
                    if v > 0:
                        mem_out_norms.append(v)
            mem_norm_avg = sum(mem_out_norms) / len(mem_out_norms) if mem_out_norms else 0.0
            print(
                f"  [Step {self.global_step}] ema_loss={self._ema_loss:.4f} "
                f"ema_acc={self._ema_acc:.2%} lr={lr:.2e} "
                f"inject={inject:.3f} pmem={pmem_norm:.3f} mtok={mtok_norm:.3f} "
                f"|g_inj|={_fmt(_fg('mac_inject_logit'))} "
                f"|g_pm|={_fmt(_fg('persistent_mem'))} "
                f"|g_mt|={_fmt(_fg('mem_token_init'))} "
                f"|g_dec|={_fmt(_fg('to_decay_factor'))} "
                f"mem_norm={mem_norm_avg:.3f}"
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
        """v9 ckpt: memory_pair_state + persistent_mem(若启用)。"""
        if path is None:
            path = f"checkpoints/xinhe_step_{self.global_step}.pt"

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        memory_pair_state = self.model.memory.state_dict()

        checkpoint = {
            "global_step": self.global_step,
            "memory_pair_state": memory_pair_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "curriculum_stage": self.current_stage_name,
            "version": "v9",
        }
        if getattr(self.model, "persistent_mem", None) is not None:
            checkpoint["persistent_mem"] = self.model.persistent_mem.detach().cpu()
        if getattr(self.model, "mem_token_init", None) is not None:
            checkpoint["mem_token_init"] = self.model.mem_token_init.detach().cpu()
        if getattr(self.model, "mac_inject_logit", None) is not None:
            checkpoint["mac_inject_logit"] = self.model.mac_inject_logit.detach().cpu()

        torch.save(checkpoint, path)
        print(f"  [Checkpoint] 保存到 {path}")

    def load_checkpoint(self, path: str):
        """加载 checkpoint(v9 strict=True,不兼容 v8 hippocampus_state)。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "memory_pair_state" not in checkpoint:
            raise RuntimeError(
                "checkpoint 缺少 'memory_pair_state' 键。v9 不兼容 v8 hippocampus_state,"
                "请从零重训。"
            )
        self.model.memory.load_state_dict(checkpoint["memory_pair_state"], strict=True)

        # persistent_mem / mem_token_init 是 v9 + MAC 后引入,旧 ckpt 没有
        # (可选加载,缺则保留随机 init)
        if (getattr(self.model, "persistent_mem", None) is not None
                and "persistent_mem" in checkpoint):
            with torch.no_grad():
                self.model.persistent_mem.copy_(checkpoint["persistent_mem"].to(
                    self.model.persistent_mem.device
                ))
        if (getattr(self.model, "mem_token_init", None) is not None
                and "mem_token_init" in checkpoint):
            with torch.no_grad():
                self.model.mem_token_init.copy_(checkpoint["mem_token_init"].to(
                    self.model.mem_token_init.device
                ))
        if (getattr(self.model, "mac_inject_logit", None) is not None
                and "mac_inject_logit" in checkpoint):
            with torch.no_grad():
                self.model.mac_inject_logit.copy_(checkpoint["mac_inject_logit"].to(
                    self.model.mac_inject_logit.device
                ))

        self.global_step = checkpoint["global_step"]

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"[Checkpoint] 从 {path} 恢复, step={self.global_step}")
