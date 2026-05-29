"""
Trainer — 训练循环(单全局 QueryHead + HippoDelta 架构)

核心特性:
- state 跨 turn 传递,state 是 XinheMemoryState(只在 _global_write_idx 层持有 HippoDeltaState)
- 截断 BPTT: 每 tbptt_turns 轮做 detach + backward + step
- 训练参数:QueryHead + HippoDelta + W_mac/W_mal + MAL α + LoRA(qkvo)
  + per-layer K/V(K_pers/V_pers,Plan B 关停)
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
        """单全局架构没有 per-layer memory ModuleDict;freeze 钩子保留作占位,Plan B 不用。"""
        # plugin_lr_multiplier=0 走 _build_optimizer 时 lr=0,等效冻结 memory 组。
        _ = config  # 留作未来 freeze 钩子的入口

    def _build_optimizer(self, config: XinheConfig) -> torch.optim.AdamW:
        """构建 optimizer:trainable 参数单组 lr × plugin_lr_multiplier。"""
        lr = config.learning_rate
        plugin_mult = getattr(config, "plugin_lr_multiplier", 1.0)

        plugin_params = self.model.get_trainable_params()
        param_groups = []
        if plugin_params:
            param_groups.append({"params": plugin_params, "lr": lr * plugin_mult})
        # fused=True:CUDA fused 内核,m/v/master 更新单 kernel,省 ~200MB 临时缓冲 + 2-3× 加速
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay, fused=True)

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

        # NOTE: trainer 顶层不开 torch.compile,只用 `compile_backbone_layers=True`
        # 局部 compile(qwen full_attention layer body),HippoDelta 等记忆模块在 compile 边界外。

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

        if self._early_stopped:
            print(f"训练已收敛, 共 {self.global_step} 步")
        else:
            print(f"训练完成, 共 {self.global_step} 步")

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
        # 改 tensor 累加器:每 turn 不再 .item() sync(每 turn 一次 sync × 16 turn = 32 次/ep
        # GPU pipeline 卡顿)。只在 backward 边界(已天然 sync)取出 Python float
        episode_correct_t = torch.zeros((), dtype=torch.long, device=self.device)
        episode_total_t = torch.zeros((), dtype=torch.long, device=self.device)
        loss_turns_t = torch.zeros((), dtype=torch.long, device=self.device)

        for turn_idx, batch in enumerate(episode_turns):
            turn_ids, labels, weights = batch
            turn_ids = turn_ids.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            if turn_idx > 0 and turn_idx % self.config.tbptt_turns == 0:
                # 一次性 sync:loss_turns / accuracy 都跟 backward 在同一 sync 点取出
                loss_turns_int = int(loss_turns_t.item())
                if loss_turns_int > 0:
                    avg_loss = accumulated_loss / loss_turns_int
                    (avg_loss / self.config.grad_accum_steps).backward()
                    avg_loss_v = avg_loss.item()
                    episode_total_loss += avg_loss_v
                    # 传 running acc:混合数据时 grad_accum cycle 可能落在 mid-ep call,
                    # EMA 必须得到真实 partial accuracy,否则会被 default 1.0 污染成假 100%
                    ec = int(episode_correct_t.item())
                    et = int(episode_total_t.item())
                    running_acc = ec / max(et, 1)
                    self._maybe_optimizer_step(avg_loss_v, running_acc)

                state = state.detach()
                accumulated_loss = torch.tensor(0.0, device=self.device)
                loss_turns_t = torch.zeros((), dtype=torch.long, device=self.device)

            # Memory Dropout(可选):随机把这 turn 强制 NM-zero,backbone 不能假设 mem 总在。
            # 与 shortcut_suppression 互补 — implicit curriculum
            do_drop = (self.config.memory_dropout > 0
                       and float(torch.rand(()).item()) < self.config.memory_dropout)
            this_mem_override = 0.0 if do_drop else None

            with torch.amp.autocast("cuda", dtype=self.dtype):
                # model.forward 内部仍叫 segment（纯实现细节，与业务 turn 解耦）
                # compute_logits=False 走 cut_cross_entropy 快路径,不材化 (B,T,V=248320) logits
                # 4 turns 累积省 ~3GB(probe 实测)。trainer 不读 result["logits"]
                result = self.model(
                    turn_ids, state, labels=labels, pad_token_id=self.pad_token_id,
                    weights=weights, compute_logits=False,
                    mem_alpha_override=this_mem_override,
                )

            state = result["state_next"]
            turn_loss = result["loss"]

            # Margin-Based Shortcut Suppression(直接 attack 'NM-on==NM-zero' 退化解):
            # NM-zero forward **带 grad**(关键!detach 会让 penalty 只是放大 loss_on,失去意义)。
            # penalty = max(0, loss_on - loss_zero + margin),active 时:
            #   d(loss_on)/d(params) 推主路径优化(NM-on 更准)
            #   d(-loss_zero)/d(params) **推 backbone 在 NM-zero 模式下变差**:
            #     因 mem_alpha=0 时 mem 通路全乘 0,梯度只能落 backbone 的 LoRA / K_pers
            # → 直接 attack backbone shortcut:不让 backbone 学到"不靠 mem 也能答对"
            # 代价:dual backward,GPU mem ~1.5x。
            # 与 memory_dropout 互斥:do_drop=True 时这 turn 本身是 NM-zero,baseline 无意义。
            if self.config.shortcut_suppression and not do_drop:
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    result_zero = self.model(
                        turn_ids, state, labels=labels,
                        pad_token_id=self.pad_token_id, weights=weights,
                        compute_logits=False, mem_alpha_override=0.0,
                    )
                loss_zero = result_zero["loss"]    # 不 detach!grad 通到 backbone
                margin = self.config.shortcut_margin
                lam = self.config.shortcut_lambda
                penalty = torch.clamp(turn_loss - loss_zero + margin, min=0.0)
                # gap > 0 = mem 真有用;gap ≤ 0 = mem 没用(penalty 在推梯度让它变有用)
                self._last_shortcut_gap = float((loss_zero.detach() - turn_loss.detach()).item())
                self._last_shortcut_penalty = float(penalty.detach().item())
                turn_loss = turn_loss + lam * penalty

            accumulated_loss = accumulated_loss + turn_loss
            correct = result.get("correct", 0)
            total = result.get("total", 0)
            # tensor accum,不 .item();非 tensor 路径(常量 0)直接加
            if torch.is_tensor(correct):
                episode_correct_t = episode_correct_t + correct.long()
            else:
                episode_correct_t = episode_correct_t + int(correct)
            if torch.is_tensor(total):
                episode_total_t = episode_total_t + total.long()
            else:
                episode_total_t = episode_total_t + int(total)
            # 用 tensor 累加 loss_turns,避开 .any() 的 .__bool__() sync
            has_valid_labels = (labels != -100).any().long()
            loss_turns_t = loss_turns_t + has_valid_labels

        loss_turns_int = int(loss_turns_t.item())
        if loss_turns_int > 0:
            avg_loss = accumulated_loss / loss_turns_int
            (avg_loss / self.config.grad_accum_steps).backward()
            avg_loss_v = avg_loss.item()
            episode_total_loss += avg_loss_v
            ec = int(episode_correct_t.item())
            et = int(episode_total_t.item())
            acc = ec / max(et, 1)
            self._maybe_optimizer_step(avg_loss_v, acc)

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
        """诊断:捕获 LoRA + per-layer K/V 的梯度 norm。
        在 optimizer.step() 前调用(grad 已 clip)。grad 为 None 表示该参数本步无梯度。
        """
        out = {}
        # LoRA grad norm 聚合(所有 lora_A.grad 的均值,验证 LoRA 在学)
        lora_norms = []
        kpers_norms = []
        for n, p in self.model.backbone.named_parameters():
            if p.grad is None:
                continue
            if "lora_A" in n or "lora_B" in n:
                lora_norms.append(p.grad.detach().float().norm().item())
            elif "K_pers" in n or "V_pers" in n:
                kpers_norms.append(p.grad.detach().float().norm().item())
        out["lora"] = sum(lora_norms) / len(lora_norms) if lora_norms else None
        out["K_pers"] = sum(kpers_norms) / len(kpers_norms) if kpers_norms else None
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
            # 通路效度指标:LoRA/K_pers 梯度 norm 是否在学
            grads = getattr(self, "_last_mac_grads", None) or {}
            def _fg(k): return grads.get(k)
            def _fmt(v): return f"{v:.1e}" if isinstance(v, float) else "—"
            print(
                f"  [Step {self.global_step}] ema_loss={self._ema_loss:.4f} "
                f"ema_acc={self._ema_acc:.2%} lr={lr:.2e} "
                f"|g_lora|={_fmt(_fg('lora'))} "
                f"|g_kp|={_fmt(_fg('K_pers'))}"
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
        """单全局架构 ckpt:qhead_state + backbone addons(LoRA + per-layer K/V)。"""
        if path is None:
            path = f"checkpoints/xinhe_step_{self.global_step}.pt"

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "curriculum_stage": self.current_stage_name,
            "version": "v-next",
        }
        # backbone addons(LoRA lora_A/B + per-layer K/V K_pers/V_pers):
        # 只取 trainable 的子张量(原 backbone weight frozen 不存)
        backbone_addons = {
            k: v.detach().cpu() for k, v in self.model.backbone.state_dict().items()
            if ("lora_A" in k or "lora_B" in k
                or "K_pers" in k or "V_pers" in k)
        }
        if backbone_addons:
            checkpoint["backbone_addons_state"] = backbone_addons
        # 单全局模块:QueryHead + HippoDelta + 注入投影(W_mac/W_mal) + MAL α
        m = self.model
        qh = {
            "hippo_impl": "delta",
            "query_head": m.query_head.state_dict(),
            "W_mac": m.W_mac.state_dict(),
            "W_mal": m.W_mal.state_dict(),
            "global_mem_rmsnorm": m.global_mem_rmsnorm.state_dict(),
            "mal_alpha_logit": m.mal_alpha_logit.detach().cpu(),
            "global_hippo": m.global_hippo.state_dict(),
        }
        checkpoint["qhead_state"] = qh

        torch.save(checkpoint, path)
        print(f"  [Checkpoint] 保存到 {path} "
              f"(addons {len(backbone_addons)} 张量)")

    def load_checkpoint(self, path: str):
        """加载 checkpoint(单全局架构;只读 qhead_state + backbone_addons_state)。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "qhead_state" not in checkpoint:
            raise RuntimeError(
                "checkpoint 缺少 'qhead_state' 键。单全局架构不兼容 v9.5 memory_pair_state,"
                "请从零重训。"
            )

        # backbone addons(LoRA + per-layer K/V),strict=False 兼容老 backbone
        if "backbone_addons_state" in checkpoint:
            addons = {
                k: v.to(self.device) for k, v in checkpoint["backbone_addons_state"].items()
            }
            missing, unexpected = self.model.backbone.load_state_dict(addons, strict=False)
            print(f"[Checkpoint] LoRA/K_pers 加载 {len(addons)} 个张量 "
                  f"(unexpected={len(unexpected)})")

        # 单全局模块
        qh = checkpoint["qhead_state"]
        m = self.model
        m.query_head.load_state_dict(qh["query_head"])
        m.W_mac.load_state_dict(qh["W_mac"])
        m.W_mal.load_state_dict(qh["W_mal"])
        m.global_mem_rmsnorm.load_state_dict(qh["global_mem_rmsnorm"])
        with torch.no_grad():
            m.mal_alpha_logit.copy_(qh["mal_alpha_logit"].to(m.mal_alpha_logit.device))
        m.global_hippo.load_state_dict(qh["global_hippo"])
        print(f"[Checkpoint] QueryHead 单全局模块加载(hippo_impl={qh.get('hippo_impl')})")

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
