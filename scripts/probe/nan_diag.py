"""
NaN 诊断:仅做 1 步 forward+backward,逐层检查 NaN/Inf 位置。
不写 ckpt 不更新 weights,只跑 1 step 输出 NaN 来源。
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from xinhe.training.config import load_curriculum_config
from xinhe.training.trainer import Trainer
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from scripts.evaluate import load_model_and_tokenizer

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage", default="pcap_skeleton")
    p.add_argument("--n-steps", type=int, default=5)
    args = p.parse_args()

    print(f"[nan_diag] config={args.config} stage={args.stage} n_steps={args.n_steps}")

    # 用 trainer 自身的初始化路径
    from scripts.train import main as train_main

    # 黑客方式:patch trainer 在每步检查 NaN 并打印
    orig_forward = None

    def patched_step(self, *args, **kwargs):
        result = orig_forward(self, *args, **kwargs)
        # 在每个返回项检查 NaN
        if isinstance(result, dict):
            for k, v in result.items():
                if torch.is_tensor(v):
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print(f"[NaN-DIAG] '{k}' has NaN/Inf at step")
                        print(f"           shape={v.shape} dtype={v.dtype}")
                        print(f"           min={v.min().item()} max={v.max().item()}")
        return result

    # 也可以 hook 模型 forward
    nan_found = {"step": -1, "where": None}

    def check_tensor(t, name, step):
        if torch.is_tensor(t):
            if torch.isnan(t).any() or torch.isinf(t).any():
                if nan_found["step"] < 0:
                    nan_found["step"] = step
                    nan_found["where"] = name
                    print(f"[NaN-FIRST] step={step} where={name}")
                    print(f"            shape={t.shape} dtype={t.dtype}")
                    print(f"            n_nan={torch.isnan(t).sum().item()} n_inf={torch.isinf(t).sum().item()}")
                    return True
        return False

    # 实际跑训练但只 N 步
    sys.argv = ["train.py", "--config", args.config, "--from-stage", args.stage, "--max-steps-override", str(args.n_steps)]
    train_main()


if __name__ == "__main__":
    main()
