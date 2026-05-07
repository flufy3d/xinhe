"""测试 CongliuGenerator 字段兼容性 + schema 输出。"""
import json
import tempfile
from pathlib import Path

import pytest

from xinhe.data.generators.base import GenerateRequest
from xinhe.data.generators.congliu.generator import CongliuGenerator


def _write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_congliu_extracts_input_output():
    """优先级:input + output 字段。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw.jsonl"
        _write_jsonl([
            {"input": "什么是函数?", "output": "<think>函数是数学映射。</think>函数是定义域到值域的映射。" * 5},
            {"input": "证明 1+1=2", "output": "<think>" + "推导..." * 50 + "</think>2" * 30},
        ], raw)

        gen = CongliuGenerator(max_turns=8, raw_path=str(raw),
                               min_asst_chars=20, max_asst_chars=10000)
        out = tmp / "out.jsonl"
        req = GenerateRequest(
            out_path=out, n_samples=2, seed=42, split="train",
            rejected_path=tmp / "rej.jsonl", max_turns=8,
        )
        kept, rejected = gen.generate(req)
        assert kept == 2
        with open(out, "r", encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 2
        for sample in lines:
            assert sample["stage"] == "congliu"
            convs = sample["conversations"]
            assert convs[0]["role"] == "user"
            assert convs[0]["content"] in ("什么是函数?", "证明 1+1=2")
            assert convs[1]["role"] == "assistant"
            assert convs[1]["train_loss"] == "true"
            assert convs[1]["value"] is None


def test_congliu_extracts_alternate_field_names():
    """字段 fallback:instruction + content / prompt + response 都能识别。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw.jsonl"
        _write_jsonl([
            {"instruction": "Q1", "content": "答案 A " * 50},
            {"prompt": "Q2", "response": "答案 B " * 50},
        ], raw)

        gen = CongliuGenerator(max_turns=8, raw_path=str(raw),
                               min_asst_chars=10, max_asst_chars=10000)
        out = tmp / "out.jsonl"
        req = GenerateRequest(
            out_path=out, n_samples=2, seed=42, split="train",
            rejected_path=tmp / "rej.jsonl", max_turns=8,
        )
        kept, _ = gen.generate(req)
        assert kept == 2


def test_congliu_filters_too_short_or_long():
    """min_asst_chars / max_asst_chars 过滤生效。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw.jsonl"
        _write_jsonl([
            {"input": "Q", "output": "短答"},                          # 太短
            {"input": "Q", "output": "好" * 5000},                      # 太长
            {"input": "Q", "output": "正常长度答案。" * 30},               # OK
        ], raw)

        gen = CongliuGenerator(max_turns=8, raw_path=str(raw),
                               min_asst_chars=100, max_asst_chars=2000)
        out = tmp / "out.jsonl"
        req = GenerateRequest(
            out_path=out, n_samples=10, seed=42, split="train",
            rejected_path=tmp / "rej.jsonl", max_turns=8,
        )
        kept, _ = gen.generate(req)
        # 仅 1 个样本通过(短答 / 超长被 generator 内部过滤,不计入 write_jsonl rejected)
        assert kept == 1, f"期望保留 1 个,实际 {kept}"


def test_congliu_reasoning_content_assembled():
    """Congliu 实际 schema 是 input + reasoning_content + content 三段;
    generator 应拼成 <think>{reasoning}</think>\\n\\n{answer} R1 标准格式。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw.jsonl"
        _write_jsonl([
            {
                "input": "什么是 1+1?",
                "reasoning_content": "我来想想,加法的基本性质是..." * 10,
                "content": "答案是 2。",
            },
        ], raw)
        gen = CongliuGenerator(max_turns=8, raw_path=str(raw),
                               min_asst_chars=50, max_asst_chars=10000)
        out = tmp / "out.jsonl"
        req = GenerateRequest(
            out_path=out, n_samples=1, seed=42, split="train",
            rejected_path=tmp / "rej.jsonl", max_turns=8,
        )
        gen.generate(req)
        with open(out, "r", encoding="utf-8") as f:
            sample = json.loads(f.read().strip())
        asst = sample["conversations"][1]["content"]
        assert asst.startswith("<think>"), f"未拼出 R1 think tag,实际开头: {asst[:50]}"
        assert "</think>" in asst
        assert "答案是 2。" in asst
        # think 段先于 answer
        assert asst.index("</think>") < asst.index("答案是 2。")


def test_congliu_no_path_raises():
    """raw_path 缺失时清晰报错。"""
    gen = CongliuGenerator(max_turns=8, raw_path=None)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.jsonl"
        req = GenerateRequest(
            out_path=out, n_samples=1, seed=42, split="train",
            rejected_path=Path(tmp) / "rej.jsonl", max_turns=8,
        )
        with pytest.raises(RuntimeError, match=r"\[congliu\] 缺少 raw_path"):
            gen.generate(req)
