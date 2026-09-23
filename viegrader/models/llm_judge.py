"""Chấm theo rubric bằng mô hình ngôn ngữ lớn (LLM-as-a-judge).

Vai trò trong kiến trúc lai:
    - Bổ sung năng lực đánh giá NỘI DUNG và LẬP LUẬN mà đặc trưng bề mặt không
      bắt được.
    - Sinh nhận xét bằng lời cho sinh viên.
    - KHÔNG dùng độc lập để ra điểm cuối: điểm LLM là một đầu vào của bộ tổng hợp,
      có trọng số học từ dữ liệu (xem ``ensemble.py``).

Ba kĩ thuật giảm thiên lệch đã cài đặt:
    1. **Few-shot bằng bài neo** - đưa kèm 2-4 bài mẫu đã có điểm hội đồng thống
       nhất, phủ các mức điểm khác nhau -> neo thang điểm của LLM.
    2. **Chấm từng tiêu chí, bắt buộc trích bằng chứng** trước khi cho điểm.
    3. **Tự nhất quán (self-consistency)** - gọi ``n_samples`` lần ở nhiệt độ
       thấp, lấy trung vị; độ phân tán = độ bất định.

Provider hỗ trợ: ``anthropic``, ``openai``, hoặc ``callable`` do người dùng truyền
vào (thuận tiện khi dùng mô hình nội bộ). Không có khoá API -> module tự tắt,
pipeline vẫn chạy bằng phần đặc trưng + PhoBERT.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..schema import Rubric

SYSTEM_PROMPT = """Bạn là giám khảo chấm bài kiểm tra tự luận đại học tại Việt Nam.
Bạn chấm NGHIÊM NGẶT theo rubric được cung cấp, không theo cảm tính.

Nguyên tắc bắt buộc:
- Với mỗi tiêu chí: TRÍCH bằng chứng nguyên văn từ bài làm TRƯỚC, rồi mới cho điểm.
- Chỉ được cho điểm bằng đúng các mức điểm mà rubric liệt kê cho tiêu chí đó.
- Không thưởng điểm cho độ dài, giọng văn hoa mỹ hay cách trình bày bắt mắt.
- Không phạt hai lần cùng một lỗi ở hai tiêu chí khác nhau.
- Không suy đoán về người viết. Bài đã được ẩn danh.
- Nếu bài lạc đề hoặc không đủ dữ kiện để chấm, ghi rõ trong trường "flags".

Chỉ trả về JSON hợp lệ, không kèm giải thích ngoài JSON."""


def build_rubric_block(rubric: Rubric) -> str:
    L = [f"RUBRIC: {rubric.name} (thang tổng {rubric.scale_max})"]
    for c in rubric.criteria:
        L.append(f"\n### Tiêu chí `{c.key}` — {c.name} "
                 f"(trọng số {c.weight:.0%}, tối đa {c.max_score} điểm)")
        if c.description:
            L.append(c.description.strip())
        L.append("Các mức điểm hợp lệ:")
        for lv in sorted(c.levels, key=lambda x: -x.score):
            L.append(f"  - {lv.score} ({lv.name}): {lv.descriptor}")
    return "\n".join(L)


def build_user_prompt(
    essay: str,
    rubric: Rubric,
    prompt_text: str = "",
    answer_key: str = "",
    anchors: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    parts: List[str] = [build_rubric_block(rubric)]
    if prompt_text:
        parts.append(f"\n=== ĐỀ BÀI ===\n{prompt_text.strip()}")
    if answer_key:
        parts.append(f"\n=== ĐÁP ÁN / Ý CỐT LÕI CẦN CÓ ===\n{answer_key.strip()}")

    if anchors:
        parts.append("\n=== BÀI MẪU ĐÃ CÓ ĐIỂM CHUẨN (dùng để neo thang điểm) ===")
        for i, a in enumerate(anchors, 1):
            body = str(a.get("text", ""))[:1500]
            parts.append(f"\n--- Bài mẫu {i} (điểm tổng chuẩn: {a.get('total')}) ---\n{body}")
            if a.get("scores"):
                parts.append(f"Điểm thành phần: {json.dumps(a['scores'], ensure_ascii=False)}")

    parts.append(f"\n=== BÀI CẦN CHẤM ===\n{essay.strip()}")

    schema = {
        "criteria": {
            c.key: {
                "evidence": ["trích dẫn nguyên văn từ bài làm"],
                "level": "tên mức",
                "score": c.level_scores()[-1] if c.levels else c.max_score,
                "comment": "nhận xét ngắn gọn bằng tiếng Việt",
            }
            for c in rubric.criteria
        },
        "flags": ["LAC_DE | SAO_CHEP | QUA_NGAN | KHONG_DOC_DUOC (nếu có)"],
        "overall_comment": "nhận xét tổng, 2-4 câu, nêu 1 điểm mạnh và 2 việc cần sửa",
        "confidence": 0.0,
    }
    parts.append(
        "\n=== YÊU CẦU ĐẦU RA ===\nTrả về JSON theo đúng cấu trúc sau:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    provider: str = "anthropic"          # anthropic | openai | custom | off
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.2
    max_tokens: int = 2000
    n_samples: int = 3                   # self-consistency
    api_key_env: str = "ANTHROPIC_API_KEY"


class LLMJudge:
    def __init__(
        self,
        rubric: Rubric,
        cfg: Optional[LLMConfig] = None,
        call_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.rubric = rubric
        self.cfg = cfg or LLMConfig()
        self._call = call_fn or self._build_client()

    # ---------------------------------------------------------------- #
    def _build_client(self) -> Optional[Callable[[str, str], str]]:
        if self.cfg.provider == "off":
            return None
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            return None
        if self.cfg.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                return None
            client = anthropic.Anthropic(api_key=key)

            def _call(system: str, user: str) -> str:
                r = client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return r.content[0].text
            return _call

        if self.cfg.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                return None
            client = OpenAI(api_key=key)

            def _call(system: str, user: str) -> str:
                r = client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return r.choices[0].message.content or ""
            return _call
        return None

    @property
    def available(self) -> bool:
        return self._call is not None

    # ---------------------------------------------------------------- #
    @staticmethod
    def _parse_json(txt: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            except Exception:
                return None

    def _snap(self, key: str, value: Any) -> float:
        c = self.rubric.get(key)
        allowed = c.level_scores() or [0.0, c.max_score]
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(min(allowed))
        return float(min(allowed, key=lambda a: abs(a - v)))

    def score(
        self,
        essay: str,
        prompt_text: str = "",
        answer_key: str = "",
        anchors: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Chấm 1 bài. Trả về None nếu LLM không khả dụng."""
        if not self.available:
            return None
        user = build_user_prompt(essay, self.rubric, prompt_text, answer_key, anchors)

        runs: List[Dict[str, Any]] = []
        for _ in range(max(1, self.cfg.n_samples)):
            try:
                raw = self._call(SYSTEM_PROMPT, user)
            except Exception:
                continue
            d = self._parse_json(raw)
            if d:
                runs.append(d)
        if not runs:
            return None

        out: Dict[str, Any] = {"criteria": {}, "flags": [], "n_runs": len(runs)}
        spread: List[float] = []
        for c in self.rubric.criteria:
            vals, evid, comments, levels = [], [], [], []
            for r in runs:
                node = (r.get("criteria") or {}).get(c.key) or {}
                if "score" in node:
                    vals.append(self._snap(c.key, node["score"]))
                ev = node.get("evidence") or []
                evid.extend(ev if isinstance(ev, list) else [str(ev)])
                if node.get("comment"):
                    comments.append(str(node["comment"]))
                if node.get("level"):
                    levels.append(str(node["level"]))
            if not vals:
                continue
            med = statistics.median(vals)
            spread.append((max(vals) - min(vals)) / max(c.max_score, 1e-9))
            out["criteria"][c.key] = {
                "score": self._snap(c.key, med),
                "level": statistics.mode(levels) if levels else "",
                "evidence": evid[:3],
                "comment": comments[0] if comments else "",
                "runs": vals,
            }
        for r in runs:
            for f in r.get("flags", []) or []:
                if isinstance(f, str) and f.strip() and "nếu có" not in f:
                    out["flags"].append(f.strip())
        out["flags"] = sorted(set(out["flags"]))
        out["overall_comment"] = next(
            (r.get("overall_comment", "") for r in runs if r.get("overall_comment")), ""
        )
        out["confidence"] = round(1.0 - (statistics.mean(spread) if spread else 0.0), 3)
        return out

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        return [
            self.score(
                it["text"], it.get("prompt_text", ""), it.get("answer_key", ""),
                it.get("anchors"),
            )
            for it in items
        ]
