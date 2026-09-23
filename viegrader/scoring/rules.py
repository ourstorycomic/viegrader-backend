"""Rule engine thi hành QUY ĐỊNH CHẤM.

Vì sao cần tách khỏi mô hình học máy: quy chế đào tạo có những điều khoản mang
tính pháp lí (bài trống 0 điểm, nghi sao chép chuyển hội đồng, bài quá ngắn bị
giới hạn điểm...). Những điều này KHÔNG được để mô hình "học lấy" - phải thi hành
tuyệt đối, kiểm tra được và ghi vết được.

Biểu thức ``when`` được đánh giá trong môi trường hạn chế: chỉ có các biến đặc
trưng/điểm và một số hàm toán học an toàn. Không truy cập được builtins, import
hay thuộc tính - tránh rủi ro thực thi mã tuỳ ý từ file rubric.
"""

from __future__ import annotations

import ast
import math
import operator as op
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Rubric, ScoreResult

# --------------------------------------------------------------------------- #
_ALLOWED_BINOP = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_ALLOWED_CMP = {
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge, ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_ALLOWED_FUNCS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "sqrt": math.sqrt, "log": math.log, "int": int, "float": float,
}


class UnsafeExpression(ValueError):
    pass


def safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    """Đánh giá biểu thức Python giới hạn trên môi trường ``env``."""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        raise UnsafeExpression(f"Biểu thức sai cú pháp: {expr!r} ({e})") from e

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            raise UnsafeExpression(f"Biến chưa khai báo trong biểu thức: {n.id}")
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOP:
            return _ALLOWED_BINOP[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return -ev(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +ev(n.operand)
            if isinstance(n.op, ast.Not):
                return not ev(n.operand)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for o, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                if type(o) not in _ALLOWED_CMP:
                    raise UnsafeExpression(f"Toán tử so sánh không cho phép: {o}")
                if not _ALLOWED_CMP[type(o)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = _ALLOWED_FUNCS.get(n.func.id)
            if fn is None:
                raise UnsafeExpression(f"Hàm không cho phép: {n.func.id}")
            return fn(*[ev(a) for a in n.args])
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        raise UnsafeExpression(f"Cấu trúc không cho phép trong biểu thức: {type(n).__name__}")

    return ev(node)


# --------------------------------------------------------------------------- #
@dataclass
class RuleOutcome:
    applied: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    review: bool = False
    reasons: List[str] = field(default_factory=list)


def apply_rules(
    result: ScoreResult,
    rubric: Rubric,
    env: Dict[str, Any],
    strict: bool = False,
) -> Tuple[ScoreResult, RuleOutcome]:
    """Thi hành các quy định chấm lên một kết quả đã có điểm mô hình.

    ``env`` chứa đặc trưng của bài (n_words, sim_prompt, dup_ratio...) cộng thêm
    các biến hệ thống: total, confidence, scale_max, min_words, max_words và điểm
    từng tiêu chí theo key.
    """
    out = RuleOutcome()
    env = dict(env)
    env.setdefault("scale_max", rubric.scale_max)
    for cs in result.criteria:
        env[cs.key] = cs.score
    env["total"] = result.total
    env["confidence"] = result.confidence

    for rule in rubric.rules:
        rid = rule.get("id", "?")
        cond = rule.get("when", "")
        try:
            hit = bool(safe_eval(cond, env)) if cond else False
        except UnsafeExpression as e:
            if strict:
                raise
            out.reasons.append(f"[{rid}] bỏ qua: {e}")
            continue

        if not hit:
            continue

        action = rule.get("action", "flag")
        value = rule.get("value")
        if "value_expr" in rule:
            value = safe_eval(str(rule["value_expr"]), env)

        if action == "cap_total":
            result.total = min(result.total, float(value))
        elif action == "subtract":
            result.total = max(0.0, result.total - float(value))
        elif action == "set_total":
            result.total = float(value)
        elif action == "set_criterion":
            tgt = rule.get("target")
            for cs in result.criteria:
                if cs.key == tgt:
                    cs.score = float(value)
                    cs.comment = (cs.comment + " " + rule.get("description", "")).strip()
            result.total = recompute_total(result, rubric)
        elif action == "review":
            out.review = True
            if rule.get("reason"):
                out.reasons.append(str(rule["reason"]))
        elif action == "flag":
            pass
        else:
            out.reasons.append(f"[{rid}] hành động không nhận dạng: {action}")

        if rule.get("flag"):
            out.flags.append(str(rule["flag"]))
        out.applied.append(rid)
        env["total"] = result.total

        if rule.get("stop"):
            break

    result.flags = sorted(set(result.flags + out.flags))
    result.applied_rules = out.applied
    if out.review:
        result.needs_human_review = True
        result.review_reason = "; ".join(dict.fromkeys(out.reasons))
    return result, out


def recompute_total(result: ScoreResult, rubric: Rubric) -> float:
    t = 0.0
    for cs in result.criteria:
        c = rubric.get(cs.key)
        t += (cs.score / c.max_score) * c.weight * rubric.scale_max
    return float(t)


def check_rules(rubric: Rubric, sample_env: Optional[Dict[str, Any]] = None) -> List[str]:
    """Kiểm tra cú pháp và biến của toàn bộ rule trước khi đưa vào vận hành."""
    env = {
        "n_words": 500.0, "sim_prompt": 0.5, "dup_ratio": 0.1, "prompt_copy_ratio": 0.1,
        "spell_error_rate": 0.02, "confidence": 0.8, "total": 7.0,
        "scale_max": rubric.scale_max, "min_words": 300.0, "max_words": 1500.0,
        "keyword_coverage": 0.7, "n_paragraphs": 4.0,
        "snap_gap": 0.3, "max_snap_gap": 0.4, "pred_std": 0.2,
        "teencode_rate": 0.01, "informal_rate": 0.01, "diacritic_ratio": 0.25,
        "ai_style_score": 0.0, "cliche_density": 0.5, "has_citation": 0.0,
        "mean_sentence_len": 22.0, "ttr": 0.5, "structure_score": 0.7,
        "connective_density": 2.0, "sim_reference": 0.4, "sim_centroid": 0.6,
    }
    for c in rubric.criteria:
        env[c.key] = c.max_score * 0.7
    if sample_env:
        env.update(sample_env)

    problems: List[str] = []
    for rule in rubric.rules:
        rid = rule.get("id", "?")
        try:
            safe_eval(rule.get("when", "True"), env)
        except UnsafeExpression as e:
            problems.append(f"[{rid}] điều kiện lỗi: {e}")
        if "value_expr" in rule:
            try:
                safe_eval(str(rule["value_expr"]), env)
            except UnsafeExpression as e:
                problems.append(f"[{rid}] value_expr lỗi: {e}")
        if rule.get("action") == "set_criterion" and rule.get("target") not in rubric.keys:
            problems.append(f"[{rid}] target '{rule.get('target')}' không có trong rubric")
    return problems
