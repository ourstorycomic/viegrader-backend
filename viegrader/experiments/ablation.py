from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AblationConfig:
    experiment_id: str
    encoder: str = "none"
    use_feature_model: bool = False
    use_vistral: bool = False
    use_qlora: bool = False
    use_rag: bool = False
    use_anchors: bool = False
    use_self_consistency: bool = False
    use_rule_engine: bool = True
    seed: int = 42

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_ablation_configs() -> List[AblationConfig]:
    """Ma trận A–H mặc định; có thể thay bằng YAML của đề cương nghiên cứu."""
    return [
        AblationConfig("A", encoder="tfidf", use_feature_model=True),
        AblationConfig("B", encoder="phobert", use_feature_model=True),
        AblationConfig("C", use_vistral=True),
        AblationConfig("D", use_vistral=True, use_anchors=True),
        AblationConfig("E", use_vistral=True, use_anchors=True, use_rag=True),
        AblationConfig("F", use_vistral=True, use_qlora=True),
        AblationConfig("G", use_vistral=True, use_qlora=True, use_rag=True),
        AblationConfig("H", encoder="phobert", use_feature_model=True, use_vistral=True,
                       use_qlora=True, use_rag=True, use_anchors=True,
                       use_self_consistency=True),
    ]
