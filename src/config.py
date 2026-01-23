from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml

@dataclass
class Paths:
    csv: str
    heatmaps_dir: str
    artifacts_dir: str
    reports_dir: str

@dataclass
class ImageCfg:
    size: int = 64
    normalize: bool = True
    augment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingCfg:
    device: str = "cuda"
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    label_smoothing: float = 0.0
    backbone: str = "resnet18"
    clip_grad_norm: float = 1.0
    scheduler: str = "onecycle"
    amp: bool = True
    compile: bool = True

@dataclass
class LossCfg:
    type: str = "bce"
    gamma: float = 2.0
    alpha: float = 0.25

@dataclass
class FeaturesCfg:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)

@dataclass
class Config:
    seed: int
    n_splits: int
    use_group_kfold: bool
    pos_weight: float
    paths: Paths
    image: ImageCfg
    training: TrainingCfg
    loss: LossCfg
    features: FeaturesCfg
    threshold: float = 0.5

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return Config(
            seed=raw["seed"],
            n_splits=raw["n_splits"],
            use_group_kfold=raw.get("use_group_kfold", False),
            pos_weight=float(raw.get("pos_weight", 1.0)),
            paths=Paths(**raw["paths"]),
            image=ImageCfg(**raw["image"]),
            training=TrainingCfg(**raw["training"]),
            loss=LossCfg(**raw.get("loss", {})),
            features=FeaturesCfg(**raw["features"]),
            threshold=float(raw.get("threshold", 0.5)),
        )
