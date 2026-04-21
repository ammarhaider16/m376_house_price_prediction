from dataclasses import dataclass
from typing import Callable
from torch import nn

@dataclass
class TrainConfig:
    model_name:   str
    layers_fn:    Callable[[], nn.ModuleList]
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    num_epochs:   int   = 100
    val_size:     float = 0.1

    def get_model_id(self) -> str:
        model_id = f"{self.model_name}_lr{self.lr}_wd{self.weight_decay}_bs{self.batch_size}"
        if self.num_epochs != 100:
            model_id += f"_ep{self.num_epochs}"
        return model_id