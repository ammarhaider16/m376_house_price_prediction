from ffnn.layers import get_layers_A, get_layers_B, get_layers_C, get_layers_D, get_layers_E
from ffnn.config import TrainConfig
from ffnn.train import run_train
from ffnn.test import run_test

configs = [
    TrainConfig(model_name="A_default", layers_fn=get_layers_A, lr=1e-3),
    TrainConfig(model_name="A_lr_low", layers_fn=get_layers_A, lr=1e-4),
    TrainConfig(model_name="B_default", layers_fn=get_layers_B, lr=1e-3),
    TrainConfig(model_name="C_default", layers_fn=get_layers_C, lr=1e-3),
    TrainConfig(model_name="A_lr_high", layers_fn=get_layers_A, lr=1e-2),
    TrainConfig(model_name="C_lr_high", layers_fn=get_layers_C, lr=1e-2),
    TrainConfig(model_name="A_lr_vhigh", layers_fn=get_layers_A, lr=1e-1),
    TrainConfig(model_name="C_lr_vhigh", layers_fn=get_layers_C, lr=1e-1),
    TrainConfig(model_name="A_no_wd", layers_fn=get_layers_B, lr=1e-3, weight_decay=0.0),
    TrainConfig(model_name="A_wd_high", layers_fn=get_layers_B, lr=1e-3, weight_decay=1e-2),
    TrainConfig(model_name="A_wd_med", layers_fn=get_layers_B, lr=1e-3, weight_decay=1e-3),
    TrainConfig(model_name="A_default_64", layers_fn=get_layers_A, lr=1e-3, batch_size=64),
    TrainConfig(model_name="C_default_64", layers_fn=get_layers_C, lr=1e-3, batch_size=64),
    TrainConfig(model_name="B_lr_high_64", layers_fn=get_layers_B, lr=1e-2, batch_size=64),
    TrainConfig(model_name="A_lr_double", layers_fn=get_layers_A, lr=0.002),
    TrainConfig(model_name="C_lr_double", layers_fn=get_layers_C, lr=0.002),
    TrainConfig(model_name="D_default", layers_fn=get_layers_D, lr=1e-3),
    TrainConfig(model_name="E_default", layers_fn=get_layers_E, lr=1e-3),
]

configs.extend([
    TrainConfig(model_name=model_name, layers_fn=layers_fn, lr=lr)
    for lr in [0.001, 0.0005, 0.0015]
    for model_name, layers_fn in [
        ("A_sweep", get_layers_A,),
        ("C_sweep", get_layers_C),
    ]
])

for cfg in configs:
    run_train(cfg)
run_test(configs, checkpoint="best", save_predictions=True)