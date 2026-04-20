from torch import nn

INPUT_FEATURES = 12

def get_layers_A() -> nn.ModuleList:
    return nn.ModuleList([
    nn.Sequential(nn.Linear(INPUT_FEATURES, 64), nn.ReLU()), # input -> hidden 1
    nn.Sequential(nn.Linear(64, 32), nn.ReLU()),             # hidden 1 -> hidden 2
    nn.Linear(32, 1),                                        # hidden 2 -> output
    ])

def get_layers_B() -> nn.ModuleList:
    return nn.ModuleList([
    nn.Sequential(nn.Linear(INPUT_FEATURES, 32), nn.ReLU()), # input -> hidden 1
    nn.Linear(32, 1),                                        # hidden 1 -> output
    ])

def get_layers_C() -> nn.ModuleList:
    return nn.ModuleList([
    nn.Sequential(nn.Linear(INPUT_FEATURES, 128), nn.ReLU()), # input -> hidden 1
    nn.Sequential(nn.Linear(128, 64), nn.ReLU()),             # hidden 1 -> hidden 2
    nn.Sequential(nn.Linear(64, 32), nn.ReLU()),              # hidden 2 -> hidden 3
    nn.Linear(32, 1),                                         # hidden 3 -> output
    ])

# Different layer dimensions
def get_layers_D() -> nn.ModuleList:
    return nn.ModuleList([
    nn.Sequential(nn.Linear(INPUT_FEATURES, 32), nn.ReLU()), # input -> hidden 1
    nn.Sequential(nn.Linear(32, 16), nn.ReLU()),             # hidden 1 -> hidden 2
    nn.Linear(16, 1),                                        # hidden 2 -> output
    ])

# Linear layers only
def get_layers_E() -> nn.ModuleList:
    return nn.ModuleList([
        nn.Linear(INPUT_FEATURES, 64), # input -> hidden 1
        nn.Linear(64, 32),             # hidden 1 -> hidden 2
        nn.Linear(32, 1),              # hidden 2 -> output
    ])