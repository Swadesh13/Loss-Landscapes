from torch import nn

def simple_nn(in_dim, hidden_dim, num_classes, num_hidden_layers=1):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, hidden_dim),
        *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)],
        nn.Linear(hidden_dim, num_classes)
    )
    return model