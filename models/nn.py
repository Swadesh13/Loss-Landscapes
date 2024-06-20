from torch import nn


def simple_nn(in_dim, hidden_dim, num_classes, num_hidden_layers=1):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, hidden_dim),
        *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)],
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def nn_bn(in_dim, hidden_dim, num_classes, num_hidden_layers=1):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        *[
            nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)])
            for _ in range(num_hidden_layers)
        ],
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def conv_nn(num_classes, in_channels=3):
    model = nn.Sequential(
        nn.Conv2d(in_channels, 64, (3, 3), padding=(1, 1)),
        nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
        nn.AdaptiveAvgPool2d((1, 1)),
        simple_nn(64, 64, num_classes, 1),
    )
    return model
