import torch
import torch.nn.functional as F


def load_data_once(train_loader_unshuffled, num_batches):
    train_loader = []
    for i, data in enumerate(train_loader_unshuffled, 1):
        train_loader.append(data)
        if i == num_batches:
            break
    return train_loader


def compute_loss(model, device, train_loader_unshuffled, criterion=None, num_batches: int = 8):
    """
    Compute and return the loss over the first num_batches batches given by the train_loader_unshuffled, using the criterion provided.

    Parameters
    ----------
    model : the torch model which will be evaluated.
    train_loader_unshuffled : torch dataloader unshuffled.
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    num_batches : number of batches to evaluate the model with. (default to 8)

    Returns
    ----------
    loss : loss computed
    """

    if criterion is None:
        criterion = F.cross_entropy

    loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_loader_unshuffled):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss += criterion(logits, labels).item()
            if batch_idx + 1 >= num_batches:
                break

    loss /= len(train_loader_unshuffled)
    return loss


def _get_weights(model):
    return [p.data for p in model.parameters()]


def _get_random_weights(weights, device):
    return [torch.randn(w.size()).to(device) for w in weights]


def filter_normalization_weights(direction, weights):
    assert len(direction) == len(weights)
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0)
        d.mul_(w.norm() / (d.norm() + 1e-10))


def create_random_direction(model, device):
    """
    Return a random direction in the model's weights space.
    This vector is normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    model : the torch model whose weights will be used to create and normalize the direction

    Returns
    ----------
    direction : a tensor, which correspond to the sampled direction.
    """

    weights = _get_weights(model)
    direction = _get_random_weights(weights, device)
    filter_normalization_weights(direction, weights)
    return direction


def create_random_directions(model, device):
    """
    Return two random directions in the model's weights space.
    These vectors are normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    model : the torch model whose weights will be used to create and normalize the directions.

    Returns
    ----------
    directions : list of two tensors, which correspond to the two sampled directions.
    """

    x_direction = create_random_direction(model, device)
    y_direction = create_random_direction(model, device)
    return [x_direction, y_direction]
