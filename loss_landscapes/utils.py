import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


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

    loss /= batch_idx + 1
    return loss


def _get_weights(model):
    return [p.data for p in model.parameters()]


def _get_diff_weights(weights1, weights2):
    return [w2 - w1 for (w1, w2) in zip(weights1, weights2)]


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


def tensorlist_to_tensor(weights):
    """Concatnate a list of tensors into one tensor.

    Args:
        weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

    Returns:
        concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def npvec_to_tensorlist(direction, params):
    """Convert a numpy vector to a list of tensors with the same shape as "params".

    Args:
        direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
        base: a list of parameter tensors from net

    Returns:
        a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(direction[idx : idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert idx == len(direction)
        return w2
    else:
        s2 = []
        idx = 0
        for k, w in params.items():
            s2.append(torch.Tensor(direction[idx : idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert idx == len(direction)
        return s2


def nplist_to_tensor(nplist):
    """Concatenate a list of numpy vectors into one tensor.

    Args:
        nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

    Returns:
        concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d * np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def setup_PCA_directions(
    model, model_files, w, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Find PCA directions for the optimization path from the initial model
    to the final trained model.

    Returns:
        dir_name: the h5 file that stores the directions.
    """
    # load models and prepare the optimization path matrix
    matrix = []
    net2 = copy.deepcopy(model)
    for model_file in model_files:
        net2.load_state_dict(torch.load(model_file, map_location=device))
        w2 = _get_weights(net2)
        d = _get_diff_weights(w, w2)
        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())

    # Perform PCA on the optimization path matrix
    print("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])

    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    # convert vectorized directions to the same shape as models to save in h5 file.
    xdirection = npvec_to_tensorlist(pc1, w)
    ydirection = npvec_to_tensorlist(pc2, w)

    f = {}
    f["xdirection"] = xdirection
    f["ydirection"] = ydirection
    f["explained_variance_ratio_"] = pca.explained_variance_ratio_
    f["singular_values_"] = pca.singular_values_
    f["explained_variance_"] = pca.explained_variance_

    return f


def project_1D(w, d):
    """Project vector w to vector d and get the length of the projection.

    Args:
        w: vectorized weights
        d: vectorized direction

    Returns:
        the projection scalar
    """
    assert len(w) == len(d), "dimension does not match for w and "
    scale = torch.dot(w, d) / d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    """Project vector d to the plane spanned by dx and dy.

    Args:
        d: vectorized weights
        dx: vectorized direction
        dy: vectorized direction
        proj_method: projection method
    Returns:
        x, y: the projection coordinates
    """

    if proj_method == "cos":
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == "lstsq":
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y


def project_trajectory(
    dir1,
    dir2,
    model,
    w,
    model_files,
    proj_method="cos",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Project the optimization trajectory onto the given two directions.

    Args:
      dir_file: the h5 file that contains the directions
      w: weights of the final model
      s: states of the final model
      model_name: the name of the model
      model_files: the checkpoint files
      dir_type: the type of the direction, weights or states
      proj_method: cosine projection

    Returns:
      proj_file: the projection filename
    """

    dx = nplist_to_tensor(dir1)
    dy = nplist_to_tensor(dir2)

    xcoord, ycoord = [], []
    net2 = copy.deepcopy(model)
    for model_file in model_files:
        net2.load_state_dict(torch.load(model_file, map_location=device))
        w2 = _get_weights(net2)
        d = _get_diff_weights(w, w2)
        d = tensorlist_to_tensor(d)
        x, y = project_2D(d, dx, dy, proj_method)
        print("%s  (%.4f, %.4f)" % (model_file, x, y))

        xcoord.append(x)
        ycoord.append(y)

    f = {}
    f["proj_xcoord"] = np.array(xcoord, dtype=np.float32)
    f["proj_ycoord"] = np.array(ycoord, dtype=np.float32)

    return f
