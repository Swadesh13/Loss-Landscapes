import os
import math
import numpy as np
import itertools
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import torch
import h5py
from .utils import (
    compute_loss,
    create_random_direction,
    create_random_directions,
    load_data_once,
    setup_PCA_directions,
    project_trajectory,
    _get_weights,
)


def create_2D_losscape(
    model=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    train_loader_unshuffled=None,
    direction=None,
    criterion=None,
    num_batches: int = 8,
    load_once: bool = True,
    x_min: float = -1.0,
    x_max: float = 1.0,
    num_points: int = 50,
    show: bool = True,
    save: bool = True,
    output_path: str = "./results",
):
    """
    Create a 2D losscape of the given model.

    Parameters
    ----------
    model : the torch model which will be used to create the losscape.
    h5 : ! Support not added
    train_loader_unshuffled : the torch dataloader. It is supposed to be fixed so that all the calls to this function will use the same data.
    optimizer : the optimizer used for training (should follow the same API as torch optimizers).(default to Adam)
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    num_batches : number of batches to evaluate the model with. (default to 8)
    load_once : Load data only once to reduce file i/o (default: True)
    x_min : min x value (that multiply the sampled direction). (default to -1.)
    x_max : max x value (that multiply the sampled direction). (default to 1.)
    output_path : path where the plot will be saved. (default to '2d_losscape.png')
    num_points : number of points to evaluate the loss, from x_min to x_max. (default to 50)

    Returns
    ----------
    coords : numpy array containing the x coords used to create the landscape
    losses : list of the losses computed
    """

    model.to(device)

    if direction is None:
        direction = [create_random_direction(model, device)]

    init_weights = [p.data for p in model.parameters()]

    coords = np.linspace(x_min, x_max, num_points)
    losses = []

    if load_once:
        data = load_data_once(train_loader_unshuffled, num_batches)
        if len(data):
            train_loader_unshuffled = data

    for x in tqdm(coords):
        _set_weights(model, init_weights, direction, x)
        loss = compute_loss(model, device, train_loader_unshuffled, criterion, num_batches)
        losses.append(loss)

    _reset_weights(model, init_weights)

    plt.plot(coords, losses)
    if save:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "2d_losscape.png"), dpi=300)
    if show:
        plt.show()
    plt.clf()
    return coords, losses


def create_3D_losscape(
    model=None,
    h5=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    train_loader_unshuffled=None,
    directions=None,
    criterion=None,
    num_batches: int = 8,
    load_once: bool = True,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    num_points: int = 50,
    show: bool = True,
    save: bool = True,
    output_path: str = "./results",
    output_vtp: bool = True,
    output_h5: bool = True,
    pca=None,
):
    """
    Create a 3D losscape of the given model.

    Parameters
    ----------
    model : the torch model which will be used to create the losscape.
    train_loader_unshuffled : the torch dataloader. It is supposed to be fixed so that all the calls to this function will use the same data.
    optimizer : the optimizer used for training (should follow the same API as torch optimizers).(default to Adam)
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    num_batches : number of batches to evaluate the model with. (default to 8)
    load_once : Load data only once to reduce file i/o (default: True)
    x_min : min x value (that multiply the first sampled direction). (default to -1.)
    x_max : max x value (that multiply the first sampled direction). (default to 1.)
    y_min : min x value (that multiply the second sampled direction). (default to -1.)
    y_max : max x value (that multiply the second sampled direction). (default to 1.)
    num_points : number of points to evaluate the loss, from x_min to x_max and y_min to y_max. (default to 50)
    output_path : path where the plot will be saved. (default to '3d_losscape.png')
    output_vpt : whether or not to also create a .vtp file, used to 3D visualize the losscape. (default to False)
    output_h5 : whether or not to also create a .h5 file, containing the data generated by this function (default to True)
    pca : List of files to do pca

    Returns
    ----------
    X : a (num_points, num_points) numpy array, the X meshgrid
    Y : a (num_points, num_points) numpy array, the Y meshgrid
    losses : a (num_points, num_points) numpy array containing all the losses computed
    """
    if save:
        os.makedirs(output_path, exist_ok=True)

    if pca:
        model = model.to(torch.device("cpu"))
        init_weights = _get_weights(model)
        res = setup_PCA_directions(model, pca, init_weights, device=torch.device("cpu"))
        directions = [res["xdirection"], res["ydirection"]]
        proj_coords = project_trajectory(
            *directions, model, init_weights, pca, "cos", device=torch.device("cpu")
        )
        _max = max(max(proj_coords["proj_xcoord"]), max(proj_coords["proj_ycoord"]))
        _min = min(min(proj_coords["proj_xcoord"]), min(proj_coords["proj_ycoord"]))
        x_min = -20  # _min - _max
        x_max = 20  # _max + _max
        y_min = -20  # _min - _max
        y_max = 20  # _max + _max

    model = model.to(device)

    if h5:
        with h5py.File(h5, "r") as f:
            X = f["X"][:]
            Y = f["Y"][:]
            losses = f["losses"][:]
    else:
        init_weights = _get_weights(model)
        if directions is None:
            directions = create_random_directions(model, device)

        directions = [[d.to(device) for d in direction] for direction in directions]

        X, Y = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))
        losses = np.empty_like(X)

        count = 0
        total = X.shape[0] * X.shape[1]

        if load_once:
            data = load_data_once(train_loader_unshuffled, num_batches)
            if len(data):
                train_loader_unshuffled = data

        for i, j in tqdm(itertools.product(range(X.shape[0]), range(X.shape[1])), total=total):
            _set_weights(model, init_weights, directions, np.array([X[i, j], Y[i, j]]))
            loss = compute_loss(model, device, train_loader_unshuffled, criterion, num_batches)
            losses[i, j] = loss
            count += 1

        _reset_weights(model, init_weights)

    fig = plt.figure()
    if pca:
        cp = plt.plot(proj_coords["proj_xcoord"], proj_coords["proj_ycoord"], marker=".", color="red")
    cp = plt.contourf(X, Y, losses, cmap="viridis_r")
    # plt.clabel(cp, colors="black", inline=1, fontsize=8)
    # plt.xticks([])
    # plt.yticks([])
    if pca:
        plt.xlabel(f"1st PCA component: {(res['explained_variance_ratio_'][0] * 100):.2f}%")
        plt.ylabel(f"2nd PCA component: {(res['explained_variance_ratio_'][1] * 100):.2f}%")
    plt.colorbar(cp)
    if save:
        plt.close()
        fig.savefig(os.path.join(output_path, "3d_losscape.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

    fig = plt.figure()
    if pca:
        cp = plt.plot(proj_coords["proj_xcoord"], proj_coords["proj_ycoord"], marker=".", color="red")
    cp = plt.contourf(X, Y, np.log(losses), cmap="viridis_r")
    # plt.clabel(cp, colors="black", inline=1, fontsize=8)
    # plt.xticks([])
    # plt.yticks([])
    if pca:
        plt.xlabel(f"1st PCA component: {(res['explained_variance_ratio_'][0] * 100):.2f}%")
        plt.ylabel(f"2nd PCA component: {(res['explained_variance_ratio_'][1] * 100):.2f}%")
    else:
        plt.xlabel("1st Direction")
        plt.ylabel("2nd Direction")
    plt.colorbar(cp)
    if save:
        plt.close()
        fig.savefig(os.path.join(output_path, "3d_log_losscape.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

    if output_vtp:
        os.makedirs(output_path, exist_ok=True)
        _create_vtp(X, Y, losses, log=False, output_path=output_path)
        _create_vtp(X, Y, losses, log=True, output_path=output_path)

    if output_h5:
        os.makedirs(output_path, exist_ok=True)
        with h5py.File(os.path.join(output_path, "data.h5"), "w") as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("Y", data=Y)
            hf.create_dataset("losses", data=losses)
            if pca:
                hf.create_dataset("proj_xcoord", data=proj_coords["proj_xcoord"])
                hf.create_dataset("proj_ycoord", data=proj_coords["proj_ycoord"])
                hf.create_dataset(
                    "explained_variance_ratio_",
                    data=np.array(res["explained_variance_ratio_"], dtype=np.float32),
                )

    return X, Y, losses


def _set_weights(model, weights, directions, step):
    if len(directions) == 2:
        dx = directions[0]
        dy = directions[1]
        changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
    else:
        changes = [d * step for d in directions[0]]

    for p, w, d in zip(model.parameters(), weights, changes):
        p.data = w + d


def _reset_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w.type(type(p.data)))


# as in https://github.com/tomgoldstein/loss-landscape
def _create_vtp(X, Y, losses, log=False, zmax=-1, interp=-1, output_path=""):
    # set this to True to generate points
    show_points = False
    # set this to True to generate polygons
    show_polys = True

    xcoordinates = X
    ycoordinates = Y
    vals = losses

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    # Interpolate the resolution up to the desired amount
    if interp > 0:
        m = interpolate.interp2d(xcoordinates[0, :], ycoordinates[:, 0], vals, kind="cubic")
        x_array = np.linspace(min(x_array), max(x_array), interp)
        y_array = np.linspace(min(y_array), max(y_array), interp)
        z_array = m(x_array, y_array).ravel()

        x_array, y_array = np.meshgrid(x_array, y_array)
        x_array = x_array.ravel()
        y_array = y_array.ravel()

    vtp_file = os.path.join(output_path, "losscape")
    if zmax > 0:
        z_array[z_array > zmax] = zmax
        vtp_file += "_zmax=" + str(zmax)

    if log:
        z_array = np.log(z_array) * 2
        vtp_file += "_log"
    vtp_file += ".vtp"
    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (
                z_array[temp_index]
                + z_array[temp_index + 1]
                + z_array[temp_index + matrix_size]
                + z_array[temp_index + matrix_size + 1]
            ) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_file, "w")
    output_file.write(
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
    )
    output_file.write("  <PolyData>\n")

    if show_points and show_polys:
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(
                number_points, number_points, number_polys
            )
        )
    elif show_polys:
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(
                number_points, number_polys
            )
        )
    else:
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(
                number_points, number_points
            )
        )

    # <PointData>
    output_file.write("      <PointData>\n")
    output_file.write(
        '        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
            min_value_array[2], max_value_array[2]
        )
    )
    for vertexcount in range(number_points):
        if (vertexcount % 6) == 0:
            output_file.write("          ")
        output_file.write("{}".format(z_array[vertexcount]))
        if (vertexcount % 6) == 5:
            output_file.write("\n")
        else:
            output_file.write(" ")
    if (vertexcount % 6) != 5:
        output_file.write("\n")
    output_file.write("        </DataArray>\n")
    output_file.write("      </PointData>\n")

    # <CellData>
    output_file.write("      <CellData>\n")
    if show_polys and not show_points:
        output_file.write(
            '        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
                avg_min_value, avg_max_value
            )
        )
        for vertexcount in range(number_polys):
            if (vertexcount % 6) == 0:
                output_file.write("          ")
            output_file.write("{}".format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) == 5:
                output_file.write("\n")
            else:
                output_file.write(" ")
        if (vertexcount % 6) != 5:
            output_file.write("\n")
        output_file.write("        </DataArray>\n")
    output_file.write("      </CellData>\n")

    # <Points>
    output_file.write("      <Points>\n")
    output_file.write(
        '        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
            min_value, max_value
        )
    )
    for vertexcount in range(number_points):
        if (vertexcount % 2) == 0:
            output_file.write("          ")
        output_file.write("{} {} {}".format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) == 1:
            output_file.write("\n")
        else:
            output_file.write(" ")
    if (vertexcount % 2) != 1:
        output_file.write("\n")
    output_file.write("        </DataArray>\n")
    output_file.write("      </Points>\n")

    # <Verts>
    output_file.write("      <Verts>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_points - 1
        )
    )
    if show_points:
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write("          ")
            output_file.write("{}".format(vertexcount))
            if (vertexcount % 6) == 5:
                output_file.write("\n")
            else:
                output_file.write(" ")
        if (vertexcount % 6) != 5:
            output_file.write("\n")
    output_file.write("        </DataArray>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_points
        )
    )
    if show_points:
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write("          ")
            output_file.write("{}".format(vertexcount + 1))
            if (vertexcount % 6) == 5:
                output_file.write("\n")
            else:
                output_file.write(" ")
        if (vertexcount % 6) != 5:
            output_file.write("\n")
    output_file.write("        </DataArray>\n")
    output_file.write("      </Verts>\n")

    # <Lines>
    output_file.write("      <Lines>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1
        )
    )
    output_file.write("        </DataArray>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys
        )
    )
    output_file.write("        </DataArray>\n")
    output_file.write("      </Lines>\n")

    # <Strips>
    output_file.write("      <Strips>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1
        )
    )
    output_file.write("        </DataArray>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys
        )
    )
    output_file.write("        </DataArray>\n")
    output_file.write("      </Strips>\n")

    # <Polys>
    output_file.write("      <Polys>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1
        )
    )
    if show_polys:
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) == 0:
                    output_file.write("          ")
                output_file.write(
                    "{} {} {} {}".format(
                        temp_index,
                        (temp_index + 1),
                        (temp_index + matrix_size + 1),
                        (temp_index + matrix_size),
                    )
                )
                if (polycount % 2) == 1:
                    output_file.write("\n")
                else:
                    output_file.write(" ")
                polycount += 1
        if (polycount % 2) == 1:
            output_file.write("\n")
    output_file.write("        </DataArray>\n")
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys
        )
    )
    if show_polys:
        for polycount in range(number_polys):
            if (polycount % 6) == 0:
                output_file.write("          ")
            output_file.write("{}".format((polycount + 1) * 4))
            if (polycount % 6) == 5:
                output_file.write("\n")
            else:
                output_file.write(" ")
        if (polycount % 6) != 5:
            output_file.write("\n")
    output_file.write("        </DataArray>\n")
    output_file.write("      </Polys>\n")

    output_file.write("    </Piece>\n")
    output_file.write("  </PolyData>\n")
    output_file.write("</VTKFile>\n")
    output_file.write("")
    output_file.close()

    print("Done with file:{}".format(vtp_file))
