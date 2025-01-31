import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def generate_modular_nd_grid(bounds, base_resolution, subgrid_index, N):
    """
    Generates an n-dimensional grid using a modular offset technique to divide a higher-resolution grid into N sub-grids.

    Parameters:
        bounds (list of tuples): [(min1, max1), ..., (minN, maxN)] defining the range for each dimension.
        base_resolution (int): The full resolution when all sub-grids are combined.
        subgrid_index (int): Which sub-grid to generate (1 to N).
        N (int): Total number of sub-grids.

    Returns:
        list of tuples: The points forming the modular sub-grid.
    """
    num_dims = len(bounds)
    
    if not (1 <= subgrid_index <= N):
        raise ValueError("subgrid_index must be between 1 and N.")
    if base_resolution % N != 0:
        raise ValueError("base_resolution must be divisible by N.")

    # Compute the reduced resolution per sub-grid
    sub_resolution = base_resolution // N

    # Generate the full high-res grid for each dimension
    full_axes = [np.linspace(b[0], b[1], base_resolution, endpoint=True) for b in bounds]

    # Select every N-th point for each sub-grid using modular selection
    sub_axes = [axis[subgrid_index - 1 :: N] for axis in full_axes]

    # Generate the modular sub-grid (Cartesian product of selected points)
    grid_points = np.array(np.meshgrid(*sub_axes, indexing="ij")).T.reshape(-1, num_dims)

    return [tuple(point) for point in grid_points]

def generate_modular_nd_grid_random_order(bounds, base_resolution, N, seed=None):
    """
    Generates an n-dimensional modular sub-grid with optional randomized partitioning.

    Parameters:
        bounds (list of tuples): [(min1, max1), ..., (minN, maxN)] defining the range for each dimension.
        base_resolution (int): The full resolution when all sub-grids are combined.
        N (int): Number of sub-grids.
        randomize (bool): If True, randomizes the subgrid assignments within each segment.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list of lists: Each list represents one sub-grid, containing tuples of n-dimensional points.
    """
    num_dims = len(bounds)

    if base_resolution % N != 0:
        raise ValueError("base_resolution must be divisible by N.")

    # Compute full high-res grid for each dimension
    full_axes = [np.linspace(b[0], b[1], base_resolution, endpoint=True) for b in bounds]

    # Generate full Cartesian grid
    full_grid = np.array(np.meshgrid(*full_axes, indexing="ij")).T.reshape(-1, num_dims)

    # Determine sub-grid assignments
    if seed is not None:
        random.seed(seed)  # Set seed for reproducibility
    assignments = [random.sample(range(N), N) for _ in range(base_resolution**num_dims//N)]

    # Split points into sub-grids
    sub_grids = [[] for _ in range(N)]
    for i, point in enumerate(full_grid):
        sub_index = assignments[np.floor(i // N)][i%N]
        sub_grids[sub_index].append(tuple(point))

    return sub_grids

def visualize_subgrids(bounds, base_resolution, N,randomize=True):
    """
    Visualizes modular sub-grids in 2D or 3D.

    Parameters:
        bounds (list of tuples): Defines the space.
        base_resolution (int): Total resolution before splitting.
        N (int): Number of sub-grids to visualize.
    """
    num_dims = len(bounds)
    colors = plt.cm.jet(np.linspace(0, 1, N))  # Generate N colors

    fig = plt.figure(figsize=(8, 8))
    if randomize:
        if num_dims == 2:
            ax = fig.add_subplot(111)
            sub_grids = generate_modular_nd_grid_random_order(bounds, base_resolution, N, seed=42)
            for i, subgrid in enumerate(sub_grids):
                x, y = np.array(subgrid).T
                ax.scatter(x, y, color=colors[i - 1], label=f"Sub-grid {i}", s=10)
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.legend()
        
        elif num_dims == 3:
            ax = fig.add_subplot(111, projection="3d")
            sub_grids = generate_modular_nd_grid_random_order(bounds, base_resolution, N, seed=42)
            i = 0
            subgrid = sub_grids[0]
            x, y, z = zip(*subgrid)
            ax.scatter(x, y, z, color=colors[i - 1], label=f"Sub-grid {i}", s=10)
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.legend()

    else:
        if num_dims == 2:
            ax = fig.add_subplot(111)
            for i in range(1, N + 1):
                subgrid = generate_modular_nd_grid(bounds, base_resolution, i, N)
                x, y = zip(*subgrid)
                ax.scatter(x, y, color=colors[i - 1], label=f"Sub-grid {i}", s=10)
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.legend()
        
        elif num_dims == 3:
            ax = fig.add_subplot(111, projection="3d")
            for i in range(1,2):# N + 1):
                subgrid = generate_modular_nd_grid(bounds, base_resolution, i, N)
                x, y, z = zip(*subgrid)
                ax.scatter(x, y, z, color=colors[i - 1], label=f"Sub-grid {i}", s=10)
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.legend()

        else:
            print("Visualization only supports 2D and 3D.")
            return

    plt.title(f"{num_dims}D Modular Grid Visualization")
    plt.show()

if __name__ == "__main__":
    # Example usage:
    bounds_2d = [(0, 1), (0, 1)]  # 2D space
    bounds_3d = [(0, 1), (0, 1), (0, 1)]  # 3D space
    base_resolution = 16
    N = 4  # Number of sub-grids

    visualize_subgrids(bounds_2d, base_resolution, N)  # 2D Visualization
    visualize_subgrids(bounds_3d, base_resolution, N)  # 3D Visualization
