# %%import numpy as np
import numba
import ray
import geowombat as gw
from glob import glob
from geowombat.core.parallel import ParallelTask
from sklearn.decomposition import KernelPCA
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


with gw.open(
    sorted(
        [
            "./tests/data/RadT_tavg_202301.tif",
            "./tests/data/RadT_tavg_202302.tif",
            "./tests/data/RadT_tavg_202304.tif",
            "./tests/data/RadT_tavg_202305.tif",
        ]
    ),
    stack_dim="band",
    band_names=[0, 1, 2, 3],
) as src:
    print(src)
    print(src.values.shape)
    src.sel(band=0).gw.imshow()

    num_features, height, width = src.shape
    data = src.values

transposed_data = np.transpose(data, (1, 2, 0))
features = transposed_data.reshape(-1, num_features)

# # drop rows with nans
features = features[~np.isnan(features).any(axis=1)]
features

# Number of random coordinates to select
num_samples = 20000

# Generate random coordinates
np.random.seed(42)  # For reproducibility

# select num samples rows
sampled_features = features[
    np.random.choice(features.shape[0], num_samples, replace=False)
]

# Fit Kernel PCA on the sampled features
print("fitting kpca")
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=4, n_jobs=-1)
kpca.fit(sampled_features)

# Extract necessary attributes from kpca for transformation
X_fit_ = kpca.X_fit_
eigenvector = kpca.eigenvectors_[:, 3]  # Take the first component
eigenvalue = kpca.eigenvalues_[3]
gamma = kpca.gamma
# Extract necessary attributes from kpca for transformation
X_fit_ = kpca.X_fit_
eigenvector = kpca.eigenvectors_[:, 0]  # Take the first component
eigenvalue = kpca.eigenvalues_[0]
gamma = kpca.gamma

# %%


@numba.jit(nopython=True, parallel=True)
def transform_entire_dataset_numba(data, X_fit_, eigenvector, eigenvalue, gamma):
    height, width = data.shape[1], data.shape[2]
    transformed_data = np.zeros((height, width))

    for i in numba.prange(height):
        for j in range(width):
            feature_vector = data[:, i, j]
            k = np.exp(-gamma * np.sum((feature_vector - X_fit_) ** 2, axis=1))
            transformed_feature = np.dot(k, eigenvector / np.sqrt(eigenvalue))
            transformed_data[i, j] = transformed_feature

    return transformed_data


@ray.remote
def process_window(
    data_block_id,
    data_slice,
    window_id,
    X_fit_,
    eigenvector,
    eigenvalue,
    gamma,
    num_workers,
):
    data_chunk = data_block_id[
        data_slice
    ].data.compute()  # Convert Dask array to NumPy array
    return transform_entire_dataset_numba(
        data_chunk, X_fit_, eigenvector, eigenvalue, gamma
    )


# Open the raster file with GeoWombat
with gw.open(
    sorted(
        [
            "./tests/data/RadT_tavg_202301.tif",
            "./tests/data/RadT_tavg_202302.tif",
            "./tests/data/RadT_tavg_202304.tif",
            "./tests/data/RadT_tavg_202305.tif",
        ]
    ),
    stack_dim="band",
    band_names=[0, 1, 2, 3],
) as src:
    pt = ParallelTask(src, row_chunks=256, col_chunks=256, scheduler="ray", n_workers=8)

    # Map the process_window function to each chunk of the dataset
    futures = pt.map(process_window, X_fit_, eigenvector, eigenvalue, gamma, 8)

# Combine the results
transformed_data = np.zeros((height, width))
for window_id, future in enumerate(ray.get(futures)):
    window = pt.windows[window_id]
    row_start, col_start = window.row_off, window.col_off
    row_end, col_end = row_start + window.height, col_start + window.width
    transformed_data[row_start:row_end, col_start:col_end] = future


# %%

# Create a sample 2D NumPy array
data = np.random.rand(10, 10)  # Replace with your actual data

# Plot the array
plt.imshow(transformed_data, cmap="viridis", interpolation="none")
plt.colorbar()  # Add a colorbar to show the scale
plt.title("2D NumPy Array Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()

# %%
# # %%
# import numpy as np
# import numba
# import ray
# import geowombat as gw
# from glob import glob
# from geowombat.core.parallel import ParallelTask
# import numpy as np
# import numba
# from sklearn.decomposition import KernelPCA, PCA
# import matplotlib.pyplot as plt
# import functools

# # Example data shape (15, 4000, 3000)
# num_features, height, width = 15, 4000, 3000
# data = np.random.rand(num_features, height, width)  # Replace with your actual data

# # Number of random coordinates to select
# num_samples = 1000

# # Generate random coordinates
# np.random.seed(42)  # For reproducibility
# x_coords = np.random.randint(0, height, num_samples)
# y_coords = np.random.randint(0, width, num_samples)
# random_coords = list(zip(x_coords, y_coords))

# # Extract features for these coordinates
# sampled_features = np.array([data[:, x, y] for x, y in random_coords])

# # Fit Kernel PCA on the sampled features
# kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
# kpca.fit(sampled_features)

# # Extract necessary attributes from kpca for transformation
# X_fit_ = kpca.X_fit_
# eigenvector = kpca.eigenvectors_[:, 0]  # Take the first component
# eigenvalue = kpca.eigenvalues_[0]
# gamma = kpca.gamma


# @numba.jit(nopython=True, parallel=False)
# def transform_entire_dataset_numba(
#     data, X_fit_, eigenvector, eigenvalue, gamma, height, width
# ):
#     transformed_data = np.zeros((height, width))

#     for i in numba.prange(height):
#         for j in range(width):
#             feature_vector = data[:, i, j]
#             k = np.exp(-gamma * np.sum((feature_vector - X_fit_) ** 2, axis=1))
#             transformed_feature = np.dot(k, eigenvector / np.sqrt(eigenvalue))
#             transformed_data[i, j] = transformed_feature

#     return transformed_data


# @ray.remote
# def process_window(data, X_fit_, eigenvector, eigenvalue, gamma, height, width):
#     return transform_entire_dataset_numba(
#         data, X_fit_, eigenvector, eigenvalue, gamma, height, width
#     )


# # Open the raster file with GeoWombat
# with gw.open(
#     glob("./tests/data/R*.tif"), stack_dim="band", band_names=[0, 1, 2, 3]
# ) as src:
#     pt = ParallelTask(src, row_chunks=256, col_chunks=256, scheduler="ray", n_workers=8)
#     # Use a lambda function to pass additional parameters
#     results = ray.get(
#         pt.map(
#             lambda data: process_window.remote(
#                 data, X_fit_, eigenvector, eigenvalue, gamma, 256, 256
#             )
#         )
#     )
# # Process the results (e.g., print the mean values)
# for result in results:
#     print(result)


# # %%


# # import jax.numpy as jnp

# # from xr_fresh.feature_calculator_series import _get_jax_backend
# from glob import glob
# import geowombat as gw
# from xr_fresh.feature_calculator_series import *

# files = glob("tests/data/*.tif")

# out_path = "test.tif"
# # use rasterio to create a new file tif file

# with gw.series(files) as src:
#     src.apply(
#         longest_strike_above_mean(mean=299),
#         bands=1,
#         num_workers=12,
#         outfile=out_path,
#     )

# # %%
# import numpy as np

# with gw.open(files) as src:
#     for i in range(len(src)):
#         src[i] = i + 1
#         # replace alternating strips rows with nan in each band, so no band has overlapping nans values
#         block_length = len(src[0, 0]) // 5
#         src[i].loc[dict(y=slice(i * block_length, (1 + i) * block_length))] = np.nan
#         print(src[i].values)
#         gw.save(
#             src.sel(time=i + 1), f"./tests/data/values_equal_{i+1}.tif", overwrite=True
#         )


# # %%
# import numpy as np
# import geowombat as gw

# files = glob("tests/data/RadT*.tif")

# # Open the dataset
# with gw.open(files) as src:
#     for i in range(len(src)):
#         # Replace the values in each band of the ith time slice with i + 1
#         src[i] = i + 1

#         # Replace alternating strips of rows with NaN values in each band
#         block_length = len(src[0, 0]) // 5  # Divide the y dimension into 5 blocks
#         start_row = i * block_length  # Start row index for the current block
#         end_row = (i + 1) * block_length  # End row index for the current block

#         # Replace the rows in each band with NaN values for the current block
#         src[i, :, start_row:end_row, :] = np.nan

#         # Save the modified dataset
#         gw.save(
#             src.sel(time=i + 1), f"./tests/data/values_equal_{i+1}.tif", overwrite=True
#         )

# # %%

# # import jax.numpy as jnp
# from xr_fresh.interpolate_series import *

# # from xr_fresh.feature_calculator_series import _get_jax_backend
# from pathlib import Path
# from glob import glob
# from datetime import datetime
# import geowombat as gw
# import os
# import tempfile


# files = sorted(glob("tests/data/values_equal_*.tif"))
# files
# # %%

# out_path = "test.tif"
# with gw.series(files, transfer_lib="jax") as src:
#     src.apply(
#         func=interpolate_nan(
#             interp_type="linear",
#             missing_value=np.nan,
#             count=len(src.filenames),
#         ),
#         outfile=out_path,
#         bands=1,
#     )


# with gw.open(out_path) as dst:
#     # assert all of band 1 are equal to 1 NOTE EDGE CASE NOT HANDLED
#     # assert np.all(dst[0] == 1)
#     # assert all of band 2 are equal to 2
#     assert np.all(dst[1] == 2)
#     # assert all of band 4 are equal to 4
#     assert np.all(dst[2] == 3)
#     # assert all of band 4 are equal to 4
#     assert np.all(dst[3] == 4)
#     # assert all of band 5 are equal to 5
#     # assert np.all(dst[4] == 5) NOTE EDGE CASE NOT HANDLED


# # %%
# import geopandas as gpd

# eas = gpd.read_file("~/Downloads/ethiopia_eas.gpkg")
# # %%
# # fix geometry column
# from shapely.validation import make_valid

# eas["geometry"] = eas["geometry"].apply(make_valid)
# # add x and y coordinates of the centroid to each row
# eas["x"] = eas.centroid.x
# eas["y"] = eas.centroid.y


# eas.to_file("~/Downloads/ethiopia_eas.gpkg", driver="GPKG")

# # %%

# %%
