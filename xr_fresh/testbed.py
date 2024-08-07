# %%
import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
from datetime import datetime

# import xr_fresh as xr
from xr_fresh.extractors import extract_features

os.chdir("~/xr_fresh/xr_fresh/data")

files = sorted(glob("evi*.tif"))
files

feature_list = {
    "minimum": [{}],
    "abs_energy": [{}],
    "mean_abs_change": [{}],
    "variance_larger_than_standard_deviation": [{}],
    "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}, {"r": 3}],
    "symmetry_looking": [{}],
    "sum_values": [{}],
}
outpath = "/home/mmann1123/Downloads/"
with gw.open(files, band_names=["evi"]) as ds:
    display(ds)

    features = extract_features(
        xr_data=ds,
        band="evi",
        feature_dict=feature_list,
        na_rm=True,
        persist=True,
        filepath=outpath,
        postfix="_2016_2017_",
    )

# %% env:testbed laptop

import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
from datetime import datetime
from geowombat.data import l8_224078_20200518


sys.path.append("/home/mmann1123/Documents/github/xr_fresh/xr_fresh")
sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
import xr_fresh as xf

from xr_fresh.feature_calculator_series import (
    doy_of_maximum,
    abs_energy,
    autocorrelation,
    doy_of_maximum,
    ols_slope_intercept,
    kurtosis,
)
from xr_fresh.interpolate_series import interpolate_nan
from xr_fresh.visualizer import plot_interpolated_actual

pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
pth = "/home/mmann1123/extra_space/Dropbox/Africa_data/Temperature/"

files = sorted(glob(f"{pth}*.tif"))[0:5]
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
date_strings = [date.strftime("%Y-%m-%d") for date in dates]
date_strings


# %% Simiple without arguments

with gw.series(
    files,
    nodata=9999,
) as src:
    src.apply(
        func=ols_slope_intercept(returns="all"),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )

# %%
with gw.series(
    files,
    nodata=9999,
) as src:
    src.apply(
        func=autocorrelation(4),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )

# %% with date argument3
with gw.series(
    files,
    nodata=9999,
) as src:
    src.apply(
        func=doy_of_maximum(dates),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=1,
        bands=1,
    )


# %%

# %% for missing values

pth = "/home/mmann1123/Documents/github/xr_fresh/tests/data/"

files = sorted(glob(f"{pth}*.tif"))
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
date_strings = [date.strftime("%Y-%m-%d") for date in dates]
date_strings


# %%
with gw.series(files, window_size=[32 * 30, 32 * 30]) as src:
    src.apply(
        func=interpolate_nan(
            missing_value=np.nan,
            interp_type="spline",
            # output band count
            count=len(src.filenames),
        ),
        outfile=f"/home/mmann1123/Downloads/test_spline.tif",
        num_workers=src.nchunks,
        # number of bands to read
        bands=1,
    )
# %%
with gw.series(files, window_size=[32 * 30, 32 * 30]) as src:
    src.apply(
        func=interpolate_nan(
            missing_value=np.nan,
            dates=dates,
            interp_type="linear",
            # output band count
            count=len(src.filenames),
        ),
        outfile=f"/home/mmann1123/Downloads/test_linear_dates.tif",
        num_workers=src.nchunks,
        # number of bands to read
        bands=1,
    )

# %%
with gw.series(files) as src:
    src.apply(
        func=interpolate_nan(
            missing_value=np.nan,
            dates=dates,
            interp_type="UnivariateSpline",
            # output band count
            count=len(src.filenames),
        ),
        outfile=f"/home/mmann1123/Downloads/test_UnivariateSpline_dates.tif",
        num_workers=15,
        # number of bands to read
        bands=1,
    )

# %%
with gw.series(files) as src:
    src.apply(
        func=interpolate_nan_dates(
            missing_value=np.nan,
            dates=dates,
            interp_type="linear",
            # output band count
            count=len(src.filenames),
        ),
        outfile=f"/home/mmann1123/Downloads/test_dates.tif",
        num_workers=15,
        # number of bands to read
        bands=1,
    )

# %%

# plot interpolated vs actual values


plot_interpolated_actual(
    interpolated_stack="/home/mmann1123/Downloads/test_linear_dates.tif",
    original_image_list=files,
    samples=20,
)


# %%
