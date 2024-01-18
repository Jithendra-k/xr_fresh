# xr_fresh

xr_fresh is designed to quickly generate a broad set of temporal features from gridded raster data time series.

It is a python package for analyzing time-series characteristics from raster data in `xarray` format as generated by `geowombat`.
It allows feature extraction, dimension reduction and applications of machine learning techniques for geospatial data.

- input data : historical raster data (e.g. Monthly temperature data (2000-2018)
- extracted features: Mean, minimum, maximum, variance, standard deviation... of the input data
- output: rasters with features as columns or raster file with features as bands

Example outputs:

![png](examples/output_8_0.png)

# Install

*Linux & Windows Install*

```
# add dependency
conda create -n xr_fresh geowombat -c conda-forge
conda activate xr_fresh
# clone repository
cd # to desired location
git clone https://github.com/fraymio/xr_fresh.git
cd xr_fresh 
pip install . 
```  

# Documentation

After cloning the repository navigate locally to and open the following:

```

/xr_fresh/docs/build/html/index.html

```

# Example

```

import xarray as xr
import geowombat as gw
import os, sys

sys.path.append("/path/to/xr_fresh/")  # if import above doesn't work point to xr_fresh directory
from xr_fresh.feature_calculators import *
from xr_fresh.backends import Cluster
from xr_fresh.extractors import extract_features
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from xr_fresh.utils import*
import logging
import warnings
import xarray as xr
from numpy import where
from xr_fresh import feature_calculators
from itertools import chain
from geowombat.backends import concat as gw_concat

_logger = logging.getLogger(__name__)
from numpy import where
from xr_fresh.utils import xarray_to_rasterio
import pandas as pd
from pathlib import Path

# %%

files = "/home/Land_Cover/data/EVI"
band_name = "evi"
file_glob = f"{files}/*.tif"
# exmple naming convension S2_SR_EVI_M_2022_01.tif
strp_glob = f"{files}/S2_SR_EVI_M_%Y_%m.tif"


f_list = sorted(glob(file_glob))

dates = sorted(datetime.strptime(string, strp_glob) for string in f_list)
dates

complete_f = {
    "minimum": [{}],
    "abs_energy": [{}],
    "mean_abs_change": [{}],
    "variance_larger_than_standard_deviation": [{}],
    "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}, {"r": 3}],
    "symmetry_looking": [{}],
    "sum_values": [{}],
    "autocorr": [
        {"lag": 1},
        {"lag": 2},
    ],  
    "ts_complexity_cid_ce": [{}],
    "mean_second_derivative_central": [{}],
    "median": [{}],
    "mean": [{}],
    "standard_deviation": [{}],
    "variance": [{}],
    "skewness": [{}],
    "kurtosis": [{}],
    "absolute_sum_of_changes": [{}],
    "longest_strike_below_mean": [{}],
    "longest_strike_above_mean": [{}],
    "count_above_mean": [{}],
    "count_below_mean": [{}],
    "doy_of_maximum_first": [
        {"band": band_name}
    ],   
    "doy_of_maximum_last": [{"band": band_name}],
    "doy_of_minimum_last": [{"band": band_name}],
    "doy_of_minimum_first": [{"band": band_name}],
    "ratio_value_number_to_time_series_length": [{}],
    "quantile": [{"q": 0.05}, {"q": 0.95}],
    "maximum": [{}],
    "linear_time_trend": [{"param": "all"}],
}
 
 
 
# start cluster

cluster = Cluster()
cluster.start_large_object()

# open xarray lazy

with gw.open(
    sorted(glob(file_glob)), band_names=[band_name], time_names=dates, nodata=0
) as ds:
    ds = ds.chunk({"time": -1, "band": 1, "y": 350, "x": 350})  # rechunk to time
    ds = ds.gw.mask_nodata()
    print(ds)

    # generate features previous Sep - current March ( Masika growing season)

    for year in [2023, 2024]:
        previous_year = str(year - 1)
        year = str(year)
        print(year)
        ds_year = ds.sel(time=slice(previous_year + "-08-01", year + "-03-01"))
        print("interpolating missing values")
        ds_year = ds_year.interpolate_na(dim="time", limit=5)

        # make output folder
        outpath = os.path.join(files, "annual_features/Sep_Mar_S2")
        os.makedirs(outpath, exist_ok=True)

        # extract growing season year month day
        features = extract_features(
            xr_data=ds_year,
            feature_dict=complete_f,
            band=band_name,
            na_rm=True,
            persist=True,
            filepath=outpath,
            postfix="_sep_mar_" + year,
        )
        cluster.restart()

cluster.close()

```
