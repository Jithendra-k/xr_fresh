
# jordans method has about a 10% improvement in smaller dataset 


import xarray as xr
import geowombat as gw
import os, sys
sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')
from xr_fresh.feature_calculators import * 
from xr_fresh.backends import Cluster
from xr_fresh.extractors import extract_features
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from xr_fresh.utils import * 
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
from geowombat.core.parallel import ParallelTask
 
 #%%

# PPT time series growing season May to Feb

files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'

band_name = 'ppt'
file_glob = f"{files}/pdsi*tif"
strp_glob = f"{files}/pdsi_%Y%m.tif"

dates = sorted(datetime.strptime(string, strp_glob)
        for string in sorted(glob(file_glob)))
 

# open xarray 
with gw.open(sorted(glob(file_glob)), 
              band_names=[band_name],
              time_names = dates  ) as ds:
                 
    ds = ds.chunk((len(ds.time), 1, 250, 250))
    ds.attrs['nodatavals'] =  (-9999,)

    
    # move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
    ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
     
    
    # start cluster
    cluster = Cluster()
    cluster.start_large_object()
     
    
    #extract growing season year month day 
    features = extract_features(xr_data= ds.sel(time=slice('2013-03-01', '2014-12-31')),
                                feature_dict=complete_f,
                                band=band_name, 
                                filepath = '/home/mmann1123/Desktop',
                                postfix = '_2015',
                                na_rm = True,
                                persist=True)
    
    cluster.close()

 
#%% Jordan's method


# import xarray as xr
# import geowombat as gw
# import os, sys
# sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')
# from xr_fresh.feature_calculators import * 
# from xr_fresh.backends import Cluster
# from xr_fresh.extractors import extract_features
# from glob import glob
# from datetime import datetime
# import matplotlib.pyplot as plt
# from xr_fresh.utils import * 
# import logging
# import warnings
# import xarray as xr
# from numpy import where
# from xr_fresh import feature_calculators
# from itertools import chain
# from geowombat.backends import concat as gw_concat
# _logger = logging.getLogger(__name__)
# from numpy import where
# from xr_fresh.utils import xarray_to_rasterio
# import pandas as pd
# from pathlib import Path
# from geowombat.core.parallel import ParallelTask
# import itertools 

# complete_f =  { #'abs_energy':[{}],
#                 #'mean':[{}],
#                 #   'ts_complexity_cid_ce':[{}], #5.62 lambda #5.31 not lambda #5.91 one line return with lambda
#                 'sum_values':[{}] #11.37 lambda ufunc
#                 #'autocorr':[{}], # 4.07s
#                 #'linear_time_trend': [{'param':"slope"}], 
#               }
 

# # PPT time series growing season May to Feb

# files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'

# band_name = 'ppt'
# file_glob = f"{files}/pdsi*tif"
# strp_glob = f"{files}/pdsi_%Y%m.tif"

# dates = sorted(datetime.strptime(string, strp_glob)
#         for string in sorted(glob(file_glob)))
 

# # open xarray 
# with gw.open(sorted(glob(file_glob)), 
#               band_names=[band_name],
#               time_names = dates  ) as ds:
                 
#     ds = ds.chunk((len(ds.time), 1, 250, 250))
#     ds.attrs['nodatavals'] =  (-9999,)

    
#     # move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
#     ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
     
    
#     def user_func(*args):
#         data, num_workers = list(itertools.chain(*args))
#         results = data.data.sum().compute(scheduler='threads', num_workers=num_workers)
#         return results
   
#     def extract_features(ds):
#         pt = ParallelTask(ds, n_workers=3)
#         res = pt.map(user_func, 4)
        
#     with gw.open(sorted(glob(file_glob)), 
#                   band_names=[band_name],
#                   time_names = dates  ) as ds:
                     
#         ds = ds.chunk((len(ds.time), 1, 250, 250))
#         ds.attrs['nodatavals'] =  (-9999,)
#         ds = ds.sel(time=slice('2013-03-01', '2017-12-31'))
        
#         # move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
#         ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
         
#         extract_features(ds)
    
#%%

#https://stackoverflow.com/questions/52108417/how-to-apply-linear-regression-to-every-pixel-in-a-large-multi-dimensional-array
#https://stackoverflow.com/questions/58719696/how-to-apply-a-xarray-u-function-over-netcdf-and-return-a-2d-array-multiple-new/62012973


def _timereg(x, t, param  ):
    
    linReg = linregress(x=t, y=x)
    
    # return getattr(linReg, param) 
 
    # return (getattr(linReg, "slope") ,getattr(linReg, "intercept"))
    return np.stack((getattr(linReg, "intercept"),getattr(linReg, "slope"),getattr(linReg, "pvalue"),getattr(linReg, "rvalue") ), axis=-1)
    
def linear_time_trend(x, param="slope", dim='time', **kwargs):
    
    # look at https://stackoverflow.com/questions/58719696/how-to-apply-a-xarray-u-function-over-netcdf-and-return-a-2d-array-multiple-new/62012973

    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param param: contains text of the attribute name of the regression model
    :type param: list
    :return: the value of this feature
    :return type: int
    """
    
    t = xr.DataArray(np.arange(len(x[dim]))+1, dims=dim,
             coords={dim: x[dim]})
    
    return xr.apply_ufunc( _timereg, x , t,
                            input_core_dims=[[dim], [dim]],
                            kwargs={ 'param':param},
                            vectorize=True,  
                            #dask='parallelized',
                            #output_dtypes=[float],
                            output_core_dims= [["pred"]]
                            # output_core_dims= [['prediction'],['prediction1']] #ValueError: dimensions {'predictions'} do not exist. Expected one or more of ('band', 'y', 'x')
                            )
 


files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'

band_name = 'ppt'
file_glob = f"{files}/pdsi*tif"
strp_glob = f"{files}/pdsi_%Y%m.tif"

dates = sorted(datetime.strptime(string, strp_glob)
        for string in sorted(glob(file_glob)))
 

# open xarray 
with gw.open(sorted(glob(file_glob)), 
              band_names=[band_name],
              time_names = dates  ) as ds:
                 
    ds = ds.chunk((len(ds.time), 1, 250, 250))
    ds.attrs['nodatavals'] =  (-9999,)
    ds = ds.load()
    
    # move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
    ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
     
    
    # start cluster
    cluster = Cluster()
    cluster.start_large_object()
     
    a = linear_time_trend(ds)#.compute()
    out = a[0].isel(pred = 1)
    out.gw.imshow()

#%%

b = xr.concat([ds,a.isel(pred = 1)],dim='variable')


#%%  WORKING
 
def _timereg(x, t, param  ):
    
    linReg = linregress(x=t, y=x)
    
    # return getattr(linReg, param) 
 
    # return (getattr(linReg, "slope") ,getattr(linReg, "intercept"))
    return np.stack((getattr(linReg, "slope") ,getattr(linReg, "intercept")), axis=-1)
    
def linear_time_trend(x, param="slope", dim='time', **kwargs):
    
    # look at https://stackoverflow.com/questions/58719696/how-to-apply-a-xarray-u-function-over-netcdf-and-return-a-2d-array-multiple-new/62012973

    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param param: contains text of the attribute name of the regression model
    :type param: list
    :return: the value of this feature
    :return type: int
    """
    
    t = xr.DataArray(np.arange(len(x[dim]))+1, dims=dim,
             coords={dim: x[dim]})
    
    return xr.apply_ufunc( _timereg, x , t,
                            input_core_dims=[[dim], [dim]],
                            kwargs={ 'param':param},
                            vectorize=True,  
                            #dask='parallelized',
                            #output_dtypes=[float],
                            output_core_dims= [["predictions"]]
                            )
 


files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'

band_name = 'ppt'
file_glob = f"{files}/pdsi*tif"
strp_glob = f"{files}/pdsi_%Y%m.tif"

dates = sorted(datetime.strptime(string, strp_glob)
        for string in sorted(glob(file_glob)))
 

# open xarray 
with gw.open(sorted(glob(file_glob)), 
              band_names=[band_name],
              time_names = dates  ) as ds:
                 
    ds = ds.chunk((len(ds.time), 1, 250, 250))
    ds.attrs['nodatavals'] =  (-9999,)
    ds = ds.load()
    
    # move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
    ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
     
    
    # start cluster
    cluster = Cluster()
    cluster.start_large_object()
     
    a = linear_time_trend(ds).compute()
    out = a.isel(predictions = 1)
    out.gw.imshow()
