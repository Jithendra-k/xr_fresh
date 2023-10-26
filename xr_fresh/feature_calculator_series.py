import jax.numpy as jnp
from jax import jit
import numpy as np
import geowombat as gw
from datetime import datetime
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline

from datetime import datetime
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Union, Any

# ratio_beyond_r_sigma


# Define a function to apply strftime('%j') to each element
def _get_day_of_year(dt):
    return int(dt.strftime("%j"))


def _check_valid_array(obj):
    # Check if the object is a NumPy or JAX array or list
    if not isinstance(obj, (np.ndarray, jnp.DeviceArray, list)):
        raise TypeError("Object must be a NumPy, JAX array or list.")

    # convert lists to numpy array
    if isinstance(obj, list):
        obj = np.array(obj)  # must be np array not jnp

    # Check if the array contains only integers or datetime objects
    if jnp.issubdtype(obj.dtype, np.integer):
        return jnp.array(obj)

    # datetime objects are converted to integers
    elif jnp.issubdtype(obj.dtype, datetime):
        return jnp.array(np.vectorize(_get_day_of_year)(obj))
    else:
        raise TypeError("Array must contain only integers, datetime objects.")


class interpolate_nan_dates(gw.TimeModule):
    """
    Interpolate missing values in a geospatial time series. This class can handle
    irregular time intervals between observations.

    Args:
        dates (list[datetime]): List of datetime objects corresponding to each time slice.
        missing_value (int or float, optional): The value to be replaced by NaNs. Default is None.
        interp_type (str, optional): The type of interpolation to use. Default is "linear".
        count (int, optional): Overrides the default output band count. Default is 1.

    Methods:
        calculate(array): Applies the interpolation on the input array.
    """

    def __init__(self, dates, missing_value=None, interp_type="linear", count=1):
        super(interpolate_nan_dates, self).__init__()

        # Validate dates is a list of datetime objects
        if not isinstance(dates, list) or not all(
            isinstance(d, datetime) for d in dates
        ):
            raise TypeError("dates must be a list of datetime objects")

        self.dates = dates
        self.date_indices = np.array(
            [(date - self.dates[0]).days for date in self.dates]
        )
        self.missing_value = missing_value
        self.interp_type = interp_type
        self.count = count

    def _interpolate_nans_with_dates(self, array):
        raise TypeError("interp1d not supported - use splines or linear - ")

        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]
            inter_fun = interp1d(
                x=valid_dates,
                y=valid_values,
                kind=self.interp_type,
                fill_value="extrapolate",
            )
            return inter_fun(self.date_indices)

    @staticmethod
    def _interpolate_nans_linear_with_dates(array, self):
        if all(np.isnan(array)):
            return array
        else:
            return np.interp(
                self.date_indices,
                self.date_indices[np.isnan(array) == False],
                array[np.isnan(array) == False],
            )

    @staticmethod
    def _interpolate_nans_CubicSpline_with_dates(array, self):
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]

            inter_fun = CubicSpline(x=valid_dates, y=valid_values, bc_type="not-a-knot")
            return inter_fun(self.date_indices)

    def calculate(self, array):
        # Replace missing_value with NaN
        if self.missing_value is not None and not np.isnan(self.missing_value):
            array = np.where(array == self.missing_value, np.NaN, array)

        if self.interp_type == "linear":
            # Interpolate using date indices
            array = np.apply_along_axis(
                self._interpolate_nans_linear_with_dates,
                axis=0,
                arr=array,
                self=self,
            )
        elif self.interp_type in [
            "cubicspline",
            "spline",
        ]:
            array = np.apply_along_axis(
                self._interpolate_nans_CubicSpline_with_dates,
                axis=0,
                arr=array,
                self=self,
            )
        return array.squeeze()


class interpolate_nan(gw.TimeModule):
    """
    Interpolate missing values in a geospatial time series. This class assumes a regular time
    interval between observations.

    Args:
        missing_value (int or float, optional): The value to be replaced by NaNs. Default is None.
        interp_type (str, optional): The type of interpolation algorithm to use. Options include "linear",
                                      "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next",
                                      "cubicspline", "spline", and "UnivariateSpline". Default is "linear".
        count (int, optional): Overrides the default output band count. Default is 1.

    Methods:
        calculate(array): Applies the interpolation on the input array.

    Example Usage:

        pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
        files = sorted(glob(f"{pth}*.tif"))[0:10]
        strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
        dates = sorted(datetime.strptime(string, strp_glob) for string in files)
        date_strings = [date.strftime("%Y-%m-%d") for date in dates]

        # window size controls RAM usage
        with gw.series(files, window_size=[640,640]) as src:
            src.apply(
                func=interpolate_nan(
                    missing_value=0,
                    count=len(src.filenames),
                ),
                outfile="/home/mmann1123/Downloads/test.tif",
                num_workers=5,
                bands=1,
            )
    """

    def __init__(self, missing_value=None, interp_type="linear", count=1, dates=None):
        super(interpolate_nan, self).__init__()
        # Validate dates is a list of datetime objects
        if dates is None:
            print("NOTE: Dates are unknown, assuming regular time interval")
            self.dates = dates
        elif not isinstance(dates, list) or not all(
            isinstance(d, datetime) for d in dates
        ):
            raise TypeError("dates must be a list of datetime objects")
        else:
            print("NOTE: Dates will be used to index the time series for interpolation")
            self.dates = dates
            self.date_indices = np.array(
                [(date - self.dates[0]).days for date in self.dates]
            )

        self.missing_value = missing_value
        self.interp_type = interp_type
        self.count = count

    @staticmethod
    def _interpolate_nans_interp1d(array, kind=None):
        # TO DO: seems to overwrite the first band with bad values
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_values = array[valid_indices]
            inter_fun = interp1d(
                x=valid_indices,
                y=valid_values,
                kind=kind,
                bounds_error=False,
                fill_value="extrapolate",
            )

            return inter_fun(np.arange(len(array)))

    def _interpolate_nans_interp1d_with_dates(array, self):
        # TO DO: seems to overwrite the first band with bad values

        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]
            inter_fun = interp1d(
                x=valid_dates,
                y=valid_values,
                kind=self.interp_type,
                fill_value="extrapolate",
            )
            return inter_fun(self.date_indices)

    @staticmethod
    def _interpolate_nans_linear(array):
        if all(np.isnan(array)):
            return array
        else:
            return np.interp(
                np.arange(len(array)),
                np.arange(len(array))[jnp.isnan(array) == False],
                array[np.isnan(array) == False],
            )

    @staticmethod
    def _interpolate_nans_linear_with_dates(array, self):
        if all(np.isnan(array)):
            return array
        else:
            return np.interp(
                self.date_indices,
                self.date_indices[np.isnan(array) == False],
                array[np.isnan(array) == False],
            )

    @staticmethod
    def _interpolate_nans_CubicSpline(array):
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_values = array[valid_indices]
            inter_fun = CubicSpline(
                x=valid_indices, y=valid_values, bc_type="not-a-knot"
            )
            return inter_fun(np.arange(len(array)))

    @staticmethod
    def _interpolate_nans_CubicSpline_with_dates(array, self):
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]

            inter_fun = CubicSpline(x=valid_dates, y=valid_values, bc_type="not-a-knot")
            return inter_fun(self.date_indices)

    @staticmethod
    def _interpolate_nans_CubicSpline_with_dates(array, self):
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]

            inter_fun = CubicSpline(x=valid_dates, y=valid_values, bc_type="not-a-knot")
            return inter_fun(self.date_indices)

    @staticmethod
    def _interpolate_nans_UnivariateSpline(array, s=1):
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_values = array[valid_indices]
            inter_fun = UnivariateSpline(x=valid_indices, y=valid_values, s=s)
            return inter_fun(np.arange(len(array)))

    @staticmethod
    def _interpolate_nans_UnivariateSpline_with_dates(array, self, s=1):
        if all(np.isnan(array)):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_date_indices = self.date_indices[valid_indices]
            valid_values = array[valid_indices]
            inter_fun = UnivariateSpline(x=valid_date_indices, y=valid_values, s=s)
            return inter_fun(self.date_indices)

    def calculate(self, array):
        # check if missing_value is not None and not np.nan
        if self.missing_value is not None:
            if not np.isnan(self.missing_value):
                array = np.where(array == self.missing_value, np.NaN, array)
            if self.interp_type == "linear":
                # check for dates as index
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_linear, axis=0, arr=array
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_linear_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                    )

            elif self.interp_type in [
                "nearest",
                "nearest-up",
                "zero",
                "slinear",
                "quadratic",
                "cubic",
                "previous",
                "next",
            ]:
                raise TypeError("interp1d not supported - use splines or linear - ")
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_interp1d,
                        axis=0,
                        arr=array,
                        kind=self.interp_type,
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_interp1d_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                        kind=self.interp_type,
                    )
            elif self.interp_type in [
                "cubicspline",
                "spline",
            ]:
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_CubicSpline,
                        axis=0,
                        arr=array,
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_CubicSpline_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                    )
            elif self.interp_type in [
                "UnivariateSpline",
            ]:
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_UnivariateSpline, axis=0, arr=array, s=1
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_UnivariateSpline_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                        s=1,
                    )
        # Return the interpolated array (3d -> time/bands x height x width)
        # If the array is (time x 1 x height x width) then squeeze to 3d
        return array.squeeze()


class abs_energy2(gw.TimeModule):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    Args:
        gw (_type_): _description_


    Example:

    with gw.series(
        files,
        nodata=9999,
    ) as src:
        print(src)
        src.apply(
            func=abs_energy(),
            outfile=f"/home/mmann1123/Downloads/test.tif",
            num_workers=5,
            bands=1,
        )

    """

    def __init__(self, missing_value=None, interp_type="linear"):
        super(abs_energy2, self).__init__()
        self.missing_value = missing_value
        self.interp_type = interp_type

    def calculate(self, array):
        # check if missing_value is not None and not np.nan
        if self.missing_value is not None:
            if not np.isnan(self.missing_value):
                array = jnp.where(array == self.missing_value, np.NaN, array)
            if self.interp_type == "linear":
                array = np.apply_along_axis(
                    _interpolate_nans_linear, axis=0, arr=array
                ).squeeze()
        print(array.shape)
        # Calculate the absolute energy
        array = jnp.sum(jnp.square(array), axis=0).squeeze()
        print("squeeze", array.shape)

        return array


class abs_energy(gw.TimeModule):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    Args:
        gw (_type_): _description_


    Example:

    with gw.series(
        files,
        nodata=9999,
    ) as src:
        print(src)
        src.apply(
            func=abs_energy(),
            outfile=f"/home/mmann1123/Downloads/test.tif",
            num_workers=5,
            bands=1,
        )

    """

    def __init__(self):
        super(abs_energy, self).__init__()

    def calculate(self, array):
        return jnp.nansum(jnp.square(array), axis=0).squeeze()


class absolute_sum_of_changes(gw.TimeModule):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(absolute_sum_of_changes, self).__init__()

    def calculate(self, array):
        return jnp.nansum(np.abs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()


class autocorrelation(gw.TimeModule):
    """Returns the autocorrelation of the time series data at a specified lag

    Args:
        gw (_type_): _description_
        lag (int): lag at which to calculate the autocorrelation (default: {1})
    """

    def __init__(self, lag=1):
        super(autocorrelation, self).__init__()
        self.lag = lag

    def calculate(self, array):
        # Extract the series and its lagged version
        series = array[: -self.lag]
        lagged_series = array[self.lag :]
        autocor = (
            jnp.sum(series * lagged_series, axis=0) / jnp.sum(series**2, axis=0)
        ).squeeze()

        return autocor


class count_above_mean(gw.TimeModule):
    """Returns the number of values in X that are higher than the mean of X

    Args:
        gw (_type_): _description_
        mean (int): An integer to use as the "mean" value of the raster
    """

    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            # Calculate the mean along the time dimension (axis=0) and broadcast it to match the shape of 'array'
            return jnp.nansum(array > jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array > self.mean, axis=0).squeeze()


class count_below_mean(gw.TimeModule):
    """Returns the number of values in X that are lower than the mean of X

    Args:
        gw (_type_): _description_
        mean (int): An integer to use as the "mean" value of the raster
    """

    def __init__(self, mean=None):
        super(count_below_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            # Calculate the mean along the time dimension (axis=0) and broadcast it to match the shape of 'array'
            return jnp.nansum(array < jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array > self.mean, axis=0).squeeze()


class doy_of_maximum(gw.TimeModule):
    """Returns the day of the year (doy) location of the maximum value of the series - treats all years as the same.

    pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
    files = sorted(glob(f"{pth}*.tif"))[0:10]
    strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
    dates = sorted(datetime.strptime(string, strp_glob) for string in files)
    dates

    with gw.series(
    files,
    nodata=9999,
        ) as src:
            print(src)
            src.apply(
                func=doy_of_maximum(dates),
                outfile=f"/home/mmann1123/Downloads/test.tif",
                num_workers=1,
                bands=1,
            )

    Args:
        gw (_type_): _description_
        dates (np.array): An array holding the dates of the time series as integers or as datetime objects.
    """

    def __init__(self, dates=None):
        super(doy_of_maximum, self).__init__()
        # check that dates is an array holding datetime objects or integers throw error if not
        dates = _check_valid_array(dates)
        self.dates = dates
        print("Day of the year found as:", self.dates)

    def calculate(self, array):
        # Find the indices of the maximum values along the time axis
        max_indices = jnp.argmax(array, axis=0)

        # Use the indices to extract the corresponding dates from the 'dates' array
        return self.dates[max_indices].squeeze()


class doy_of_minimum(gw.TimeModule):
    """Returns the day of the year (doy) location of the minimum value of the series - treats all years as the same.

    pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
    files = sorted(glob(f"{pth}*.tif"))[0:10]
    strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
    dates = sorted(datetime.strptime(string, strp_glob) for string in files)
    dates

    with gw.series(
    files,
    nodata=9999,
        ) as src:
            print(src)
            src.apply(
                func=doy_of_max(dates),
                outfile=f"/home/mmann1123/Downloads/test.tif",
                num_workers=1,
                bands=1,
            )

    Args:
        gw (_type_): _description_
        dates (np.array): An array holding the dates of the time series as integers or as datetime objects.
    """

    def __init__(self, dates=None):
        super(doy_of_minimum, self).__init__()
        # check that dates is an array holding datetime objects or integers throw error if not
        dates = _check_valid_array(dates)
        self.dates = dates
        print("Day of the year found as:", self.dates)

    def calculate(self, array):
        # Find the indices of the maximum values along the time axis
        min_indices = jnp.argmin(array, axis=0)

        # Use the indices to extract the corresponding dates from the 'dates' array
        return self.dates[min_indices].squeeze()


class kurtosis_excess(gw.TimeModule):
    """
    # https://medium.com/@pritul.dave/everything-about-moments-skewness-and-kurtosis-using-python-numpy-df305a193e46
    Returns the excess kurtosis of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).
    Args:
        gw (_type_): _description_

    """

    def __init__(self):
        super(kurtosis_excess, self).__init__()

    def calculate(self, array):
        mean_ = jnp.mean(array, axis=0)

        mu4 = jnp.mean((array - mean_) ** 4, axis=0)
        mu2 = jnp.mean((array - mean_) ** 2, axis=0)
        beta2 = mu4 / (mu2**2)
        gamma2 = beta2 - 3
        return gamma2.squeeze()


class large_standard_deviation(gw.TimeModule):
    """Boolean variable denoting if the standard dev of x is higher than 'r' times the range.

    Args:
        gw (_type_): _description_
        r (float, optional): The percentage of the range to compare with. Default is 2.0.
    """

    def __init__(self, r=2):
        super(large_standard_deviation, self).__init__()
        self.r = r

    def calculate(self, array):
        std_dev = jnp.nanstd(array, axis=0)
        max_val = jnp.nanmax(array, axis=0)
        min_val = jnp.nanmin(array, axis=0)

        return (std_dev > self.r * (max_val - min_val)).astype(jnp.int8).squeeze()


def _count_longest_consecutive(values):
    max_count = 0
    current_count = 0

    for value in values:
        if value:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


class longest_strike_above_mean(gw.TimeModule):
    """Returns the length of the longest consecutive subsequence in X that is larger than the mean of X

    Args:
        gw (_type_): _description_
    Returns:
        bool:
    """

    def __init__(self, mean=None):
        super(longest_strike_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        # compare to mean
        if self.mean is None:
            below_mean = array > jnp.mean(array, axis=0)
        else:
            below_mean = array > self.mean

        # Count the longest consecutive True values along the time dimension
        consecutive_true = jnp.apply_along_axis(
            _count_longest_consecutive, axis=0, arr=below_mean
        ).squeeze()

        # Count the longest consecutive False values along the time dimension
        consecutive_false = jnp.apply_along_axis(
            _count_longest_consecutive, axis=0, arr=~below_mean
        ).squeeze()

        return jnp.maximum(consecutive_true, consecutive_false)


class longest_strike_below_mean(gw.TimeModule):
    """Returns the length of the longest consecutive subsequence in X that is smaller than the mean of X

    Args:
        gw (_type_): _description_
    Returns:
        bool:
    """

    def __init__(self, mean=None):
        super(longest_strike_below_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        # compare to mean
        if self.mean is None:
            below_mean = array < jnp.mean(array, axis=0)
        else:
            below_mean = array < self.mean

        # Count the longest consecutive True values along the time dimension
        consecutive_true = jnp.apply_along_axis(
            _count_longest_consecutive, axis=0, arr=below_mean
        ).squeeze()

        # Count the longest consecutive False values along the time dimension
        consecutive_false = jnp.apply_along_axis(
            _count_longest_consecutive, axis=0, arr=~below_mean
        ).squeeze()

        return jnp.maximum(consecutive_true, consecutive_false)


class maximum(gw.TimeModule):
    """Calculate the highest value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the maximum.
                             Default is "time".
    """

    def __init__(self):
        super(maximum, self).__init__()

    def calculate(self, x):
        return jnp.nanmax(x, axis=0).squeeze()


class minimum(gw.TimeModule):
    """Calculate the lowest value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the minimum.
                             Default is "time".
    """

    def __init__(self):
        super(minimum, self).__init__()

    def calculate(self, x):
        return jnp.nanmin(x, axis=0).squeeze()


class mean(gw.TimeModule):
    """Calculate the mean value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(mean, self).__init__()

    def calculate(self, x):
        return jnp.nanmean(x, axis=0).squeeze()


class mean_abs_change(gw.TimeModule):
    """Calculate the mean over the absolute differences between subsequent time series values.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(mean_abs_change, self).__init__()

    def calculate(self, x):
        abs_diff = jnp.abs(jnp.diff(x, axis=0))
        return jnp.nanmean(abs_diff, axis=0).squeeze()


class mean_change(gw.TimeModule):
    """Calculate the mean over the differences between subsequent time series values.

    Args:
        gw (_type_): _description_

    """

    def __init__(self):
        super(mean_change, self).__init__()

    def calculate(self, array):
        diff = array[1:] - array[:-1]
        return jnp.nanmean(diff, axis=0).squeeze()


class mean_second_derivative_central(gw.TimeModule):
    """
    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    def __init__(self):
        super(mean_second_derivative_central, self).__init__()

    def calculate(self, array):
        series2 = array[:-2]
        lagged2 = array[2:]
        lagged1 = array[1:-1]
        msdc = jnp.sum(0.5 * (lagged2 - 2 * lagged1 + series2), axis=0) / (
            (2 * (len(array) - 2))
        )

        return msdc.squeeze()


class median(gw.TimeModule):
    """Calculate the mean value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(median, self).__init__()

    def calculate(self, x):
        return jnp.nanmedian(x, axis=0).squeeze()


class quantile(gw.TimeModule):
    """Compute the q-th quantile of the data along the time axis
        Args:
            q (int): Probability or sequence of probabilities for the quantiles to compute. Values must be between 0 and 1 inclusive.

    with gw.series(
        files,
        nodata=9999,
    ) as src:
        print(src)
        src.apply(
            func=quantile(0.90),
            outfile=f"/home/mmann1123/Downloads/test.tif",
            num_workers=1,
            bands=1,
        )

    """

    def __init__(self, q=None, method="linear"):
        super(quantile, self).__init__()
        self.q = q
        self.method = method

    def calculate(self, array):
        return jnp.quantile(array, q=self.q, method=self.method, axis=0).squeeze()


class ratio_beyond_r_sigma(gw.TimeModule):
    """Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    Args:
        gw (_type_): _description_
        r (int, optional):   Defaults to 2.
    """

    def __init__(self, r=2):
        super(ratio_beyond_r_sigma, self).__init__()
        self.r = r

    def calculate(self, array):
        return (
            jnp.nansum(
                jnp.abs(array - jnp.nanmean(array, axis=0))
                > self.r * jnp.nanstd(array, axis=0),
                axis=0,
            )
            / len(array)
        ).squeeze()


class skewness(gw.TimeModule):
    """
    # https://medium.com/@pritul.dave/everything-about-moments-skewness-and-kurtosis-using-python-numpy-df305a193e46
    Returns the sample skewness of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1). Normal value = 0, skewness > 0 means more weight in the left tail of
    the distribution.
    Args:
        gw (_type_): _description_
        axis (int, optional): Axis along which to compute the kurtosis. Default is 0.
        fisher (bool, optional): If True, Fisher's definition is used (normal=0).
                                 If False, Pearson's definition is used (normal=3).
                                 Default is False.
    """

    def __init__(self):
        super(skewness, self).__init__()

    def calculate(self, array):
        _mean = jnp.nanmean(array, axis=0)
        _diff = array - _mean
        _mu3 = jnp.nanmean(_diff**3, axis=0)
        _mu2 = jnp.nanmean(_diff**2, axis=0)
        beta = _mu3**2 / _mu2**3
        return jnp.sqrt(beta).squeeze()


class standard_deviation(gw.TimeModule):
    """Calculate the standard_deviation value of the time series.

    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(standard_deviation, self).__init__()

    def calculate(self, x):
        return jnp.nanstd(x, axis=0).squeeze()


class sum(gw.TimeModule):
    """Calculate the standard_deviation value of the time series.

    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(sum, self).__init__()

    def calculate(self, x):
        return jnp.sum(x, axis=0).squeeze()


class symmetry_looking(gw.TimeModule):
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param r: the percentage of the range to compare with (default: 0.1)
    :type r: float
    :return: the value of this feature
    :return type: bool
    """

    def __init__(self, r=0.1):
        super(symmetry_looking, self).__init__()
        self.r = r

    def calculate(self, array):
        return (
            jnp.abs(jnp.mean(array, axis=0) - jnp.median(array, axis=0))
            < (self.r * (jnp.max(array, axis=0) - jnp.min(array, axis=0)))
        ).squeeze()


class ts_complexity_cid_ce(gw.TimeModule):
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.


    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """

    def __init__(self, normalize=True):
        super(ts_complexity_cid_ce, self).__init__()
        self.normalize = normalize

    def calculate(self, array):
        if self.normalize:
            s = jnp.std(array, axis=0)

            array = jnp.where(s != 0, (array - jnp.mean(array, axis=0)) / s, array)
            array = jnp.where(s == 0, 0.0, array)

        x = jnp.diff(array, axis=0)

        # Compute dot product along the time dimension
        try:
            dot_prod = jnp.einsum("tijk, tijk->jk", x, x)
        except:
            dot_prod = jnp.einsum("ijk, ijk->jk", x, x)
        return jnp.sqrt(dot_prod)


class unique_value_number_to_time_series_length(gw.TimeModule):
    """SLOW
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # of unique values / # of values
    """

    def __init__(self):
        super(unique_value_number_to_time_series_length, self).__init__()
        print("this is slow and needs more work")

    def calculate(self, array):
        # Count the number of unique values along the time axis (axis=0)
        unique_counts = jnp.sum(jnp.unique(array, axis=0), axis=0)

        return (unique_counts / len(array)).squeeze()

        # def count_unique_values(arr):
        #     unique_counts = jnp.sum(np.unique(arr, axis=0), axis=0)
        #     return unique_counts

        # # Apply the function along the time axis (axis=0)
        # result = jnp.apply_along_axis(count_unique_values, axis=0, arr=array)

        # # Divide the count of unique values by the length of time
        # result /= array.shape[0]

        # return result.squeeze()


class variance(gw.TimeModule):
    """Calculate the variance of the time series

    Args:
        gw (_type_): _description_
    Returns:
        bool:
    """

    def __init__(self):
        super(variance, self).__init__()

    def calculate(self, x):
        return jnp.var(x, axis=0).squeeze()


class variance_larger_than_standard_deviation(gw.TimeModule):
    """Calculate the variance of the time series is larger than the standard_deviation.

    Args:
        gw (_type_): _description_
    Returns:
        bool:
    """

    def __init__(self):
        super(variance_larger_than_standard_deviation, self).__init__()

    def calculate(self, x):
        return (jnp.var(x, axis=0) > jnp.nanstd(x, axis=0)).astype(np.int8).squeeze()


# visualize interpolation
def open_files(predict: str, actual: str):
    with gw.open(predict) as predict:
        with gw.open(actual, stack_dim="band") as actual:
            return predict, actual


def sample_data(predict, actual, n=20):
    df1 = gw.sample(predict, n=20).dropna().reset_index(drop=True)
    df2 = gw.extract(actual, df1[["point", "geometry"]])
    return df1, df2


def plot_data(df1, df2):
    fig, ax = plt.subplots(figsize=(10, 6))
    time_points = list(range(1, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df1)))

    for idx, row in df2.iterrows():
        ax.scatter(
            time_points,
            row[time_points],
            color=colors[idx],
            label=f"actual, Point {row['point']}",
            linestyle="-",
        )

    for idx, row in df1.iterrows():
        ax.plot(
            time_points,
            row[time_points],
            color=colors[idx],
            label=f"predicted, Point {row['point']}",
            linestyle="--",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series Comparison Between Predicted and Actual Values")
    plt.show()


def plot_interpolated_actual(
    interpolated_stack: str, original_image_list: list, samples: int = 20
):
    """Plots the interpolated and actual values for a given time series.

    Args:
        interpolated_stack (str): multiband stack of images representing interpolated time series. Defaults to None.
        original_image_list (list): list of files used in interpolation. Defaults to None.
        samples (int, optional): number of random points to compare time series. Defaults to 20.
    """
    predict, actual = open_files(
        interpolated_stack,
        original_image_list,
    )
    df1, df2 = sample_data(predict, actual, n=samples)
    plot_data(df1, df2)


# skipped
# def pearson_r(a, b, dim="time", skipna=False, **kwargs):
# linear_time_trend
# longest_run
