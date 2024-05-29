import jax.numpy as jnp
import numpy as np
import geowombat as gw
from datetime import datetime
from typing import List, Union, Any

__all__ = [
    "abs_energy",
    "absolute_sum_of_changes",
    "autocorrelation",
    "count_above_mean",
    "count_below_mean",
    "doy_of_maximum",
    "doy_of_minimum",
    "kurtosis",
    "kurtosis_excess",
    "large_standard_deviation",
    "longest_strike_above_mean",
    "longest_strike_below_mean",
    "maximum",
    "minimum",
    "mean",
    "mean_abs_change",
    "mean_change",
    "mean_second_derivative_central",
    "median",
    "ols_slope_intercept",
    "quantile",
    "ratio_beyond_r_sigma",
    "skewness",
    "standard_deviation",
    "sum",
    "symmetry_looking",
    "ts_complexity_cid_ce",
    "unique_value_number_to_time_series_length",
    "variance",
    "variance_larger_than_standard_deviation",
]


def _get_day_of_year(dt):
    return int(dt.strftime("%j"))


def _check_valid_array(obj):
    # Check if the object is a NumPy or JAX array or list
    if not isinstance(obj, (np.ndarray, list)):  # jnp.DeviceArray,
        raise TypeError("Object must be a NumPy array or list.")

    # Convert lists to NumPy array
    if isinstance(obj, list):
        obj = np.array(obj)  # Must be np array not jnp

    # Check if the array contains only integers or datetime objects
    if jnp.issubdtype(obj.dtype, np.integer):
        return jnp.array(obj)

    # datetime objects are converted to integers
    elif jnp.issubdtype(obj.dtype, datetime):
        return jnp.array(np.vectorize(_get_day_of_year)(obj))
    else:
        raise TypeError("Array must contain only integers, datetime objects.")


class abs_energy(gw.TimeModule):
    """
    Returns the absolute energy of the time series which is the sum over the squared values.

    .. math::

        E = \\sum_{i=1}^{n} x_i^2

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=abs_energy(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(abs_energy, self).__init__()

    def calculate(self, array):
        return jnp.nansum(jnp.square(array), axis=0).squeeze()


class absolute_sum_of_changes(gw.TimeModule):
    """
    Returns the sum over the absolute value of consecutive changes in the series x.

    .. math::

        \\sum_{i=1}^{n-1} |x_{i+1} - x_i|

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=absolute_sum_of_changes(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(absolute_sum_of_changes, self).__init__()

    def calculate(self, array):
        return jnp.nansum(jnp.abs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()


class autocorrelation(gw.TimeModule):
    """
    Returns the autocorrelation of the time series data at a specified lag.

    .. math::

        \\text{Autocorrelation} = \\frac{\\sum_{i=1}^{n-k} (x_i \\cdot x_{i+k})}{\\sum_{i=1}^{n} x_i^2}

    Args:
        lag (int): Lag at which to calculate the autocorrelation (default: 1).

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=autocorrelation(lag=1), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, lag=1):
        super(autocorrelation, self).__init__()
        self.lag = lag

    def calculate(self, array):
        series = array[: -self.lag]
        lagged_series = array[self.lag:]
        autocor = (jnp.nansum(series * lagged_series, axis=0) / jnp.nansum(series ** 2, axis=0)).squeeze()
        return autocor


class count_above_mean(gw.TimeModule):
    """
    Returns the number of values in X that are higher than the mean of X.

    Args:
        mean (int): An integer to use as the "mean" value of the raster. If None, the mean of X is used.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=count_above_mean(mean=50), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            return jnp.nansum(array > jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array > self.mean, axis=0).squeeze()


class count_below_mean(gw.TimeModule):
    """
    Returns the number of values in X that are lower than the mean of X.

    Args:
        mean (int): An integer to use as the "mean" value of the raster. If None, the mean of X is used.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=count_below_mean(mean=50), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, mean=None):
        super(count_below_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            return jnp.nansum(array < jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array < self.mean, axis=0).squeeze()


class doy_of_maximum(gw.TimeModule):
    """
    Returns the day of the year (doy) location of the maximum value of the series - treats all years as the same.

    Args:
        dates (np.array): An array holding the dates of the time series as integers or as datetime objects.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=doy_of_maximum(dates=date_array), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, dates=None):
        super(doy_of_maximum, self).__init__()
        dates = _check_valid_array(dates)
        self.dates = jnp.array(dates) if dates is not None else None

    def calculate(self, array):
        if self.dates is None:
            raise ValueError("Dates array is not provided.")
        max_indices = jnp.argmax(array, axis=0)
        return self.dates[max_indices].squeeze()


class doy_of_minimum(gw.TimeModule):
    """
    Returns the day of the year (doy) location of the minimum value of the series - treats all years as the same.

    Args:
        dates (np.array): An array holding the dates of the time series as integers or as datetime objects.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=doy_of_minimum(dates=date_array), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, dates=None):
        super(doy_of_minimum, self).__init__()
        dates = _check_valid_array(dates)
        self.dates = jnp.array(dates) if dates is not None else None

    def calculate(self, array):
        if self.dates is None:
            raise ValueError("Dates array is not provided.")
        min_indices = jnp.argmin(array, axis=0)
        return self.dates[min_indices].squeeze()


class kurtosis(gw.TimeModule):
    """
    Compute the sample kurtosis of a given array along the time axis.

    Args:
        fisher (bool, optional): If True, Fisher’s definition is used (normal ==> 0.0).
                                 If False, Pearson’s definition is used (normal ==> 3.0).

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=kurtosis(fisher=True), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, fisher=True):
        super(kurtosis, self).__init__()
        self.fisher = fisher

    def calculate(self, array):
        mean_ = jnp.nanmean(array, axis=0)
        mu4 = jnp.nanmean((array - mean_) ** 4, axis=0)
        mu2 = jnp.nanmean((array - mean_) ** 2, axis=0)
        beta2 = mu4 / (mu2 ** 2)
        if self.fisher:
            return (beta2 - 3).squeeze()
        return beta2.squeeze()


class kurtosis_excess(gw.TimeModule):
    """
    Compute the excess kurtosis of the sample, defined as kurtosis(X) - 3

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=kurtosis_excess(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(kurtosis_excess, self).__init__()

    def calculate(self, array):
        mean_ = jnp.nanmean(array, axis=0)
        mu4 = jnp.nanmean((array - mean_) ** 4, axis=0)
        mu2 = jnp.nanmean((array - mean_) ** 2, axis=0)
        beta2 = mu4 / (mu2 ** 2)
        return (beta2 - 3).squeeze()


class large_standard_deviation(gw.TimeModule):
    """
    Computes a large standard deviation, defined as:

    .. math::

        \\text{large\_std} = \\text{std}(x) > 1

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=large_standard_deviation(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(large_standard_deviation, self).__init__()

    def calculate(self, array):
        return (jnp.nanstd(array, axis=0) > 1).squeeze()


class longest_strike_above_mean(gw.TimeModule):
    """
    Returns the length of the longest consecutive subsequence that is bigger than the series mean.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=longest_strike_above_mean(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(longest_strike_above_mean, self).__init__()

    def calculate(self, array):
        mean_ = jnp.nanmean(array, axis=0)
        is_above_mean = array > mean_
        max_strike = jnp.zeros(array.shape[1:])
        for i in range(array.shape[0]):
            streak = jnp.zeros_like(max_strike)
            strike = jnp.zeros_like(max_strike)
            for j in range(array.shape[0]):
                streak = jnp.where(is_above_mean[j], streak + 1, 0)
                strike = jnp.maximum(strike, streak)
            max_strike = jnp.maximum(max_strike, strike)
        return max_strike.squeeze()


class longest_strike_below_mean(gw.TimeModule):
    """
    Returns the length of the longest consecutive subsequence that is smaller than the series mean.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=longest_strike_below_mean(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(longest_strike_below_mean, self).__init__()

    def calculate(self, array):
        mean_ = jnp.nanmean(array, axis=0)
        is_below_mean = array < mean_
        max_strike = jnp.zeros(array.shape[1:])
        for i in range(array.shape[0]):
            streak = jnp.zeros_like(max_strike)
            strike = jnp.zeros_like(max_strike)
            for j in range(array.shape[0]):
                streak = jnp.where(is_below_mean[j], streak + 1, 0)
                strike = jnp.maximum(strike, streak)
            max_strike = jnp.maximum(max_strike, strike)
        return max_strike.squeeze()


class maximum(gw.TimeModule):
    """
    Returns the maximum value of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=maximum(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(maximum, self).__init__()

    def calculate(self, array):
        return jnp.nanmax(array, axis=0).squeeze()


class minimum(gw.TimeModule):
    """
    Returns the minimum value of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=minimum(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(minimum, self).__init__()

    def calculate(self, array):
        return jnp.nanmin(array, axis=0).squeeze()


class mean(gw.TimeModule):
    """
    Returns the mean value of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=mean(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(mean, self).__init__()

    def calculate(self, array):
        return jnp.nanmean(array, axis=0).squeeze()


class mean_abs_change(gw.TimeModule):
    """
    Returns the mean over the absolute differences between subsequent time series values.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=mean_abs_change(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(mean_abs_change, self).__init__()

    def calculate(self, array):
        return jnp.nanmean(jnp.abs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()


class mean_change(gw.TimeModule):
    """
    Returns the mean value of consecutive changes in the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=mean_change(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(mean_change, self).__init__()

    def calculate(self, array):
        return jnp.nanmean(jnp.diff(array, n=1, axis=0), axis=0).squeeze()


class mean_second_derivative_central(gw.TimeModule):
    """
    Returns the mean value of a central approximation of the second derivative.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=mean_second_derivative_central(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(mean_second_derivative_central, self).__init__()

    def calculate(self, array):
        return jnp.nanmean((array[2:] - 2 * array[1:-1] + array[:-2]) / 2, axis=0).squeeze()


class median(gw.TimeModule):
    """
    Returns the median value of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=median(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(median, self).__init__()

    def calculate(self, array):
        return jnp.nanmedian(array, axis=0).squeeze()


class ols_slope_intercept(gw.TimeModule):
    """
    Returns the slope and intercept of the ordinary least-squares (OLS) linear regression.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=ols_slope_intercept(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(ols_slope_intercept, self).__init__()

    def calculate(self, array):
        x = jnp.arange(array.shape[0])
        y = array
        x_mean = jnp.nanmean(x)
        y_mean = jnp.nanmean(y, axis=0)
        slope = jnp.nansum((x - x_mean)[:, None] * (y - y_mean), axis=0) / jnp.nansum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        return jnp.stack([slope, intercept], axis=0).squeeze()


class quantile(gw.TimeModule):
    """
    Returns the q-th quantile of the series.

    Args:
        q (float): The quantile to compute, which must be between 0 and 1 inclusive.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=quantile(q=0.5), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, q=0.5):
        super(quantile, self).__init__()
        self.q = q

    def calculate(self, array):
        return jnp.nanquantile(array, self.q, axis=0).squeeze()


class ratio_beyond_r_sigma(gw.TimeModule):
    """
    Returns the ratio of values beyond r times sigma (standard deviation) from the mean.

    Args:
        r (float): The number of standard deviations from the mean to use as the threshold.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=ratio_beyond_r_sigma(r=2), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, r):
        super(ratio_beyond_r_sigma, self).__init__()
        self.r = r

    def calculate(self, array):
        mean_ = jnp.nanmean(array, axis=0)
        std_ = jnp.nanstd(array, axis=0)
        return (jnp.sum(jnp.abs(array - mean_) > self.r * std_, axis=0) / array.shape[0]).squeeze()


class skewness(gw.TimeModule):
    """
    Returns the skewness of the data.

    .. math::

        \\text{skewness} = \\frac{n}{(n-1)(n-2)} \\sum_{i=1}^{n} \\left(\\frac{x_i - \\bar{x}}{s}\\right)^3

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=skewness(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(skewness, self).__init__()

    def calculate(self, array):
        n = array.shape[0]
        mean_ = jnp.nanmean(array, axis=0)
        std_ = jnp.nanstd(array, axis=0)
        skew = (jnp.nansum(((array - mean_) / std_) ** 3, axis=0) * n / ((n - 1) * (n - 2))).squeeze()
        return skew


class standard_deviation(gw.TimeModule):
    """
    Returns the standard deviation of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=standard_deviation(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(standard_deviation, self).__init__()

    def calculate(self, array):
        return jnp.nanstd(array, axis=0).squeeze()


class sum(gw.TimeModule):
    """
    Returns the sum of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=sum(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(sum, self).__init__()

    def calculate(self, array):
        return jnp.nansum(array, axis=0).squeeze()


class symmetry_looking(gw.TimeModule):
    """
    Returns the symmetry looking statistic of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=symmetry_looking(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(symmetry_looking, self).__init__()

    def calculate(self, array):
        n = array.shape[0]
        half = n // 2
        if n % 2 == 0:
            first_half = array[:half]
            second_half = array[half:]
        else:
            first_half = array[:half]
            second_half = array[half + 1:]
        return jnp.nansum(jnp.abs(first_half - second_half), axis=0).squeeze()


class ts_complexity_cid_ce(gw.TimeModule):
    """
    Returns the complexity-invariant distance (CID) of the series.

    Args:
        normalize (bool): If True, normalize the series before calculating CID.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=ts_complexity_cid_ce(normalize=True), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self, normalize=False):
        super(ts_complexity_cid_ce, self).__init__()
        self.normalize = normalize

    def calculate(self, array):
        if self.normalize:
            array = (array - jnp.nanmean(array, axis=0)) / jnp.nanstd(array, axis=0)
        return jnp.nansum(jnp.sqrt(1 + jnp.diff(array, axis=0) ** 2), axis=0).squeeze()


class unique_value_number_to_time_series_length(gw.TimeModule):
    """
    Returns the ratio of unique values to the length of the time series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=unique_value_number_to_time_series_length(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(unique_value_number_to_time_series_length, self).__init__()

    def calculate(self, array):
        return (jnp.unique(array, axis=0).shape[0] / array.shape[0]).squeeze()


class variance(gw.TimeModule):
    """
    Returns the variance of the series.

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=variance(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(variance, self).__init__()

    def calculate(self, array):
        return jnp.nanvar(array, axis=0).squeeze()


class variance_larger_than_standard_deviation(gw.TimeModule):
    """
    Checks if the variance is larger than the standard deviation.

    .. math::

        \\text{var}(x) > \\text{std}(x)

    Example:
        with gw.series(files, nodata=9999) as src:
            src.apply(func=variance_larger_than_standard_deviation(), outfile="test.tif", num_workers=5, bands=1)
    """

    def __init__(self):
        super(variance_larger_than_standard_deviation, self).__init__()

    def calculate(self, array):
        variance_ = jnp.nanvar(array, axis=0)
        std_ = jnp.nanstd(array, axis=0)
        return (variance_ > std_).squeeze()


if __name__ == "__main__":
    # Example usage of one of the functions:
    data = np.random.rand(10, 5, 5)
    abs_energy_instance = abs_energy()
    result = abs_energy_instance.calculate(data)
    print("Absolute Energy:", result)


function_mapping = {
    "abs_energy": abs_energy,
    "absolute_sum_of_changes": absolute_sum_of_changes,
    "autocorrelation": autocorrelation,
    "count_above_mean": count_above_mean,
    "count_below_mean": count_below_mean,
    "doy_of_maximum": doy_of_maximum,
    "doy_of_minimum": doy_of_minimum,
    "kurtosis": kurtosis,
    "kurtosis_excess": kurtosis_excess,
    "large_standard_deviation": large_standard_deviation,
    "longest_strike_above_mean": longest_strike_above_mean,
    "longest_strike_below_mean": longest_strike_below_mean,
    "maximum": maximum,
    "minimum": minimum,
    "mean": mean,
    "mean_abs_change": mean_abs_change,
    "mean_change": mean_change,
    "mean_second_derivative_central": mean_second_derivative_central,
    "median": median,
    "ols_slope_intercept": ols_slope_intercept,
    "quantile": quantile,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "skewness": skewness,
    "standard_deviation": standard_deviation,
    "sum": sum,
    "symmetry_looking": symmetry_looking,
    "ts_complexity_cid_ce": ts_complexity_cid_ce,
    "unique_value_number_to_time_series_length": unique_value_number_to_time_series_length,
    "variance": variance,
    "variance_larger_than_standard_deviation": variance_larger_than_standard_deviation,
}
