import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from BaselineRemoval import BaselineRemoval


def get_wide_table(path):
    df = pd.read_csv(path, sep="\t",
                     names=["wavenumber", "ignore",
                            "spectrum", "intensity"])
    return df.pivot(index="wavenumber", columns="spectrum",
                  values="intensity")


def splines(x, y, steps=None):
    # assume already sorted...
    xmin = x[0]
    xmax = x[-1]
    steps = steps if steps else int((xmax-xmin)*4)
    x_spline = np.linspace(xmin, xmax, num=steps, endpoint=True)
    y_spline = interp1d(x, y)(x_spline)
    return x_spline, y_spline


def get_wavenumber(df, wavenumber: float):  # => tuple (np.array, np.array)
    data = df.filter(items=[wavenumber], axis=0)
    x = data.columns.to_numpy()                  # get column names
    y = data.to_numpy()[0, :]                     # get values from the row
    return (x, y)


def get_spectrum(df, spectrum: int):
    data = df.filter(items=[spectrum], axis=1)
    x = data.index.to_numpy()                    # get row names
    y = data.to_numpy()[:, 0]                     # get values from the column
    return (x, y)


def find_wavenumbers(df, approx: str):  # => a NumPy array of search results
    search = df.filter(like=approx, axis=0)
    return search.index.to_numpy()


def find_nearest_wavenumber(df, wavenumber: float):
    return float(find_wavenumbers(df, str(wavenumber))[0])


def plot_wavenumber(x, y) -> None:
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel('Raman intensity')
    ax.set_xlabel('Spectrum number')
    ax.plot(*splines(x, y))


def plot_spectrum(x, y) -> None:
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel('Raman intensity')
    ax.set_xlabel('Wavenumber / cm$^{-1}$')
    ax.plot(*splines(x, y))


def BGR(data, method='ZhangFit', polynomial_degree=2, lambda_=100,
        porder=1, itermax=15):
    '''Background correction of the data

    Args:
        data: data to background correct
        method:
            ZhangFit [https://doi.org/10.1039/B922045C]:
                porder: signal-to-noise ratio (typically 3)
                lambda_ : smoothing-factor for the background (typically in the order of 500)
                itermax: number of iterations
            ModPoly [https://doi.org/10.1366/000370203322554518]
            ModPoly [https://doi.org/10.1366/000370207782597003]
                polynomial_degree: degree of the polynomial for background correction (typically 3)

    Returns:
        Background corrected data

    Raises:
        Method not found. Possible options are: ZhangFit, ModPoly and IModPoly. Check spelling.
    '''

    baseObj = BaselineRemoval(data)
    if method == 'ZhangFit':
        BGR_output = baseObj.ZhangFit(lambda_=lambda_, porder=porder, itermax=itermax)
    elif method == 'ModPoly':
        BGR_output = baseObj.ModPoly(polynomial_degree)
    elif method == 'IModPoly':
        BGR_output = baseObj.IModPoly(polynomial_degree)
    else:
        raise Exception('Method not found. Possible options are: ZhangFit, ModPoly and IModPoly. Check spelling.')
    return BGR_output
