import os
from copy import copy, deepcopy
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
#import solver  # try pymcr instead...



class Table:
    def __init__(self, df, title, mode="raman"):
        # the df must be in wide format!
        self.df = df
        self.title = title
        self.wavenumbers = df.index.to_numpy()
        self.spectrum_numbers = df.columns.to_numpy()
        self.duration = None
        self.mode = mode

    def spectrum(self, spectrum_number: int) -> "Spectrum":
        # get values from the column
        y = (self.df.filter(items=[spectrum_number], axis=1)
                 .to_numpy()[:, 0])  # values in a column
        return Spectrum(self.wavenumbers, y, self.mode)

    def timeplot(self, wavenumber: float) -> "Timeplot":
        y = (self.df.filter(items=[wavenumber], axis=0)
                 .to_numpy()[0, :])  # values in a row
        return Timeplot(self.spectrum_numbers, y, self.duration)

    def spectra(self) -> "Spectra":
        s = Spectra(mode=self.mode)
        for spectrum_number in self.spectrum_numbers:
            s[spectrum_number] = self.spectrum(spectrum_number)
        return s

    def timeplots(self) -> "Timeplots":
        t = Timeplots(mode=self.mode)
        for wavenumber in self.wavenumbers:
            t[wavenumber] = self.timeplot(wavenumber)
        return t

    def find_wavenumbers(self, approx: str) -> np.ndarray:
        search = self.df.filter(like=approx, axis=0)
        return search.index.to_numpy()

    def find_nearest_wavenumber(self, wavenumber: float) -> float:
        return float(self.find_wavenumbers(str(wavenumber))[0])
    
    #def mcr(self, components=3):
        #time_dimension = len(self.spectrum_numbers)
        #wave_dimension = len(self.wavenumbers)
        #A = self.df.to_numpy().transpose()
        #CInit = np.zeros((time_dimension, components))
        #SInit = np.zeros((components, wave_dimension))
        #return solver.solve(A, CInit, SInit)
    
    # def mcr(self, components=3, init="s"):
    #     from pymcr.mcr import McrAR
    #     mcrar = McrAR()
    #     time_dimension = len(self.spectrum_numbers)
    #     wave_dimension = len(self.wavenumbers)
    #     A = self.df.to_numpy().transpose()
    #     c = self.timeplot(self.find_nearest_wavenumber(650)).ydata
    #     CInit = np.array([c, c, c]).transpose()
    #     s = self.spectrum(1).ydata
    #     SInit = np.array([s, s, s])
    #     return mcrar.fit(A, ST)

    def mcr(self, spectrum_numbers):
        # TODO: implement different MCR modes
        from pymcr.mcr import McrAR
        mcrar = McrAR()
        A = self.df.to_numpy().transpose()
        ST0 = np.array(
                       [self.spectrum(i)
                            .ydata for i in spectrum_numbers])
        mcrar.fit(A, ST=ST0)
        C = Timeplots()
        for i in range(len(mcrar.C_.transpose())):
            C["Component " + str(i)] = Timeplot(
                self.spectrum_numbers,
                mcrar.C_.transpose()[i])
        ST = Spectra()
        for i in range(len(mcrar.ST_)):
            ST["Component " + str(i)] = Spectrum(
                self.wavenumbers,
                mcrar.ST_[i])
        return (C, ST)


class Slice:
    def __init__(self, xdata, ydata, mode="raman"):
        self.xdata = xdata
        self.ydata = ydata
        self.title = ""
        self.xlabel = ""
        if mode == "raman":
            self.ylabel = "Raman intensity"
        elif mode == "emission":
            self.ylabel = "Emission intensity"
        self.interpolated = False
        self.normalized = False

    def plot(self, savepath=None):  # should take xmin, xmax args
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        ax.plot(self.xdata, self.ydata)
        if savepath is not None:
            plt.savefig(savepath)

    def write(self):
        ...

    def crop(self, lower, upper):
        xdata, ydata = self.xdata, self.ydata
        mask = (lower <= xdata) & (xdata <= upper)

        tmp = copy(self)
        tmp.xdata = xdata[mask]
        tmp.ydata = ydata[mask]
        return tmp

    def interpolate(self, steps=None):
        xmin = min(self.xdata)
        xmax = max(self.xdata)
        steps = steps if steps else int((xmax-xmin)*4)
        xspline = np.linspace(xmin, xmax, num=steps, endpoint=True)
        yspline = interp1d(self.xdata, self.ydata)(xspline)

        tmp = copy(self)
        tmp.xdata = xspline
        tmp.ydata = yspline
        tmp.interpolated = True
        return tmp

    def pick(self, center, bnd):
        ...


class Spectrum(Slice):
    def __init__(self, xdata, ydata, mode="raman"):
        super().__init__(xdata, ydata, mode)
        if mode == "raman":
            self.title = "Raman spectrum"
            self.xlabel = "wavenumber / cm$^{-1}$"
            self.ylabel = "Raman intensity"
        elif mode == "emission":
            self.title = "Emission spectrum"
            self.xlabel = "wavelength / nm"
            self.ylabel = "Emission intensity"
        else:
            raise ValueError("Invalid mode " + str(mode))
        self.spectrum_number = None
        self.normalized = False
        self.subtracted = False
        self.BGRed = False

    def norm_peak(self, center=1373, bnd=5, steps=None) -> "Spectrum":

        '''Normalize data on the maximum in the given range peak plus/minus bnd
        and shows a plot of the normalized data in the given plot-range
        [xmin_p, xmax_p]

        Args:
            peak: estimation of the rel. wavenumber of the peak maximum for
                normalization (integer/float, default: 1373 (acetonitrile))
            bnd: boundaries added and subtracted from peak for finding a local
                maximum (interger, default: 5)
        Returns:
            Instance of the class (normalized spectra)
        '''

        xdata, ydata = self.xdata, self.ydata
        lower, upper = center - bnd, center + bnd
        maximum = ydata[(lower <= xdata) & (xdata <= upper)].max()
        norm_ydata = ydata / maximum

        tmp = copy(self)
        tmp.normalized = True
        tmp.ydata = norm_ydata
        return tmp

    def subtract(self, subtrahend: "Spectrum"):
        diff_ydata = self.ydata - subtrahend.ydata

        tmp = copy(self)
        tmp.subtracted = True
        tmp.ydata = diff_ydata
        return tmp

    def BGR(self, method, polynomial_degree, lambda_, porder, itermax):
        from BaselineRemoval import BaselineRemoval
        # TODO: implement!


class Timeplot(Slice):
    def __init__(self, xdata, ydata, mode="raman", duration=None):
        super().__init__(xdata, ydata, mode)
        self.title = "Raman spectrum"
        self.xlabel = "spectrum number"
        if duration is not None:
            self.xdata = deepcopy(self.xdata)
            self.xdata *= duration
            self.xlabel = "time / minutes"


class Slices(dict):
    def __init__(self, mode="raman", axis=0):
        #self.title = title
        #self.common_xdata = 
        if mode == "raman":
            self.title = "Raman spectra"
            self.xlabel = "wavenumber / cm$^{-1}$"
            self.ylabel = "Raman intensity"
        elif mode == "emission":
            self.title = "Emission spectra"
            self.xlabel = "wavelength / nm"
            self.ylabel = "Emission intensity"
        else:
            raise ValueError("Invalid mode " + str(mode))
        

    def interpolate(self, steps=None):
        ...

    def plot(self, savepath=None):
        nr = 1
        frames_per_plot = 30
        nr_of_frames = len(self)
        nr_of_plots = int(nr_of_frames / frames_per_plot)+1
        plt.rcParams.update({'font.size': 16})
        f, ax = plt.subplots(nr_of_plots, sharex=True, figsize=(10, 5*nr_of_plots))
        if nr_of_plots == 1:
            ax = [ax]
            nth_plot_title = lambda n: self.title
        else:
            nth_plot_title = lambda n: self.title + ', part ' + str(n+1)
        for plot_nr in range(nr_of_plots):
            first_frame_idx = plot_nr * frames_per_plot
            last_frame_idx = min(len(self), (plot_nr + 1) * frames_per_plot)
            for idx, key in enumerate(islice(self, first_frame_idx, last_frame_idx)):
                x_plot = self[key].xdata
                y_plot = self[key].ydata
                symbol = ['-', '-.', '--'][int(idx/10)]
                ax[plot_nr].plot(x_plot, y_plot, symbol, linewidth=3, label=str(key))
            ax[plot_nr].set_ylabel(self.ylabel)
            ax[plot_nr].set_xlabel(self.xlabel)
            ax[plot_nr].set_title(nth_plot_title(plot_nr))
            ax[plot_nr].legend(ncol=3, bbox_to_anchor=(1.5, 1.0), loc='upper right')
        if savepath is not None:
            plt.savefig(savepath)

    def write(self):
        ...
        
    #def to_dataframe(self):
        #if self.axis == 0:  # add rows
            #df = pd.DataFrame(index=self[0].xdata)
            #shape = lambda x: x
        #elif self.axis == 1:  # add columns
            #df = pd.DataFrame(columns=self[1].xdata)
        #else:
            #raise TypeError()
        #for key in self:
            #df.append_row 
        #df
        #return 


class Spectra(Slices):
    def __init__(self, mode="raman"):
        super().__init__(mode)
        self.title+=" at different times"

    def norm_peak(self, peak=650, bnd=5):
        tmp = copy(self)
        for key in self:
            tmp[key] = self[key].norm_peak(peak, bnd)
        return tmp

    def subtract(self, subtrahend):
        ...

    def BGR(self, method, polynomial_degree, lambda_, porder, itermax):
        ...
    
    # concaternate ydata values as rows
    def to_table(self):
        a = []
        for key in self:
            a.append(self[key].ydata)
        # transpose the rows to columns
        a = np.array(a).transpose()
        # then convert the array to a dataframe
        keys = list(self.keys())
        df = pd.DataFrame(a, index=self[keys[0]].xdata, columns=keys)
        return Table(df, "Reconstituted table")


class Timeplots(Slices):
    def __init__(self, mode="raman"):
        super().__init__(mode)
        self.xlabel = "Spectrum number"
        self.title = "Time varience by wavelength"


def read_file_unicolumn(folder, filename, mode="raman"):
    # assuming single-column; converts to wide (multicolumn) format
    path = os.path.join(folder, filename)
    df = (pd.read_csv(path,
                      sep="\t",
                      names=["wavenumber",
                             "ignore",
                             "spectrum",
                             "intensity"])
            .pivot(index="wavenumber",
                   columns="spectrum",
                   values="intensity")
          )
    return Table(df, filename, mode)


def read_file_multicolumn(folder, filename):
    ...  # TODO: what does multicolumn output look like?


def read_file_simple(folder, filename, mode="raman"):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path,
                     sep="\t",
                     names=["wavenumber",
                             "ignore",
                             "spectrum",
                             "intensity"])
    xdata = df["wavenumber"].to_numpy()
    ydata = df["intensity"].to_numpy()
    return Spectrum(xdata, ydata, mode)


def read(folder, filename):
    """
    Determines the filetype (simple, single or multicolumn) and calls the
    appropriate helper function.
    """
    ...
