import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

def set_matplotlib_settings():
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['figure.figsize'] = (5, 4)
    plt.rcParams["figure.autolayout"] = True
    mpl.rcParams['legend.loc'] = 'upper right'  # Set legend location
    mpl.rcParams['legend.fontsize'] = 'x-small'
    return


def set_matplotlib_multiplot_settings():
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams["figure.autolayout"] = True
    mpl.rcParams['legend.loc'] = 'upper right'  # Set legend location
    mpl.rcParams['legend.fontsize'] = 'x-small'
    return

# For getting square displays of some number of things to plot
def closest_factors(n):
    """
    Return a tuple (rows, cols) such that:
    - rows * cols == n
    - rows <= cols
    - The difference between rows and cols is minimized
    """
    sqrt_n = int(math.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)  # fallback for n == 0 or 1
    