import matplotlib.pyplot as plt
import matplotlib as mpl


def confplot():
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "Ubuntu"
    mpl.rcParams["font.monospace"] = "Ubuntu Mono"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["figure.titlesize"] = 12

    plt.style.use("bmh")
