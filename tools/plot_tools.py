import matplotlib.pyplot as plt
import os

def save_current_plot(filename="plot.png", dpi=150):
    plt.savefig(filename, dpi=dpi)
    return filename

def list_saved_plots():
    return [f for f in os.listdir() if f.endswith(".png")]
