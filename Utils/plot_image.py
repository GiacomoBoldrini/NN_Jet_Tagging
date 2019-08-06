import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import LogNorm

def plot_image(image_vector, normalized = True, Save = False, outpath = "./", title = "Image"):
    plt.figure(figsize=(15,15))
    SUM_Image = np.sum(image_vector, axis = 0)
    if normalized:
        plt.imshow(SUM_Image/float(image_vector.shape[0]), origin='lower',norm=LogNorm(vmin=0.01))
    else:
        plt.imshow(SUM_Image, origin='lower',norm=LogNorm(vmin=0.01))
    plt.colorbar()
    plt.xlabel("Δη", fontsize=20)
    plt.ylabel("ΔΦ", fontsize=20)
    plt.title(title, fontsize=20)
    if Save:
        plt.savefig(outpath+"density_jets.pdf")
    else:
        plt.show()