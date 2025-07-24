import math
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.path as mpath

def plot_settings(plt_sub):
    plt_sub.xaxis.set_ticks_position("none")
    plt_sub.yaxis.set_ticks_position("none")
    plt_sub.set_yticklabels([])
    plt_sub.set_xticklabels([])
    plt_sub.grid(visible=False)
    plt_sub.grid(visible=False)
    return plt_sub


def plot_images_grid(array_3d, overlay=None, overlay_alpha=0.5,case_id=None, n_col=5, img_size=2, label='', display=False, overlay_cmap='viridis', cmap='gray', interpolation='none'):
    r""" plot 3d array
    args:
        array_3d (numpy.array) : numpy array of images in (z, h, w) order
        case_id (str) : case id of given image
        n_col (int) : number of columns in subplot image (default:5)
        img_size (int) : size of each grid (default:2)
        label (str) : label of given image (default: '')
        display (bool) : whether or not to display image (this might take time)
    return:
        plt_fig (plt.fig) : figure to be saved
    """
    len_rows = int(math.ceil(len(array_3d)/n_col))
    plt_fig = plt.figure(figsize=( n_col*img_size, len_rows*img_size))
    plt_axis = plt_fig.subplots(len_rows,n_col)
    for idx in range(len(array_3d)):

        array_2d = array_3d[idx, :, :] 
        if len_rows == 1:
            plt_sub = plot_settings(plt_axis[idx%n_col])
        else:
            plt_sub = plot_settings(plt_axis[idx//n_col][idx%n_col])
        plt_sub.imshow(array_2d, cmap=cmap)
        if overlay is not None:
            overlay_mask = (overlay[idx] != 0)*overlay_alpha
            plt_sub.imshow(overlay[idx], cmap=overlay_cmap, alpha=overlay_mask, interpolation=interpolation)
        
    for i in range(idx, len_rows*n_col):
        if len_rows == 1:
            plt_sub = plt_axis[i%n_col]
        else:
            plt_sub = plt_axis[i//n_col][i%n_col]
        plt_sub.set_axis_off()
    #plt_fig.tight_layout()
    if case_id:
        plt_fig.suptitle(f'{case_id} | {label}')
    if not display:
        plt.close()
    return plt_fig


# Define a function to create a rounded square path
def create_rounded_square():
    # Vertices for a rounded square path
    vertices = [(0.2-0.5, 0.0-0.5),
         (0.8-0.5, 0.0-0.5), # start of the lower right corner
         (1.0-0.5, 0.0-0.5), # intermediate point (as if it wasn't rounded)
         (1.0-0.5, 0.2-0.5), # end point of the lower right corner
         (1.0-0.5, 0.8-0.5), # move to the next point etc.
         (1.0-0.5, 1.0-0.5),
         (0.8-0.5, 1.0-0.5),
         (0.2-0.5, 1.0-0.5),
         (0.0-0.5, 1.0-0.5),
         (0.0-0.5, 0.8-0.5),
         (0.0-0.5, 0.2-0.5),
         (0.0-0.5, 0.0-0.5),
         (0.2-0.5, 0.0-0.5)]

    codes = [mpath.Path.MOVETO,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3]
    return mpath.Path(vertices, codes)

ROUND_SQAURE = create_rounded_square()