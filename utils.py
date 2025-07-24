import os
import sys
import math
import glob
import pathlib
import warnings
import numpy as np
import cv2
import pydicom
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from einops import rearrange
from skimage import morphology
from scipy import ndimage, stats
import SimpleITK as sitk
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from collections.abc import Iterable

from CT_Preprocessing import PreprocessingCTImage as pp
from CT_Preprocessing import window_image, normalize_array


from Visualization import plot_settings, ROUND_SQAURE



AFFECTED_LIST = [
    [188/255, 121/255, 203/255, 1.0],  # Puple
    [168/255, 225/255, 12/255, 1.0],    # Green
    [57/255, 122/255, 211/255, 1.0],  # Blue
    [255/255, 219/255, 21/255, 1.0], # Yellow
    [255/255, 87/255, 101/255, 1.0],   # Dark red
]
AFFECTED_CMAP = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', AFFECTED_LIST, len(AFFECTED_LIST))

def draw_overlay_cti(raw_img, total_region):
    raw_img = raw_img#[min_idx:max_idx+1]
    total_region = total_region#[min_idx:max_idx+1]
    input_imgs = normalize_array(window_image(raw_img, 40, 80))
    len_image = len(input_imgs)
    nrow = math.ceil(len_image/6)
    # print(len_image/6, math.ceil(len_image/6))
    # if nrow < 6:
    #     nrow=6
    plt_fig = plt.figure(figsize=( (6*3), (nrow)*3))
    plt_fig.patch.set_facecolor('black')
    spec = plt_fig.add_gridspec((nrow)*3, (6*3))
    spec.update(wspace=0.0, hspace=0)

    count = 0
    
    draw_arr, open_arr = [], []
    draw_arr = total_region

    
    for idx in range(len_image):
        #print(idx,input_imgs[idx].shape, true_masks[idx].shape, pred_masks[idx].shape)
        # print([(count//7)*3,(count//7 +1)*3,1+(count%7)*3,1+(count%7+1)*3])
        axes = plt_fig.add_subplot(spec[(count//6)*3:(count//6 +1)*3,(count%6)*3:(count%6+1)*3])#plt_fig.add_subplot(plt_axis[count//6,count%6])
        plt_sub = plot_settings(axes)
        bg_img = input_imgs[idx] == 0
        
        plt_sub.imshow(input_imgs[idx], cmap='gray')
        # Check if there are any non-zero elements and draw the outline if true
        if np.sum(draw_arr[idx] != 0) > 0:
            plt_sub.imshow(bg_img, cmap='gray', alpha=(bg_img!=0)*0.3)
            
            
        plt_sub.imshow(draw_arr[idx], vmax=5, vmin=1, cmap=AFFECTED_CMAP, alpha=(draw_arr[idx]!=0)*0.95, interpolation='none')
        count += 1


    for others in range(count, nrow*6):
        axes =  plt_fig.add_subplot(spec[(count//6)*3:(count//6 +1)*3,1+(count%6)*3:1+(count%6+1)*3])
        plt_sub = plot_settings(axes)
        plt_sub.set_axis_off()
        count += 1

    plt_fig.tight_layout()
    plt_fig.set_facecolor('black')
    plt.close()
    
    np_fig = convert_fig_to_array(plt_fig, format='png', dpi=200, bbox_inches='tight')
    
    ### 250715 np_fig_empty
    
    plt_fig = plt.figure(figsize=( (6*3), (nrow)*3))
    plt_fig.patch.set_facecolor('black')
    spec = plt_fig.add_gridspec((nrow)*3, (6*3))
    spec.update(wspace=0.0, hspace=0)

    count = 0
    
    
    # draw_arr, open_arr = [], []
    draw_arr = total_region
    idx = 0
    for idx in range(len_image):
        #print(idx,input_imgs[idx].shape, true_masks[idx].shape, pred_masks[idx].shape)
        # print([(count//7)*3,(count//7 +1)*3,1+(count%7)*3,1+(count%7+1)*3])
        axes = plt_fig.add_subplot(spec[(count//6)*3:(count//6 +1)*3,(count%6)*3:(count%6+1)*3])#plt_fig.add_subplot(plt_axis[count//6,count%6])
        plt_sub = plot_settings(axes)
        bg_img = input_imgs[idx] == 0
        
        plt_sub.imshow(input_imgs[idx], cmap='gray')
        # Check if there are any non-zero elements and draw the outline if true
        if np.sum(draw_arr[idx] != 0) > 0:
            plt_sub.imshow(bg_img, cmap='gray', alpha=(bg_img!=0)*0.3)
            
            
        # plt_sub.imshow(draw_arr[idx], vmax=5, vmin=1, cmap=AFFECTED_CMAP, alpha=(draw_arr[idx]!=0)*0.95, interpolation='none')
        count += 1


    for others in range(count, nrow*6):
        axes =  plt_fig.add_subplot(spec[(count//6)*3:(count//6 +1)*3,1+(count%6)*3:1+(count%6+1)*3])
        plt_sub = plot_settings(axes)
        plt_sub.set_axis_off()
        count += 1

    plt_fig.tight_layout()
    plt_fig.set_facecolor('black')
    plt.close()
    
    np_fig_empty = convert_fig_to_array(plt_fig, format='png', dpi=200, bbox_inches='tight')
    
    
    return np_fig,np_fig_empty, nrow, np.asarray(draw_arr)

    
def convert_fig_to_array(fig, dpi=100, format='png', **kwargs):

    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, pad_inches=0, **kwargs) #format='png'j,
    buf.seek(0)
    # Use Pillow to open the image and convert it to a NumPy array
    image = Image.open(buf)
    image_np = np.array(image)
    # Close the buffer
    buf.close()
    plt.close()
    # Now `image_np` is a NumPy array representing the image of the figure
    return image_np