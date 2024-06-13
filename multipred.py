from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import normalize
from glob import glob
from skimage.color import rgb2gray, rgba2rgb
from stardist import random_label_cmap
from stardist.models import StarDist2D
from tifffile import imread
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from collections import Counter
import copy

# displays original pixels without any smoothing or interpolation
matplotlib.rcParams["image.interpolation"] = "none"

# set a random seed for reproducibility
np.random.seed(42)

# generates a random color map for labeled regions
lbl_cmap = random_label_cmap()

# directory containing images for prediction
image_dir = "prediction"

# sorts files by name
file_names = sorted(os.listdir(image_dir))

# List of file extensions to include
extensions = ["*.tiff", "*.tif"]

# Combine the results for both extensions
X = sorted([file for ext in extensions for file in glob(os.path.join(image_dir, ext))])

# read images from X
X = list(map(imread, X))

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

# define the axes for normalization
axis_norm = (0, 1)

# load the specific model you created from the directory of your models
model = StarDist2D(None, name="datasize_9/customModel_9_epochs_100", basedir="models")

# Function to extract class information from the prediction result
def class_from_res(res):
    cls_dict = dict((i+1, c) for i, c in enumerate(res['class_id']))
    return cls_dict

# Function to split masks by cell type
def class_splitter(mask, cls_dict):
    # 1 is Nonviable and 2 is viable
    # the first return is the mask of all viable and the second is nonviable
    mask_len = len(mask)
    mask_height = len(mask[0])
    mask_viable = copy.deepcopy(mask)
    mask_nonviable = copy.deepcopy(mask)
    for i in range(mask_len):
        for j in range(mask_height):
            if mask[i][j] == 0:
                pass
            elif cls_dict[mask[i][j]] == 1:
                mask_viable[i][j] = 0
            elif cls_dict[mask[i][j]] == 2:
                mask_nonviable[i][j] = 0
    return mask_viable, mask_nonviable


# Function for prediction
def prediction(model, i, show_dist=True):
    # Normalize the image for better prediction quality
    img = normalize(X[i], 1, 99.8, axis=axis_norm)

    # If image has four channels (RGBA), convert it to RGB
    if img.ndim == 3 and img.shape[-1] == 4:
        img = rgba2rgb(img)

    # If image has more than one channel, convert to grayscale
    if img.ndim == 3 and img.shape[-1] != 1:
        img = rgb2gray(img)

    # Expand dimensions to add a channel axis if needed
    if img.ndim == 2:
        img = img[..., np.newaxis]

    # Check the number of channels
    if img.shape[-1] != model.config.n_channel_in:
        raise ValueError(f"Expected {model.config.n_channel_in} input channels, but got {img.shape[-1]}")

    # Process the image in tiles to avoid OOM error
    n_tiles = (1, 2, 1)  # You can adjust this to use more tiles
    labels, details = model.predict_instances(img, axes="YXC", n_tiles=n_tiles)

    # Extract classes if available
    classes = class_from_res(details) if 'class_id' in details else None

    if classes is not None:
        # Split the mask by cell type
        mask_viable, mask_nonviable = class_splitter(labels, classes)

        # Count the number of V and NV instances
        num_v = len(np.unique(mask_viable)) - 1  # subtract 1 to exclude the background label 0
        num_nv = len(np.unique(mask_nonviable)) - 1  # subtract 1 to exclude the background label 0
        print(f"Image {i}: V = {num_v}, NV = {num_nv}")
    else:
        num_v = num_nv = 0
        print(f"Image {i}: No class information available")

    # Count the number of unique labels (subtract 1 to exclude the background label 0)
    num_objects = len(np.unique(labels)) - 1

    # Get the file name for the current image
    file_name = file_names[i]

    # Initialize a plot of specified size
    plt.figure(figsize=(13, 10))

    # Checks if the image is grayscale or color for visualization
    img_show = img[..., 0] if img.ndim == 3 else img

    # Display the segmented labels over the original image
    plt.imshow(img_show, cmap="gray")
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)

    # Add the title from the first image
    plt.title(f"Predicted Objects: {num_objects}\nV = {num_v}, NV = {num_nv}", fontsize=16)

    # Hide axes
    plt.axis("off")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure to your specified directory
    plt.savefig(f"prediction/prediction_test{i}.png", dpi=500)
    plt.close()

# Starts the process of predicting
# Will predict for each image in X
for i in range(len(X)):
    prediction(model, i)

print("Prediction Complete.")
