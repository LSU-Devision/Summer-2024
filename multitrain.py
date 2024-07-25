# necessary imports
from __future__ import print_function, unicode_literals, absolute_import, division
from collections import OrderedDict
from csbdeep.utils import normalize
from glob import glob
from stardist import fill_label_holes, calculate_extents
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from PIL import Image
from tqdm import tqdm
import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import warnings

# function to parse arguments given in command line
def parse_args():
    parser = argparse.ArgumentParser(description="Here is the help section for the optional commands.")
    parser.add_argument("--epochs", type=int, nargs='+', default=[10], 
        help="Sets the number of epochs. Accepts a number or list of numbers. E.g. --epochs 10 50 100 300. Default: 10.")
    return parser.parse_args()

# function to read images using Pillow
def read_image(filename):
    with Image.open(filename) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return np.array(img)
        #return img.convert('L')  # Convert image to grayscale
    
# main function
def main(args):
    warnings.filterwarnings("ignore")
    
    # ensures that any random operations are reproducible across all runs
    random.seed(42)

    # X_filenames represents the raw images
    X_filenames = sorted(glob("images/*.*"))

    # read images from X
    X = list(map(read_image, X_filenames))
    print("NORMALIZING IMAGES...")
    X = [normalize(x, 1, 99.8, axis = (0,1) ) for x in tqdm(X)]

    # Y represents the masks
    Y_filenames = sorted(glob("masks/*.*"))

    # read images from X
    Y = list(map(read_image, Y_filenames))
    print("FILLING HOLES IN LABELS...")
    # applied to each mask in Y, ensuring that any labeled regions are solid without internal holes
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    # Assuming class dictionaries are stored in 'class_dicts/' with filenames matching the images
    cls_filenames = sorted(glob("labels/*.*"))

    def process_file(file_path):
        label_dict = {}
        
        with open(file_path, "r") as file:
            for line in file:
                # Remove leading/trailing whitespaces
                line = line.strip()
                
                # Use regex to extract the number and label
                match = re.match(r"Label\s*(\d+)\s*:\s*(UF|F)", line, re.IGNORECASE)
                if match:
                    label_number = int(match.group(1))
                    label_value = match.group(2).upper()
                    
                    # Convert the label value to 1 for 'UF' and 2 for 'F'
                    if label_value == "UF":
                        label_dict[label_number] = 1
                    elif label_value == "F":
                        label_dict[label_number] = 2
                    #print(f"Added label: {label_number} with value: {label_dict[label_number]}")
                else:
                    print(f"Unmatched line in {file_path}: {line}")
        # Sort the dictionary by keys and return it
        sorted_label_dict = OrderedDict(sorted(label_dict.items()))
        return sorted_label_dict
    
    C = [process_file(f) for f in cls_filenames]

    # configuration object with model parameters to be used in initilization/training
    conf = Config2D(
        n_rays          = 32,
        grid            = (4, 4),
        n_channel_in    = 3,
        n_classes       = 2
    )

    # function for random flips/rotations of the data
    def random_fliprot(img, mask): 
        assert img.ndim >= mask.ndim
        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))
        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(perm) 
        for ax in axes: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 

    # function to give data random intensity
    def random_intensity_change(img):
        img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        return img

    # this is the augmenter that will go into the training of the model
    def augmenter(x, y):
        x, y = random_fliprot(x, y)
        x = random_intensity_change(x)
        sig = 0.02*np.random.uniform(0,1)
        x = x + sig*np.random.normal(0,1,x.shape)
        return x, y
    
    testing_size = int(.2 * len(X))

    # determine sizes for training and validation
    training_size = int(.8 * (len(X) - testing_size))

    # Generate a list of random indices
    test_indices = random.sample(range(len(X)), testing_size)

    # Select images, masks, and class dictionaries for the test set
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    # Remove the test set from the original lists
    X = [X[i] for i in range(len(X)) if i not in test_indices]
    Y = [Y[i] for i in range(len(Y)) if i not in test_indices]
    C = [C[i] for i in range(len(C)) if i not in test_indices]

    # Generate a list of random indices for the remaining data
    remaining_indices = list(range(len(X)))
    random.shuffle(remaining_indices)

    # Split the remaining indices into training and validation
    train_indices = remaining_indices[:training_size]
    val_indices = remaining_indices[training_size:]

    # Get your training datasets
    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    C_train = [C[i] for i in train_indices]

    # Get your validation datasets
    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]
    C_val = [C[i] for i in val_indices]

    print(f"training set size: {len(X_train)}")
    print(f"validation set size: {len(X_val)}")
    print(f"testing set size: {len(X_test)}")

    # function to save training, validation, and testing data in a grid to a PNG file
    def save_images_to_file(images, filename, title):
        # specify the dimensions of the subplot grid
        n = len(images)
        cols = int(math.sqrt(n))  # assuming you want a square grid, change this as per your requirements
        rows = int(math.ceil(n / cols))

        # create a new figure with specified size
        fig = plt.figure(figsize=(20, 20))  # adjust as needed

        # set title
        plt.title(title, fontsize=40)  # adjust font size as needed

        # iterate over each image and add it to the subplot
        for i in range(n):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.imshow(images[i], cmap='gray')  # using gray colormap as these are grayscale images
            ax.axis('off')  # to remove axis

        # adjust layout and save the figure
        fig.tight_layout()  # adjust layout so labels do not overlap
        fig.savefig(filename, dpi=600)



    # where the model will be saved
    base_dir = "models"

    # make a new directory for the dataset size
    dataset_dir = os.path.join(base_dir, f'datasize_{len(X)}')
    os.makedirs(dataset_dir, exist_ok=True)



    # this section saves the training, validation, and testing data in separate images to
    # see the data the program selected using the function from above
    # saving the training images
    training_filename = os.path.join(dataset_dir, 'training_images.png')  # define the path and name for your image
    save_images_to_file(X_train, training_filename, "Training Images")

    # saving the validation images
    validation_filename = os.path.join(dataset_dir, 'validation_images.png')  # define the path and name for your image
    save_images_to_file(X_val, validation_filename, "Validation Images")

    # saving the testing images
    testing_filename = os.path.join(dataset_dir, 'testing_images.png')  # define the path and name for your image
    save_images_to_file(X_test, testing_filename, "Testing Images")

    # function to evaluate and save csv files
    def evaluate_and_save(model, X_data, Y_data, data_type='validation'):

        # prediction
        Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(X_data)]
        
        # evaluation
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset(Y_data, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
        
        # plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
        counts = ('fp', 'tp', 'fn')
        
        for m in metrics:
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in counts:
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        
        # save figure
        figure_filename = os.path.join(model.basedir, model.name, f"{data_type}_plots.png")
        fig.savefig(figure_filename, dpi=300)
        
        # save CSV
        filename = os.path.join(model.basedir, model.name, f'{data_type}_stats.csv')
        fieldnames = list(stats[0]._asdict().keys())
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in stats:
                writer.writerow(entry._asdict())

        return stats

    # number of epochs
    epochs = args.epochs

    # main training loop
    for i in epochs:

        # naming the model
        model_name = "customModel_" + str(len(X)) + '_epochs_' + str(i)

        # instantiate the model with custom parameters
        model = StarDist2D(conf, name=model_name, basedir=dataset_dir)

        # calculates the average size of labeled objects in mask images
        median_size = calculate_extents(list(Y_train), np.median)

        # refers to how much the network can "see" the image in a single pass
        fov = np.array(model._axes_tile_overlap('YX'))

        # printing median size and fov
        print(f"median object size:      {median_size}")
        print(f"network field of view :  {fov}")

        # this is to warn the user that the median object size is larger than the fov
        # which can cause the network to struggle to detect the objects properly
        # this can lead to partial segmentations or missed detections
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")

        # epochs based on where i is in the list of epochs
        epochs = i

        # code to train the model based on the data given
        model.train(X_train, Y_train, classes = C_train, validation_data=(X_val, Y_val, C_val), augmenter=augmenter, epochs=epochs)

        # optimizing thresholds for validation data
        model.optimize_thresholds(X_val, Y_val)

        # evaluation of validation data
        stats_val = evaluate_and_save(model, X_val, Y_val, 'validation')

        # evaluation of testing data
        stats_test = evaluate_and_save(model, X_test, Y_test, 'test')
    print("Training is complete.")

if __name__ == "__main__":
    # loading arguments
    args = parse_args()
    main(args)