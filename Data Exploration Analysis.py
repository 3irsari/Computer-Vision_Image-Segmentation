# Importing Dependencies

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
import cv2
import numpy as np

# Ignore warnings and set TensorFlow optimizations
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.python.data.experimental.ops.distribute import batch_sizes_for_worker

# DATA LOADING SECTION
# Define directories for dataset
TRAIN_DIR = 'chest_xray/train'
TEST_DIR = 'chest_xray/test'
VAL_DIR = 'chest_xray/val'

# Loading Dataset

# List of label names
LABELS = ['PNEUMONIA','NORMAL']
# Image size to format them into same size during the loading stage
img_size = 128

def load_image_data(directory,label):
    """
    :param directory: The base directory where labeled images are stored.
    :param label: The specific label or category whose image data is to be loaded.
    :return: A tuple containing a list of processed image data and their corresponding labels.
    """
    path = os.path.join(directory,label)
    class_num = LABELS.index(label)
    data,labels = [],[]

    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path,img))
        resized_arr = cv2.resize(img_arr,(img_size,img_size))
        data.append(resized_arr)
        labels.append(class_num)

    return data,labels

def load_training_data(data_dir):
    """
    :param data_dir: Directory containing the image data.
    :return: Tuple of numpy arrays containing the training data and labels.
    """
    all_data,all_labels = [],[]

    for label in LABELS:
        data,labels = load_image_data(data_dir,label)
        all_data.extend(data)
        all_labels.extend(labels)

    return np.array(all_data),np.array(all_labels)

# Load and preprocess training, testing, and validation data
train_data,train_labels = load_training_data(TRAIN_DIR)
test_data,test_labels = load_training_data(TEST_DIR)
val_data,val_labels = load_training_data(VAL_DIR)

# DATA VISUALIZATION SECTION

# Parameters working as constants for our visualization to control
figure_size = (18,10)
num_images = 12
# bone, viridis
colormap = 'bone'
title_size = 20

def plot_sample_images(data,LABELS,num_images,figure_size,colormap,title_size):
    """
    :param data: List or array-like object containing image data.
    :param labels: List or array-like object containing image labels, where 0 represents 'Pneumonia' and 1 represents 'Normal'.
    :param num_images: Integer specifying the number of sample images to plot.
    :param figure_size: Tuple specifying the size of the figure.
    :param colormap: Color map to use for displaying images (e.g., 'gray')
    """
    random_indices = np.random.choice(len(data),num_images,replace=False)
    plt.figure(figsize=figure_size)
    for i,idx in enumerate(random_indices):
        plt.subplot(3,4,i + 1)
        plt.imshow(data[idx],cmap=colormap)
        plt.title('Pneumonia' if LABELS[idx] == 0 else 'Normal')
        plt.axis('off')
    plt.suptitle("Sample Set Image Examples",size=title_size)
    plt.tight_layout()
    plt.show()

# Checking number of Sample
plot_sample_images(train_data,train_labels,num_images,figure_size,colormap,title_size)

# Training Data Distribution
train_df = pd.DataFrame({
    "Labels": train_labels,
    "Set": "Train"
})

val_df = pd.DataFrame({
    "Labels": val_labels,
    "Set": "Validation"
})

test_df = pd.DataFrame({
    "Labels": test_labels,
    "Set": "Test"
})

# Combine all DataFrames
combined_df = pd.concat([train_df,test_df,val_df])

# Data Distribution Graph
plt.figure(figsize=(9,5))
##0073e6,ff758f
colors = sns.light_palette("#0073e6",n_colors=7)
ax = sns.countplot(data=combined_df,x='Labels',hue='Set',palette=[colors[1],colors[3],colors[6]])
ax.set_xticklabels(['Pneumonia','Normal'])

# Annotate bars with the counts
for p in ax.patches:
    count = int(p.get_height())  # Get height of each bar (the count)
    ax.annotate(f'{count}',  # Text to annotate with
                (p.get_x() + p.get_width() / 2.,count),  # Position of the text
                ha='center',  # Horizontal alignment
                va='baseline')  # Vertical alignment
# displaying the title
plt.title("Image Distribution Graph")
plt.show()
