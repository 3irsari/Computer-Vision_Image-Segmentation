import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
import cv2


# Load the saved model
model = load_model('best_model.keras')  # Replace with your saved model path

# Define directories for dataset
TRAIN_DIR = 'chest_xray/train'
TEST_DIR = 'chest_xray/test'
VAL_DIR = 'chest_xray/val'

# Loading Dataset
# List of label names
LABELS = ['PNEUMONIA','NORMAL']
# Image size to format them into same size during the loading stage
img_size = 256

def load_image_data(directory,label):

    path = os.path.join(directory,label)
    class_num = LABELS.index(label)
    data,labels = [],[]

    for img in os.listdir(path):
        #img_arr = cv2.imread(os.path.join(path,img))
        img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        resized_arr = cv2.resize(img_arr,(img_size,img_size))
        data.append(resized_arr)
        labels.append(class_num)

    return data,labels

def load_training_data(data_dir):

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

# DATA VISUALIZATION SECTION (EDA) in Data Exploration Analysis.py file

# DATA PROCESSING SECTION

# Normalize and reshape the data for the model
X_train, X_test, X_val = [x / 255.0 for x in [train_data, test_data, val_data]]
y_train, y_test, y_val = map(np.array, [train_labels, test_labels, val_labels])


# Normalize and reshape the data (if needed)

X_val = X_val.reshape(-1,256,256,3)  # Match the image size used during training
X_test = X_test.reshape(-1,256,256,3)  # Match the image size used during training

# Make predictions on the validation set
val_predictions = (model.predict(X_val) >= 0.5).astype(int).reshape(-1)

# Make predictions on the test set
test_predictions = (model.predict(X_test) >= 0.5).astype(int).reshape(-1)


def display_sample_images(X,y,predictions,title,cmap):
    num_samples = 12  # Display 12 samples
    random_indices = np.random.choice(len(X),num_samples,replace=False)

    plt.figure(figsize=(12,6))
    for i,idx in enumerate(random_indices):
        plt.subplot(3,4,i + 1)

        # Convert the image to grayscale
        gray_image = np.dot(X[idx][...,:3],[0.2989,0.5870,0.1140])  # Convert RGB to grayscale

        # Apply the colormap
        colored_image = plt.cm.get_cmap(cmap)(gray_image)  # Get the colormap and apply it
        colored_image = (colored_image[...,:3] * 255).astype(np.uint8)  # Convert back to uint8 format

        # Display the image with the specified colormap
        plt.imshow(colored_image,interpolation='none')  # Display the colored image

        # Set the title with predicted and actual classes
        plt.title(f"Predicted: {predictions[idx]}   Actual: {y[idx]}",fontsize=10)

        # Remove x and y ticks
        plt.axis('off')

    # Set the main title for the figure
    plt.suptitle(title,size=18)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


# Display sample images for validation set with viridis colormap
display_sample_images(X_val,y_val,val_predictions,"Validation Set Predictions",cmap='viridis')

# Display sample images for test set with magma colormap
display_sample_images(X_test,y_test,test_predictions,"Test Set Predictions",cmap='magma')
