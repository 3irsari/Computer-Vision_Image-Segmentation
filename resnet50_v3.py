
# Importing Dependencies
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
import cv2
import numpy as np

# Ignore warnings
warnings.filterwarnings('ignore')

# DATA LOADING SECTION

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
X_train = X_train.reshape(-1, img_size, img_size, 3)
X_test = X_test.reshape(-1, img_size, img_size, 3)
X_val = X_val.reshape(-1, img_size, img_size, 3)
y_train, y_test, y_val = map(np.array, [train_labels, test_labels, val_labels])

# Enhanced Data Augmentation with ImageDataGenerator
data_generator = ImageDataGenerator(
    rotation_range=40,             # Rotate images up to 40 degrees from 30
    width_shift_range=0.2,         # Shift width up to 20% from 0.1
    height_shift_range=0.2,        # Shift height up to 20% from 0.1
    shear_range=0.2,               # Apply shearing
    zoom_range=0.3,                # Zoom in/out within 30% from 0.2
    #horizontal_flip=True,          # Flip images horizontally (removed since the X-Rays will not be flipped)
    brightness_range=[0.7, 1.3],   # Adjust brightness (darker to brighter)
    fill_mode='nearest',           # Fill mode for empty pixels after shifts
    channel_shift_range=20.0       # Adjust color intensity by shifting channels
)

# Checking and balancing the test data
if len(X_test) > len(y_test):
    # Down sampling X_test to match the number of y_test
    selected_indices = np.random.choice(len(X_test), len(y_test), replace=False)
    X_test = X_test[selected_indices]
elif len(X_test) < len(y_test):
    # Down sampling y_test to match the number of X_test
    selected_indices = np.random.choice(len(y_test), len(X_test), replace=False)
    y_test = y_test[selected_indices]

# Ensuring X_test and y_test have the same number of samples
assert len(X_test) == len(y_test), "X_test and y_test must have the same number of samples"

# Printing the new shapes for verification
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Balancing X_train and y_train
if len(X_train) > len(y_train):
    selected_indices = np.random.choice(len(X_train), len(y_train), replace=False)
    X_train = X_train[selected_indices]
elif len(X_train) < len(y_train):
    selected_indices = np.random.choice(len(y_train), len(X_train), replace=False)
    y_train = y_train[selected_indices]

# Ensuring X_train and y_train have the same number of samples
assert len(X_train) == len(y_train), "X_train and y_train must have the same number of samples"

# Printing the new shapes for verification
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Balancing X_val and y_val
if len(X_val) > len(y_val):
    selected_indices = np.random.choice(len(X_val), len(y_val), replace=False)
    X_val = X_val[selected_indices]
elif len(X_val) < len(y_val):
    selected_indices = np.random.choice(len(y_val), len(X_val), replace=False)
    y_val = y_val[selected_indices]

# Ensuring X_val and y_val have the same number of samples
assert len(X_val) == len(y_val), "X_val and y_val must have the same number of samples"

# Printing the new shapes to verify them
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# LEARNING SECTION (using ResNet50)

# Load the model with pre-trained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the layers to avoid re-training them
for layer in base_model.layers:
    layer.trainable = False

# Custom model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add global pooling layer to reduce parameters -flattens
x = Dense(128, activation='relu')(x)# Fully connected layer
x = Dropout(0.3)(x) # Dropout for regularization
output = Dense(1, activation='sigmoid')(x)  # Sigmoid activation  for binary classification

# Define the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()

# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[-4:]:  # Adjust the number of layers to unfreeze as needed
    layer.trainable = True

# Define callbacks for training
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=0.0001, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    # ModelCheckpoint('best_model_vgg16.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Data generators for training, validation, and testing data
train_generator = data_generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary'
)

val_generator = data_generator.flow_from_directory(
    VAL_DIR,
    target_size=(img_size, img_size),
    batch_size=16,
    class_mode='binary',
    shuffle=False  # Disable shuffle
)

test_generator = data_generator.flow_from_directory(
    TEST_DIR,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Disable shuffle
)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=2
    #,steps_per_epoch=(len(X_train) // 256)
)

# MODEL PERFORMANCE TESTING

# Visualize Training and Validation Metrics
history_data = history.history
epochs = range(1, len(history_data['accuracy']) + 1)

# Retrieve metrics from the training history
train_acc, train_loss = history_data['accuracy'], history_data['loss']
val_acc, val_loss = history_data['val_accuracy'], history_data['val_loss']

# Create a figure and axes for the plots
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Main figure title
fig.suptitle('Model Training and Validation Performance', fontsize=20, fontweight='bold')
# Plot training and validation accuracy
ax[0].plot(epochs, train_acc, 'o-', color='darkgreen', label='Training Accuracy', markersize=8)
ax[0].plot(epochs, val_acc, 's--', color='darkred', label='Validation Accuracy', markersize=8)
ax[0].set_title('Training vs. Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Accuracy', fontsize=14)
ax[0].legend()
ax[0].grid(True)
# Plot training and validation loss
ax[1].plot(epochs, train_loss, 'o-', color='darkblue', label='Training Loss', markersize=8)
ax[1].plot(epochs, val_loss, 's--', color='orange', label='Validation Loss', markersize=8)
ax[1].set_title('Training vs. Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Loss', fontsize=14)
ax[1].legend()
ax[1].grid(True)
# Display the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the subtitle
plt.show()

# Test Set Evaluation
test_eval = model.evaluate(test_generator, verbose=1)
print("==" * 20)
print(f"Test Set Accuracy - {test_eval[1] * 100:.2f}%")
print(f"Test Set Loss - {test_eval[0]:.4f}")
print("==" * 20)

# Predictions and Confusion Matrix for Test Data
test_predictions = (model.predict(test_generator) >= 0.5).astype(int).reshape(-1)
test_cm = confusion_matrix(test_labels, test_predictions)

plt.figure(figsize=(5, 4))
sns.heatmap(pd.DataFrame(test_cm, index=LABELS, columns=LABELS), cmap="Blues", annot=True, fmt="d")
plt.title("Test Data Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()

print(classification_report(test_labels, test_predictions, target_names=LABELS))

# Validation Performance
val_evaluation = model.evaluate(val_generator, verbose=1)
print("==" * 20)
print(f"Validation Set Accuracy - {val_evaluation[1] * 100:.2f}%")
print(f"Validation Set Loss - {val_evaluation[0]:.4f}")
print("==" * 20)

# Predictions and Confusion Matrix for Validation Data
val_predictions = (model.predict(val_generator) >= 0.5).astype(int).reshape(-1)
val_cm = confusion_matrix(y_val, val_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Validation Data Confusion Matrix")
plt.show()

print(classification_report(y_val, val_predictions, target_names=LABELS))

# Visualize Training and Validation Metrics
history_data = history.history
epochs = range(1, len(history_data['accuracy']) + 1)

# Retrieve metrics from the training history
train_acc, train_loss = history_data['accuracy'], history_data['loss']
val_acc, val_loss = history_data['val_accuracy'], history_data['val_loss']

# Create a figure and axes for the plots
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Main figure title
fig.suptitle('Model Training and Validation Performance', fontsize=20, fontweight='bold')

# Plot training and validation accuracy
ax[0].plot(epochs, train_acc, 'o-', color='darkgreen', label='Training Accuracy', markersize=8)
ax[0].plot(epochs, val_acc, 's--', color='darkred', label='Validation Accuracy', markersize=8)
ax[0].set_title('Training vs. Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Accuracy', fontsize=14)
ax[0].legend()
ax[0].grid(True)

# Plot training and validation loss
ax[1].plot(epochs, train_loss, 'o-', color='darkblue', label='Training Loss', markersize=8)
ax[1].plot(epochs, val_loss, 's--', color='orange', label='Validation Loss', markersize=8)
ax[1].set_title('Training vs. Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Loss', fontsize=14)
ax[1].legend()
ax[1].grid(True)

# Display the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the subtitle
plt.show()
