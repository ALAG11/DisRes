# Importing Packages
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image, ImageFile
import tensorflow as tf
import json
import glob
import random
import pathlib
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

import gradio as gr
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths to the dataset
dataset_path = "/kaggle/working/train"
train_data_dir = pathlib.Path(dataset_path + "/images")
test_data_dir = pathlib.Path("/kaggle/working/test/images")

# Prepare training and testing datasets
train_files = glob.glob(r"" + dataset_path + "/images/*.png")
train_files = list(filter(lambda x: "post" in x, train_files))
train_files = random.sample(train_files, 1500)
train_datasize = len(train_files)
print("Training data:", train_datasize)

test_files = glob.glob(r"" + "/kaggle/working/test/images/*.png")
test_files = list(filter(lambda x: "post" in x, test_files))
test_files = random.sample(test_files, 500)
test_datasize = len(test_files)
print("Test data:", test_datasize)

# Image dimensions
img_height = 1024
img_width = 1024
class_names = np.array(['no damage', 'minor damage', 'major damage', 'destroyed'])
print(class_names)

# Label extraction function
def get_label(file_path, type):
    parts = file_path.split(os.path.sep)

    # Define path for test labels
    path = dataset_path+'/labels/'
    if type == "test":
        path = '/kaggle/working/test/labels/'
    f = open(path + parts[-1].split('.')[0] +'.json')
    data = json.load(f)

    building_list = data["features"]["xy"]
    total_damage_level = 0
    for i in range(len(building_list)):
        building = building_list[i]
        damage_level = building['properties']['subtype']
        if damage_level == 'no-damage':
            total_damage_level += 0
        elif damage_level == 'minor-damage':
            total_damage_level += 1
        elif damage_level == 'major-damage':
            total_damage_level += 2
        else:
            total_damage_level += 3
    f.close()

    average_damage_level = total_damage_level
    if len(building_list) > 0:
        average_damage_level /= len(building_list)

    label = class_names[round(average_damage_level)] == class_names
    one_hot = np.zeros(len(class_names), dtype=np.uint8)
    one_hot[label] = 1

    return one_hot

# Label decoding function
def get_label_from_one_hot(array):
    return class_names[np.where(array == 1)]

# Preprocessing the training and testing datasets
train_X = np.zeros((train_datasize, img_height, img_width, 3), dtype=np.uint8)
train_Y = np.zeros((train_datasize, len(class_names)), dtype=np.uint8)

for i in range(len(train_files)):
    img = PIL.Image.open(train_files[i])
    train_X[i] = np.array(img)
    train_Y[i] = get_label(train_files[i], "train")
print("Train", train_X.shape, train_Y.shape)

test_X = np.zeros((test_datasize, img_height, img_width, 3), dtype=np.uint8)
test_Y = np.zeros((test_datasize, len(class_names)), dtype=np.uint8)

for i in range(len(test_files)):
    img = PIL.Image.open(test_files[i])
    test_X[i] = np.array(img)
    test_Y[i] = get_label(test_files[i], "test")
print("Test", test_X.shape, test_Y.shape)

# Plot a few training images with labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    choice = random.randint(0, train_datasize-1)
    plt.title(get_label_from_one_hot(train_Y[choice]))
    plt.imshow(train_X[choice])

plt.tight_layout()
plt.show()

# Build the DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the base model

model = models.Sequential([
    layers.Input(shape=(1024, 1024, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define the maximum number of epochs
max_epochs = 50
batch_size = 16

# Create an EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=5,          # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
)

# Train the model with EarlyStopping
history = model.fit(
    train_X, train_Y,
    validation_data=(test_X, test_Y),
    epochs=max_epochs,
    batch_size=batch_size,
    callbacks=[early_stopping]  # Pass the callback
)

# Plot training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(max_epochs)

plt.figure(figsize=(14, 5))

# Training and validation accuracy graph
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Training and validation loss graph
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Evaluate the model on the test dataset
print("\nEvaluating model on test data:\n")
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)

# Print loss and accuracy
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict on the test dataset
#test_predictions = model.predict(test_X)
#test_pred_labels = np.argmax(test_predictions, axis=1)
#true_test_labels = np.argmax(test_Y, axis=1)

# Accuracy Score
#accuracy = accuracy_score(true_test_labels, test_pred_labels)
#print(f"\nAccuracy on test data: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(true_test_labels, test_pred_labels)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
report = classification_report(true_test_labels, test_pred_labels, target_names=class_names)
print("\nClassification Report:\n", report)

# Save as a Keras model (Native Keras format)
model.save('Multi_Class_Regional_Damage_Classification_Model.keras')

# Save as a .h5 file (HDF5 format), explicitly including the optimizer
model.save('Multi_Class_Regional_Damage_Classification_Model.h5', include_optimizer=True)

# Export as TensorFlow SavedModel format (directory format)
model.export('Multi_Class_Regional_Damage_Classification_Model')

# Gradio Interface for Inference
def disaster_classification(img):
    # Resize the image to the input size expected by the model
    img_resized = np.array(Image.fromarray(img).resize((img_height, img_width)))

    # Normalize the image (1./255 scaling as used in the model's preprocessing layer)
    img_resized = img_resized / 255.0

    # Prepare the image for prediction
    image = np.zeros((1, img_height, img_width, 3), dtype=np.float32)
    image[0] = img_resized

    # Make prediction using the model
    prediction = model.predict(image).tolist()[0]

    # Return a dictionary with class names and their predicted probabilities
    return {class_names[i]: prediction[i] for i in range(len(class_names))}

# Define Gradio interface
iface = gr.Interface(
    fn=disaster_classification,
    inputs=gr.Image(image_mode='RGB', type='numpy'),
    outputs=gr.Label(),
    title="Disaster Regional Damage Classification",
    description="Upload an image to classify the extent of disaster damage."
)

# Launch the interface
iface.launch(share=True, show_error=True, debug=True)