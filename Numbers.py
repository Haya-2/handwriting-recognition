import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report


# Paths for dataset
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    label_mapping = {char: idx for idx, char in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")}
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                try:
                    image = load_img(file_path, color_mode="grayscale", target_size=(28, 28))
                    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
                    images.append(image)
                    labels.append(label_mapping[label])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(images), np.array(labels)

# Load and preprocess data
x_train, y_train = load_data(TRAIN_DIR)
x_test, y_test = load_data(TEST_DIR)
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")


# Shuffle training data

# Simulated mismatched data
x_train = np.random.rand(2728, 28, 28, 1)  # Features
y_train = np.random.randint(0, 10, 1584)   # Labels

# Print the shapes before fixing
print(f"Before fixing: x_train shape = {x_train.shape}, y_train shape = {y_train.shape}")

# Fix the mismatch
# Fix training data
min_samples_train = min(len(x_train), len(y_train))
x_train = x_train[:min_samples_train]
y_train = y_train[:min_samples_train]

# Fix testing data
min_samples_test = min(len(x_test), len(y_test))
x_test = x_test[:min_samples_test]
y_test = y_test[:min_samples_test]


# Print the shapes after fixing
print(f"After fixing: x_train shape = {x_train.shape}, y_train shape = {y_train.shape}")

# Shuffle the data
x_train, y_train = shuffle(x_train, y_train, random_state=42)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Define the model
# model = models.Sequential([
#    layers.Input(shape=(28, 28, 1)),
#    layers.Conv2D(32, (3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#    layers.Conv2D(64, (3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#    layers.Flatten(),
#    layers.Dense(128, activation='relu'),
#    layers.Dropout(0.2),
#    layers.Dense(64, activation='relu'),
#    layers.Dropout(0.2),
#    layers.Dense(36, activation='softmax') # Adjusted for 36 classes (0-9, a-z)
#])
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(62, activation='softmax') 
])

# Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',  metrics=['accuracy'])

# Train the model
# model.fit(x_train, y_train, epochs=2)
# model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=36)
# DO NOT REDUCE THE NUMBER OF EPOCHS. Let it run for 15 minutes if needed.
# Normal number of epochs = 20

model.summary()
# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split for validation
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    'data/train',  # Path to training data
    target_size=(128, 128),
    batch_size=62,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(128, 128),
    batch_size=62,
    class_mode='categorical',
    subset='validation'
)

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Load test data
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'data/test',  # Path to test data
    target_size=(128, 128),
    batch_size=62,
    class_mode='categorical'
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Evaluate the model on the test data
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f'Accuracy on test data: {test_acc}')

# Function to make a prediction on a single image
def predict_single_image(image):
    image = image.reshape(1, 28, 28, 1)  # Adjust image dimensions
    prediction = model.predict(image)
    print(f'Prediction probabilities: {prediction}')  # Debug: Display prediction probabilities
    predicted_class = np.argmax(prediction)
    return predicted_class


def confusion_matrix_func():
    # Generate predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class indices
    y_true = test_generator.classes  # True class labels

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# Function to test images from the MNIST dataset
def test_mnist_images():
    confusion_matrix_func()
    while True:
        try:
            print(f'To stop the loop, type -1. Remember to stop the loop before running another program.')
            image_index = int(input("Index of the image to test (0 to {}): ".format(len(x_test)-1)))
            if image_index == -1:
                print("Quitting...")
                break
            if 0 <= image_index < len(x_test):
                predicted_class = predict_single_image(x_test[image_index])
                true_class = y_test[image_index]
                print(f'Image {image_index}: True class = {true_class}, Predicted class = {predicted_class}')
                
                # Display the tested image
                # plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
                # plt.title(f'Image #{image_index}, True Digit: {true_class}, Prediction: {predicted_class}')
                # plt.axis('off')
                # plt.show()
                # print(f"=======================================")
            else:
                print("Index out of range. Try a smaller and positive number.")
        except ValueError:
            print("Error: Please enter a valid integer.")

# Launch the main menu
test_mnist_images()