import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



# Paths for dataset
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


# Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    label_mapping = {char: idx for idx, char in enumerate("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    
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

# Define the model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(52, activation='softmax') # Adjusted for 36 classes (0-9, a-z, A-Z)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
#model.fit(x_train, y_train, epochs=2)
model.fit(x_train, y_train, epochs=20, validation_split=0.2, batch_size=64)
# DO NOT REDUCE THE NUMBER OF EPOCHS. Let it run for 15 minutes if needed.
# Normal number of epochs = 20

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Accuracy on test data: {test_acc}')

# Function to make a prediction on a single image
def predict_single_image(image):
    image = image.reshape(1, 28, 28, 1)  # Adjust image dimensions
    prediction = model.predict(image)
    print(f'Prediction probabilities: {prediction}')  # Debug: Display prediction probabilities
    predicted_class = np.argmax(prediction)
    return predicted_class


# Function to test images from the MNIST dataset
def test_mnist_images():
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