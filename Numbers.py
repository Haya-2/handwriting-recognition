import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

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
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=2)
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

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale if needed
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img)  # Convert to a numpy array
    img = img / 255.0
    return img.reshape(28, 28, 1)

# Function to test images from the MNIST dataset
def test_mnist_images():
    while True:
        try:
            print(f'To stop the loop, type -1. Remember to stop the loop before running another program.')
            image_index = int(input("Index of the image to test (0 to 9999): "))
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