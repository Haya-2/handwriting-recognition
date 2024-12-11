import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Enhanced preprocessing for easier segmentation using noise reduction, thresholding, and morphology.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    scale_percent = 200
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Step 2: Noise reduction using a median blur
    denoised_image = cv2.medianBlur(resized_image, 5)

    # Step 3: Adaptive thresholding with Otsu's thresholding for dynamic binarization
    _, otsu_thresh = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(
        otsu_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )

    # Step 4: Morphological opening to remove small noise and improve text isolation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    cleaned_image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 5: Dilate the text slightly to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    processed_image = cv2.dilate(cleaned_image, kernel, iterations=1)

    return processed_image, (width / image.shape[1], height / image.shape[0])

def extract_words_from_connected_components(image, output_folder):
    """
    Extract words using connected components to group pixels.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Find connected components in the image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    bounding_boxes = []
    word_count = 0

    for i in range(1, num_labels):  # Skip the background component (label 0)
        x, y, w, h, area = stats[i]

        # Ignore small components that are likely noise
        if w > 30 and h > 20:  # Thresholds can be adjusted based on the expected text size
            word_count += 1
            bounding_boxes.append((x, y, w, h))

            # Extract and save the word image
            margin = 5
            word_image = image[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
            cv2.imwrite(os.path.join(output_folder, f"word_{word_count}.png"), word_image)

    return bounding_boxes

def draw_bounding_boxes(original_image_path, bounding_boxes, output_path, scale_factors):
    """
    Draw bounding boxes around detected words on the original image.
    """
    original_image = cv2.imread(original_image_path)

    if original_image is None:
        raise ValueError(f"Cannot load original image: {original_image_path}")

    scale_x, scale_y = scale_factors

    for (x, y, w, h) in bounding_boxes:
        # Scale bounding box coordinates back to match original image size
        x = int(x / scale_x)
        y = int(y / scale_y)
        w = int(w / scale_x)
        h = int(h / scale_y)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, original_image)

if __name__ == "__main__":
    # File paths
    input_image_path = "hwr_example.jpg"
    output_folder = "image"
    output_image_path = "annotated_image.jpg"

    # Preprocess the image
    processed_image, scale_factors = preprocess_image(input_image_path)

    # Extract words using connected components
    bounding_boxes = extract_words_from_connected_components(processed_image, output_folder)

    # Annotate original image
    draw_bounding_boxes(input_image_path, bounding_boxes, output_image_path, scale_factors)

    print(f"Words have been extracted and saved in {output_folder}.")
    print("Intermediate steps have been saved as images.")
    print(f"The annotated image has been saved as {output_image_path}.")