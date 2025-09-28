## CAPTCHA Recognition System

This project implements a complete pipeline for recognizing 5-character CAPTCHAs using a Convolutional Neural Network (CNN). The system is designed for a specific type of CAPTCHA where the images have a consistent font, spacing, character count (5), and background/foreground textures, varying only in the actual characters displayed.

The pipeline covers **data reconstruction**, **character segmentation and labeling**, **CNN model training** with data augmentation, and **final inference**.

***

## 1. Project Description

The target CAPTCHA system generates strictly **5-character captchas**, where each character is either an **upper-case letter (A-Z)** or a **numeral (0-9)**. A key vulnerability is the **consistency** of the CAPTCHA style: same font, spacing, and similar color/texture characteristics, which simplifies the task by allowing for fixed-width segmentation.

The initial dataset is provided in a raw `.txt` format, which must first be reconstructed into image files.

***

## 2. Usage Pipeline

The CAPTCHA recognition process is broken down into three main, sequential steps.

### Step 3.1: Images Reconstruction

The initial script handles the conversion of raw `.txt` files (which store image pixel data) into standard `.jpg` image files.

* **Input:** `.txt` files in the `input/` directory. Each file contains image dimensions on the first line, followed by RGB pixel values.
* **Output:** `.jpg` files saved to the `output/` directory.

### Step 2.2: Character Segmentation and Labeling (Training Data Prep)

This script is crucial for preparing the training dataset.

1.  **Reconstruction:** Re-runs the `txt_to_image` function to ensure images are available.
2.  **Preprocessing:** Converts the image to grayscale and applies a binary inverse threshold to isolate the foreground characters.
3.  **Segmentation:** Uses a fixed-width slicing method to segment the 5-character image into 5 individual character images.
4.  **Labeling:** Uses a hardcoded `image_labels` dictionary to assign the correct character label to each segmented image slice.
5.  **Saving:** Saves each segmented character into its corresponding class folder within the `characters/` directory (e.g., a 'G' from a captcha goes into `characters/G/`).

### Step 2.3: Model Training and Saving

This section loads the segmented characters, augments the dataset, trains the CNN model, and saves the trained model and label encoder.

1.  **Load & Augment:** Iterates through the `characters/` folders, loads each image, and applies 5 random **augmentations** (rotation, brightness, contrast) to artificially expand the dataset.
2.  **Data Preparation:**
    * Reshapes the image data to `(20, 20, 1)` and normalizes pixel values.
    * Uses `LabelEncoder` to convert character labels (e.g., 'A', '1') to numerical indices, and then `to_categorical` for one-hot encoding.
3.  **Model Definition:** Defines a simple **Convolutional Neural Network (CNN)** with two `Conv2D` and `MaxPooling2D` layers, followed by a `Dense` classifier with a 36-class output (A-Z, 0-9).
4.  **Training:** Trains the model using an `Adam` optimizer and `EarlyStopping` callback for regularization.
5.  **Saving:** The trained model is saved as **`cnn_model.h5`** and the `LabelEncoder` is saved as **`label_encoder.pkl`**.

***

## 3. Inference (Making Predictions)

The `Captcha` class provides a final, reusable module for recognizing unseen CAPTCHAs.

### Class: `Captcha`

The class initializes by loading the saved model and label encoder. The main method is the `__call__` method, which takes an image path:

1.  **`segment_image(image_path)`:** Loads, preprocesses (grayscale, threshold), and segments the input image into 5 character segments.
2.  **Prediction Loop:**
    * Iterates over the 5 segmented characters.
    * **`preprocess_char()`:** Resizes the character segment to the model input size (20x20) and normalizes it.
    * Predicts the character using the loaded CNN model.
    * Uses the `LabelEncoder` to convert the predicted index back into the character label (e.g., 'A').
3.  **Output:** Concatenates the 5 predictions into a single string and prints/returns the result.

### Testing

The final part of the script demonstrates the use of the `Captcha` class:

* **Single Test:** Predicts the CAPTCHA from a single specified image.
* **Batch Testing:** Runs the recognition on the original labeled dataset (`image_labels`) to calculate the overall **character-level accuracy**. The output uses colored text to highlight correct (green) and incorrect (red) predictions.
