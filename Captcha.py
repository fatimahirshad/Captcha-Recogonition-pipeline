# ============ IMAGES RECONSTRUCTION===============
import numpy as np
from PIL import Image
import os

base_path = '/content/drive/MyDrive/captcha_project'
txt_dir = os.path.join(base_path, '/content/drive/MyDrive/captcha_project/input')
output_dir = os.path.join(base_path, '/content/drive/MyDrive/captcha_project/output')
os.makedirs(output_dir, exist_ok=True)

def txt_to_image(txt_path, save_path=None):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    height, width = map(int, lines[0].split())
    pixels = []
    for line in lines[1:]:
        row = [tuple(map(int, pixel.split(','))) for pixel in line.strip().split()]
        pixels.extend(row)

    img_array = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
    img = Image.fromarray(img_array)

    if save_path:
        img.save(save_path)
    return img

# Loop through all .txt files and convert
for filename in os.listdir(txt_dir):
    if filename.endswith('.txt'):
        txt_path = os.path.join(txt_dir, filename)
        save_path = os.path.join(output_dir, filename.replace('.txt', '.jpg'))
        txt_to_image(txt_path, save_path)

print("✅ All .txt files converted to .jpg.")




#================== CHARACTER SEGMENTATION AND LABELING =============

import os
import cv2
import numpy as np
from PIL import Image

# Step 1: Paths
base_path = '/content/drive/MyDrive/captcha_project'
input_folder = os.path.join(base_path, '/content/drive/MyDrive/captcha_project/input')
reconstructed_folder = os.path.join(base_path, '/content/drive/MyDrive/captcha_project/reconstructed_imgs')
character_output_folder = os.path.join(base_path, '/content/drive/MyDrive/captcha_project/characters')
os.makedirs(reconstructed_folder, exist_ok=True)
os.makedirs(character_output_folder, exist_ok=True)

# Step 2: Your manual image-label mapping
image_labels = {
    'input00.jpg': 'EGYK4',
    'input01.jpg': 'GRC35',
    'input02.jpg': '605W1',
    'input03.jpg': 'J627C',
    'input04.jpg': 'VLI2C',
    'input05.jpg': 'O1R7Q',
    'input06.jpg': '0YTAD',
    'input07.jpg': 'ZRMQU',
    'input08.jpg': 'N9DQS',
    'input09.jpg': 'ZGJS3',
    'input10.jpg': 'GZMBA',
    'input11.jpg': 'I14DM',
    'input12.jpg': 'PQ9AE',
    'input13.jpg': 'VWZD0',
    'input14.jpg': 'WGST7',
    'input15.jpg': 'XKMS2',
    'input16.jpg': '1D2KB',
    'input17.jpg': '20BHQ',
    'input18.jpg': 'OAH0V',
    'input19.jpg': '5I8VE',
    'input20.jpg': 'Z97ME',
    'input21.jpg': 'CL69V',
    'input22.jpg': 'HCE91',
    'input23.jpg': 'WELXV',
    'input24.jpg': 'UHVFO'
}

# Step 3: Convert .txt to image
def txt_to_image(txt_path, save_path=None):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    height, width = map(int, lines[0].split())
    pixels = []
    for line in lines[1:]:
        row = [tuple(map(int, pixel.split(','))) for pixel in line.strip().split()]
        pixels.extend(row)
    img_array = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
    img = Image.fromarray(img_array)
    if save_path:
        img.save(save_path)
    return img

# Step 4: Preprocess & segment
def preprocess_image(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def segment_characters(thresh_img):
    h, w = thresh_img.shape
    char_width = w // 5
    return [thresh_img[:, i*char_width:(i+1)*char_width] for i in range(5)]

# Step 5: Loop through all txt files
for file_name in image_labels.keys():
    txt_name = file_name.replace('.jpg', '.txt')
    txt_path = os.path.join(input_folder, txt_name)
    recon_img_path = os.path.join(reconstructed_folder, file_name)

    # Step 5.1: Convert to image
    img = txt_to_image(txt_path, recon_img_path)

    # Step 5.2: Segment
    thresh = preprocess_image(img)
    chars = segment_characters(thresh)
    label = image_labels[file_name]

    # Step 5.3: Save each char with its label
    for i, char_img in enumerate(chars):
        char_label = label[i]
        char_folder = os.path.join(character_output_folder, char_label)
        os.makedirs(char_folder, exist_ok=True)
        char_file = f"{file_name.replace('.jpg', '')}_{i}.png"
        Image.fromarray(char_img).save(os.path.join(char_folder, char_file))

print("✅ All characters segmented and labeled successfully.")


import os
import numpy as np
from PIL import Image, ImageEnhance
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- 1 AUGUMENTATION FUNCTION  ---

def augment_image(img):
    # Random rotation
    angle = random.uniform(-10, 10)
    img = img.rotate(angle)

    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    return img

# --- 2. Load + Augment Dataset ---
data = []
labels = []
char_folder = '/content/drive/MyDrive/captcha_project/characters'

for label in sorted(os.listdir(char_folder)):
    label_path = os.path.join(char_folder, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith('.png'):
                img_path = os.path.join(label_path, file)
                img = Image.open(img_path).convert('L')
                img = img.resize((20, 20))
                arr = np.array(img)

                # Add original
                data.append(arr)
                labels.append(label)

                # Add 5 augmentations
                for _ in range(5):
                    aug = augment_image(img)
                    data.append(np.array(aug))
                    labels.append(label)

# --- 3 PREPARE DATA  ---

X = np.array(data).reshape(-1, 20, 20, 1) / 255.0
y = np.array(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=36)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# --- 4 CNN MODEL--------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(20, 20, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(36, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5 TRAIN -----------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# --- 6 Save Model & Encoder ---
model.save('/content/drive/MyDrive/captcha_project/cnn_model.h5')

import pickle
with open('/content/drive/MyDrive/captcha_project/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Training complete and model saved.")


#========Final Inference Class:========

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import pickle

class Captcha(object):
    def __init__(self, model_path, encoder_path):
        # Load trained CNN model
        self.model = tf.keras.models.load_model(model_path)

        # Load label encoder (for decoding predicted class index to char)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def preprocess_char(self, char_img):
        # Convert to grayscale, resize to 20x20, normalize
        img = Image.fromarray(char_img).convert('L').resize((20, 20))
        arr = np.array(img).reshape(1, 20, 20, 1) / 255.0
        return arr

    def segment_image(self, image_path):
        # Open image and preprocess
        img = Image.open(image_path)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        h, w = thresh.shape
        char_width = w // 5

        # Cut image into 5 equal-width slices
        chars = [thresh[:, i*char_width:(i+1)*char_width] for i in range(5)]
        return chars

    def __call__(self, im_path, save_path=None):
        chars = self.segment_image(im_path)
        predicted = ""

        for char in chars:
            x = self.preprocess_char(char)
            y_pred = self.model.predict(x, verbose=0)
            label = self.encoder.inverse_transform([np.argmax(y_pred)])
            predicted += label[0]

        if save_path:
            with open(save_path, 'w') as f:
                f.write(predicted)

        print(f"✅ Predicted CAPTCHA: {predicted}")
        return predicted



# =============TEST WITH UNSEEN IMAGES=============
# Create the model
cap = Captcha(
    model_path='/content/drive/MyDrive/captcha_project/cnn_model.h5',
    encoder_path='/content/drive/MyDrive/captcha_project/label_encoder.pkl'
)

# Predict from a new image (input25.jpg should be a 5-char captcha)
cap('/content/drive/MyDrive/captcha_project/input/input07.jpg')


#============= BATCH TESTING ====================
import os

# Your ground-truth labels
image_labels = {
    'input00.jpg': 'EGYK4',
    'input01.jpg': 'GRC35',
    'input02.jpg': '605W1',
    'input03.jpg': 'J627C',
    'input04.jpg': 'VLI2C',
    'input05.jpg': 'O1R7Q',
    'input06.jpg': '0YTAD',
    'input07.jpg': 'ZRMQU',
    'input08.jpg': 'N9DQS',
    'input09.jpg': 'ZGJS3',
    'input10.jpg': 'GZMBA',
    'input11.jpg': 'I14DM',
    'input12.jpg': 'PQ9AE',
    'input13.jpg': 'VWZD0',
    'input14.jpg': 'WGST7',
    'input15.jpg': 'XKMS2',
    'input16.jpg': '1D2KB',
    'input17.jpg': '20BHQ',
    'input18.jpg': 'OAH0V',
    'input19.jpg': '5I8VE',
    'input20.jpg': 'Z97ME',
    'input21.jpg': 'CL69V',
    'input22.jpg': 'HCE91',
    'input23.jpg': 'WELXV',
    'input24.jpg': 'UHVFO'
}

# Path to the captcha images
captcha_folder = '/content/drive/MyDrive/captcha_project/reconstructed_imgs'

# Initialize model
cap = Captcha(
    model_path='/content/drive/MyDrive/captcha_project/cnn_model.h5',
    encoder_path='/content/drive/MyDrive/captcha_project/label_encoder.pkl'
)

# Tracking
total = 0
correct = 0

print("🧪 Batch Testing Results:\n")

for filename, true_label in image_labels.items():
    img_path = os.path.join(captcha_folder, filename)
    predicted = cap(img_path)

    # Compare and highlight
    comparison = ""
    match_count = 0
    for p, t in zip(predicted, true_label):
        if p == t:
            comparison += f"\033[92m{p}\033[0m"  # green for correct
            match_count += 1
        else:
            comparison += f"\033[91m{p}\033[0m"  # red for wrong

    total += 5
    correct += match_count

    print(f"{filename}: Predicted: {comparison}  |  Actual: {true_label}")

# Overall accuracy
accuracy = (correct / total) * 100
print(f"\n✅ Overall character-level accuracy: {accuracy:.2f}%")
