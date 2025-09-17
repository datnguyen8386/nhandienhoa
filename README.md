# nhandienhoa
# ===== Full Colab cell: Train grayscale (60x60) + Gradio UI =====

# 1) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2) Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import math
import random
import gradio as gr

# 3) Config (đã dùng path của bạn)
data_dir = "/content/drive/MyDrive/hoa"   # <-- giữ nguyên đường dẫn
img_size = (60, 60)   # resize về 60x60 grayscale
batch_size = 16
epochs = 15

# 4) Load dataset (grayscale)
# Add a function to print the path of files being loaded for debugging
def debug_image_dataset_from_directory(*args, **kwargs):
    ds = tf.keras.utils.image_dataset_from_directory(*args, **kwargs)
    def _map_fn(image, label):
        tf.print("Loading file:", tf.strings.reduce_join(tf.strings.bytes_split(image.numpy())),'') # Use tf.strings for path
        return image, label
    # Not mapping here directly, but keep the original behavior. This is just for printing.
    # The actual file loading happens within image_dataset_from_directory
    return ds

# Use the original tf.keras.utils.image_dataset_from_directory for actual loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'
)


# 5) Get class names BEFORE mapping
class_names = train_ds.class_names
print("Danh sách lớp:", class_names)
idx_to_class = {i: name for i, name in enumerate(class_names)}

# 6) Preprocess: scale to [0,1] and one-hot encode labels (for categorical_crossentropy)
def preprocess(image, label):
    image = image / 255.0
    num_classes = len(class_names)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# 7) Build CNN (input shape 60x60x1)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 8) Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 9) Plot (optional)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend(); plt.title('Loss')
plt.show()

# 10) Save model to Drive for reuse
save_path = "/content/drive/MyDrive/hoa_model_gray.keras"
model.save(save_path)
print("Model saved to:", save_path)

# 11) Helper functions for testing / display
def predict_random_images(base_dir, total_images=60):
    img_files = []
    if not os.path.isdir(base_dir):
        print(f"⚠️ Thư mục không tồn tại: {base_dir}")
        return

    for subfolder in os.listdir(base_dir):
        subpath = os.path.join(base_dir, subfolder)
        if os.path.isdir(subpath):
            #print(f"Checking subfolder: {subpath}") # Debugging print
            for f in os.listdir(subpath):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_files.append(os.path.join(subpath, f))
                #else:
                    #print(f"Skipping non-image file: {f} (extension: {f.split('.')[-1] if '.' in f else 'none'})") # Debugging print

    if not img_files:
        print("⚠️ Không tìm thấy ảnh trong thư mục con của:", base_dir)
        return

    random.shuffle(img_files)
    img_files = img_files[:total_images]
    n_images = len(img_files)
    if n_images == 0:
        print("⚠️ Không tìm thấy ảnh trong thư mục:", base_dir)
        return

    cols = 10
    rows = math.ceil(n_images / cols)
    plt.figure(figsize=(20, 2*rows))

    for i, img_path in enumerate(img_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Lỗi load ảnh: {img_path}")
            continue
        img_resized = cv2.resize(img, img_size)
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        pred = model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(pred)
        class_name = idx_to_class[class_idx]
        confidence = float(np.max(pred))

        title = f"{class_name} ({confidence*100:.1f}%)"
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_resized, cmap='gray')
        plt.title(title, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def predict_folder(folder_path, max_images=60):
    img_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(os.path.join(root, f))
    if not img_files:
        print("⚠️ Không có ảnh trong:", folder_path)
        return
    img_files = img_files[:max_images]
    for p in img_files:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("⚠️ Không load được:", p)
            continue
        img_resized = cv2.resize(img, img_size)
        arr = img_resized.astype('float32') / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))
        pred = model.predict(arr, verbose=0)[0]
        class_idx = np.argmax(pred)
        class_name = idx_to_class[class_idx]
        conf = float(np.max(pred))
        print(f"{os.path.basename(p)} -> {class_name} (conf={conf:.3f})")

# 12) Run quick demo prints (optional) — comment out if not needed
base_folder = data_dir
predict_random_images(base_folder, total_images=30)
# predict_folder(data_dir, max_images=30)

# 13) Gradio predict (upload from local machine)
from PIL import Image
def predict_gradio(pil_img):
    if pil_img is None:
        return {}, None
    # convert to grayscale, resize, normalize
    img_gray = pil_img.convert("L").resize(img_size)
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1,H,W,1)

    preds = model.predict(arr, verbose=0)[0]
    # ensure numerical stability
    if preds.sum() != 0:
        preds = preds / preds.sum()
    out = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    # return (label-dict, image to show)
    return out, img_gray
