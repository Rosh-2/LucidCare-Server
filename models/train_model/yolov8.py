import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2  # We need OpenCV for this step
import shutil # For copying files
from tqdm import tqdm # For a nice progress bar

# --- 1. HYPERPARAMETERS & SETUP ---

# --- MODEL/DATA PATHS ---
CLASSIFIER_MODEL_PATH = 'densenet121_classifier_4_class_gpu.h5'
SOURCE_DATA_DIR = 'C:\\Users\\rejir\\OneDrive\\Desktop\\DataSet\\chest_xray' # The 'data' folder with 'train' and 'val'
YOLO_DATASET_DIR = 'C:\\Users\\rejir\\OneDrive\\Desktop\\DataSet\\yolo_dataset' # NEW directory to be created

# --- GRAD-CAM & BBOX SETTINGS ---
IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = 'relu' # Last conv layer in DenseNet121
# This is the most important hyperparameter to tune.
# It controls how "hot" a pixel needs to be to be included in the box.
# (Range: 0-255). A higher value means a tighter box.
THRESHOLD_VALUE = 200 

# --- 2. LOAD MODEL & GET CLASS NAMES ---

print(f"Loading classifier model from {CLASSIFIER_MODEL_PATH}...")
model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)

# Get class names from the folder structure
CLASS_NAME_SOURCE_DIR = os.path.join(SOURCE_DATA_DIR, 'train')
CLASS_NAMES = sorted([d for d in os.listdir(CLASS_NAME_SOURCE_DIR) if os.path.isdir(os.path.join(CLASS_NAME_SOURCE_DIR, d))])

# CRITICAL: Map original class names to indices
# We need this to tell Grad-CAM which class to look for
ORIGINAL_CLASS_INDICES = {name: i for i, name in enumerate(CLASS_NAMES)}
print(f"Original classes found: {ORIGINAL_CLASS_INDICES}")

# CRITICAL: Create the YOLO class map
# We EXCLUDE "normal" and re-index the abnormal classes from 0
ABNORMAL_CLASS_NAMES = [name for name in CLASS_NAMES if name != 'normal']
YOLO_CLASS_MAP = {name: i for i, name in enumerate(ABNORMAL_CLASS_NAMES)}
print(f"YOLO classes will be: {YOLO_CLASS_MAP}")


# --- 3. GRAD-CAM FUNCTIONS (From Step 2) ---

def get_img_array(img_path, size):
    """Loads and preprocesses an image for model prediction."""
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = array / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """Generates the Grad-CAM heatmap for a specific class index."""
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 4. NEW FUNCTION: CONVERT HEATMAP TO BBOX ---

def convert_heatmap_to_yolo_bbox(heatmap, img_width, img_height, threshold_val):
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    _, binary_mask = cv2.threshold(heatmap_uint8, threshold_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

    # 6. Convert to normalized YOLO format
    x_center_norm = (x + w / 2) / img_width
    y_center_norm = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return (x_center_norm, y_center_norm, w_norm, h_norm)

# --- 5. MAIN SCRIPT: PROCESS ALL IMAGES ---

def create_yolo_dataset(source_split_dir, yolo_img_dir, yolo_label_dir):
    """
    Loops through a 'train' or 'val' directory, generates heatmaps
    for abnormal images, and saves them in YOLO format.
    """
    
    # Create the output directories
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_label_dir, exist_ok=True)

    # Loop over class folders ('normal', 'covid-19', etc.)
    for class_name in os.listdir(source_split_dir):
        class_dir = os.path.join(source_split_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue

        # --- THIS IS THE KEY: We SKIP the 'normal' class ---
        if class_name == 'normal':
            print(f"Skipping 'normal' class...")
            continue
        
        print(f"Processing class: {class_name}")
        
        # Get the class indices
        original_class_idx = ORIGINAL_CLASS_INDICES[class_name]
        yolo_class_idx = YOLO_CLASS_MAP[class_name]

        # Use tqdm for a progress bar for images in this class
        for img_filename in tqdm(os.listdir(class_dir), desc=f"  {class_name}"):
            img_path = os.path.join(class_dir, img_filename)
            
            # Define output paths
            output_img_path = os.path.join(yolo_img_dir, img_filename)
            output_label_path = os.path.join(yolo_label_dir, os.path.splitext(img_filename)[0] + '.txt')

            try:
                # 1. Generate heatmap
                img_array = get_img_array(img_path, IMG_SIZE)
                heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, original_class_idx)
                
                # 2. Convert heatmap to YOLO bounding box
                bbox = convert_heatmap_to_yolo_bbox(heatmap, IMG_SIZE[0], IMG_SIZE[1], THRESHOLD_VALUE)
                
                if bbox:
                    
                    x_center, y_center, w_norm, h_norm = bbox
                    yolo_label_string = f"{yolo_class_idx} {x_center} {y_center} {w_norm} {h_norm}"
                    
                    with open(output_label_path, 'w') as f:
                        f.write(yolo_label_string)
                        
                    # 5. Copy the original image
                    shutil.copy(img_path, output_img_path)

            except Exception as e:
                print(f"  [Warning] Failed to process {img_filename}: {e}")

# --- 6. RUN THE ENTIRE PROCESS ---

print("--- Starting YOLO Dataset Generation ---")

# Process the training set
print("\nProcessing TRAINING data...")
train_source_dir = os.path.join(SOURCE_DATA_DIR, 'train')
yolo_train_img_dir = os.path.join(YOLO_DATASET_DIR, 'images', 'train')
yolo_train_label_dir = os.path.join(YOLO_DATASET_DIR, 'labels', 'train')
create_yolo_dataset(train_source_dir, yolo_train_img_dir, yolo_train_label_dir)

# Process the validation set
print("\nProcessing VALIDATION data...")
val_source_dir = os.path.join(SOURCE_DATA_DIR, 'val')
yolo_val_img_dir = os.path.join(YOLO_DATASET_DIR, 'images', 'val')
yolo_val_label_dir = os.path.join(YOLO_DATASET_DIR, 'labels', 'val')
create_yolo_dataset(val_source_dir, yolo_val_img_dir, yolo_val_label_dir)

print("\n--- YOLO Dataset Generation Complete! ---")
print(f"Your new dataset is ready in: {YOLO_DATASET_DIR}")