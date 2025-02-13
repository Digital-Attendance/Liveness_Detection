import os
import numpy as np
import cv2
from src.generate_patches import CropImage
from src.anti_spoof_predict import AntiSpoofPredict
from src.utility import parse_model_name

MODEL_DIR = "./resources/anti_spoof_models"
DEVICE_ID = 1

# Preload heavy resources once at module load time
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found.")

# Create a single instance of the predictor and cropper to reuse across invocations
model_test = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()

# List model files once and reuse the list on each call
MODEL_FILES = os.listdir(MODEL_DIR)
if not MODEL_FILES:
    raise ValueError("No model files found in the specified directory.")


def detect(image, model_dir=MODEL_DIR):
    try:
        # Get bounding box once per image
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        for model_name in MODEL_FILES:
            h_input, w_input, _, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": scale is not None,
            }
            img = image_cropper.crop(**param)
            model_path = os.path.join(model_dir, model_name)
            prediction += model_test.predict(img, model_path)
        label = int(np.argmax(prediction))
        value = float(prediction[0][label] / 2)
        if label == 2:
            label = 0
        return label, value
    except Exception as e:
        raise Exception(f"Error in detect function: {e}")
