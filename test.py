import os
import numpy as np
from src.generate_patches import CropImage
from src.anti_spoof_predict import AntiSpoofPredict
from src.utility import parse_model_name


def check_image(image):
    height, width, _ = image.shape
    return width / height == 3 / 4


def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    if not check_image(image):
        return None, None

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))

    for model_name in os.listdir(model_dir):
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
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return label, value


def islive(image, model_dir="./resources/anti_spoof_models", device_id=1):
    return test(image, model_dir, device_id)
