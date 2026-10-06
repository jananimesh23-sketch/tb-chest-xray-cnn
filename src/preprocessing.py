import cv2
import numpy as np

TARGET_SIZE = (224, 224)

def preprocess_for_inference(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file: unable to decode.")
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    expanded = np.expand_dims(normalized, axis=(0, -1))
    return expanded
