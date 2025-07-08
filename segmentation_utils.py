import requests
import numpy as np
from PIL import Image
import io

def segment_clothes(image: Image.Image):
    """
    ส่งภาพไปยัง HuggingFace Spaces Human Parsing API (LIP) เพื่อแยกส่วนเสื้อ/กางเกง
    คืน mask (np.ndarray) ที่ label: 5=upper-clothes, 6=pants
    """
    API_URL = "https://hf.space/embed/akhaliq/LIP_Human_Parsing/api/predict/"
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                mask_bytes = bytes(result["data"][0]["mask"]["data"])
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask = np.array(mask_img)
                return mask
        return None
    except Exception as e:
        return None

def extract_part(image: Image.Image, mask: np.ndarray, part_labels):
    """
    คืนภาพเฉพาะส่วนที่ label อยู่ใน part_labels (list of int)
    """
    arr = np.array(image)
    part_mask = np.isin(mask, part_labels)
    if arr.shape[-1] == 3:
        arr_rgba = np.concatenate([arr, np.full(arr.shape[:2]+(1,), 255, dtype=np.uint8)], axis=-1)
    else:
        arr_rgba = arr.copy()
    arr_rgba[...,3] = np.where(part_mask, 255, 0)
    return Image.fromarray(arr_rgba, mode="RGBA")
