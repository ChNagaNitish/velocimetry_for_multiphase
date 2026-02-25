import cv2
import numpy as np
import torch

def crop_frame(frame, roi):
    """
    Crops a frame using the given ROI tuple (y_start, y_end, x_start, x_end).
    """
    y_start, y_end, x_start, x_end = roi
    y_slice = slice(y_start, y_end if y_end != -1 else None)
    x_slice = slice(x_start, x_end if x_end != -1 else None)
    return frame[y_slice, x_slice]

def apply_clahe(frame, clip_limit=2.0, tile_size=8, is_grayscale=False):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    Directly processes grayscale 16-bit/8-bit arrays if flagged, else processes L-channel of BGR.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if is_grayscale:
        return clahe.apply(frame)
    else:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def prepare_bgr_to_tensor(frame, is_grayscale=False):
    """
    Converts numpy frame to FloatTensor format (3, H, W).
    Handles upscaling 1-channel grayscale directly to 3-channel RGB.
    Normalizes 12/16-bit raw `.cine` arrays cleanly into the 0.0-255.0 bounds expected by optical flow models.
    """
    if is_grayscale:
        frame_f = frame.astype(np.float32)
        if frame.dtype == np.uint16:
            # Phantom Cines generally store 12-bit data in uint16 containers.
            # Scale exactly down to the matching 8-bit density float values.
            frame_f = (frame_f / 4095.0) * 255.0
            frame_f = np.clip(frame_f, 0, 255)
            
        img = np.stack([frame_f]*3, axis=-1)
        return torch.from_numpy(img).permute(2, 0, 1)
    else:
        img = frame[..., ::-1].copy()
        return torch.from_numpy(img).permute(2, 0, 1).float()
