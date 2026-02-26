import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pipeline_utils.preprocessing import crop_frame, apply_clahe, prepare_bgr_to_tensor, rotate_frame

class VideoPairDataset(Dataset):
    """
    A PyTorch Dataset that yields (image1, image2, index) pairs for optical flow processing.
    It reads frames directly from a video file and applies optional rotation, cropping + CLAHE.
    """
    def __init__(self, video_path, roi=(0, -1, 0, -1), use_clahe=False, clahe_clip_limit=2.0, clahe_tile_size=8, max_frames=-1,
                 rotate_angle=0.0, rotate_center=None):
        self.video_path = video_path
        self.roi = roi
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.max_frames = max_frames
        self.rotate_angle = rotate_angle
        self.rotate_center = rotate_center  # (y, x) or None
        
        self.is_cine = self.video_path.lower().endswith('.cine')
        
        if self.is_cine:
            import pims
            with pims.open(self.video_path) as v:
                self.frame_count = len(v)
        else:
            # We need a dedicated Capture object per worker, but we only query this one for length
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open {self.video_path}")
                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        # Determine valid pairs (N-1 pairs)
        if self.max_frames > 0:
            self.frame_count = min(self.frame_count, self.max_frames)
            
        self.length = max(0, self.frame_count - 1)

    def __len__(self):
        return self.length

    def preprocess(self, frame_raw):
        is_grayscale = (len(frame_raw.shape) == 2)
        
        # For grayscale 12-bit .cine: scale to 8-bit before rotation
        if is_grayscale and frame_raw.dtype == np.uint16:
            frame_raw = np.clip((frame_raw.astype(np.float32) / 4095.0) * 255.0, 0, 255).astype(np.uint8)
        
        # Rotate BEFORE cropping so ROI operates on aligned image
        if self.rotate_angle != 0.0 and self.rotate_center is not None:
            frame_raw = rotate_frame(frame_raw, self.rotate_center, self.rotate_angle)
        
        frame = crop_frame(frame_raw, self.roi)
        
        if self.use_clahe:
            frame = apply_clahe(frame, self.clahe_clip_limit, self.clahe_tile_size, is_grayscale=is_grayscale)
            
        return prepare_bgr_to_tensor(frame, is_grayscale=is_grayscale)

    def __getitem__(self, idx):
        """
        Warning: For `num_workers > 0`, opening cv2.VideoCapture inside `__getitem__` safely isolates instances.
        If speed becomes a random access bottleneck, we will need to load the video entirely into RAM 
        or use advanced FFmpeg sequencers.
        """
        if self.is_cine:
            import pims
            with pims.open(self.video_path) as v:
                if idx + 1 >= len(v):
                    raise IndexError(f"Failed to read frame pair at index {idx} - End of CINE.")
                # Native pims parsing to dense numpy arrays
                frame1_raw = v[idx]
                frame2_raw = v[idx + 1]
        else:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx) # Seek to Frame t
            
            ret1, frame1_raw = cap.read()
            ret2, frame2_raw = cap.read() # Frame t+1
            cap.release()
    
            if not ret1 or not ret2:
                raise IndexError(f"Failed to read frame pair at index {idx}")

        image1 = self.preprocess(frame1_raw)
        image2 = self.preprocess(frame2_raw)

        return image1, image2, idx

