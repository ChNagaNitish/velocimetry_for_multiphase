import cv2
import numpy as np
import scipy.signal
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, max_frames=None, skip=1):
    """
    Extracts frames from a video file and saves them as images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames is not None and saved >= max_frames):
            break
            
        if count % skip == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved:05d}.png"), frame)
            saved += 1
            
        count += 1
        
    cap.release()
    return saved

def change_framerate(video_path, output_path, keep_every_n=2, new_fps=30):
    """
    Reduces the temporal resolution of a video by keeping only every Nth frame.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    count = 0
    for _ in tqdm(range(total_frames), desc="Changing Framerate"):
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % keep_every_n == 0:
            out.write(frame)
        count += 1
        
    cap.release()
    out.release()

def compute_shedding_frequency(video_path, roi=None, fps=130000):
    """
    Computes the dominant shedding frequency by analyzing brightness fluctuations
    using Welch's method for Power Spectral Density estimation.
    """
    cap = cv2.VideoCapture(video_path)
    brightness_over_time = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi is not None:
            y1, y2, x1, x2 = roi
            y_slice = slice(y1, y2 if y2 != -1 else None)
            x_slice = slice(x1, x2 if x2 != -1 else None)
            gray = gray[y_slice, x_slice]
            
        brightness_over_time.append(np.mean(gray))
        
    cap.release()
    
    signal = np.array(brightness_over_time)
    signal = signal - np.mean(signal) # Remove DC offset
    
    f, Pxx = scipy.signal.welch(signal, fs=fps, nperseg=1024)
    peak_idx = np.argmax(Pxx)
    dominant_freq = f[peak_idx]
    
    return signal, f, Pxx, dominant_freq

def compute_cavity_metrics(video_path):
    """
    Computes the time-averaged brightness and variance of a video to 
    determine the maximum cavity length bounds.
    Based on the two-pass iterative variance algorithm.
    """
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    avg = np.zeros((height, width), dtype=np.float32)
    
    # Pass 1: Combine means
    for _ in tqdm(range(frameCount), desc="Calculating Cavity Average"):
        ret, frame = cap.read()
        if not ret: break
        avg += cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    avg /= frameCount
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Pass 2: Combine variance
    var = np.zeros((height, width), dtype=np.float32)
    for _ in tqdm(range(frameCount), desc="Calculating Cavity Variance"):
        ret, frame = cap.read()
        if not ret: break
        var += (avg - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))**2
        
    var /= frameCount
    cap.release()
    
    limit = 100
    if var.shape[1] > limit:
        # Find maximum variance pixel after 'limit' x-coordinate
        max_indx = np.unravel_index(np.argmax(var[:, limit:]), var[:, limit:].shape)
        max_indx = (max_indx[0], max_indx[1] + limit)
        
        mag = avg[max_indx]
        cavity_mask = avg > mag
        
        return avg, var, cavity_mask, max_indx
    else:
        return avg, var, None, None
