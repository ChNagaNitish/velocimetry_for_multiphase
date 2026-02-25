import cv2
import numpy as np
import torch

from .base import BaseOpticalFlowModel
from pipeline_utils.uncertainty import compute_uncertainty_batch

class FarnebackOpticalFlow(BaseOpticalFlowModel):
    def __init__(self, **kwargs):
        # Farneback natively runs on CPU via OpenCV Python bindings.
        super().__init__(device='cpu')

    def predict_batch(self, image1_batch, image2_batch):
        """
        Runs Farneback optical flow sequentially over the batch.
        
        Args:
            image1_batch: PyTorch tensor (B, 3, H, W)
            image2_batch: PyTorch tensor (B, 3, H, W)
            
        Returns:
            numpy.ndarray of shape (B, H, W, 2)
        """
        # Convert PyTorch Tensors (B, 3, H, W) back to numpy (B, H, W, 3) 
        # and from RGB to Grayscale for cv2.calcOpticalFlowFarneback
        batch_size, _, h, w = image1_batch.shape
        flow_out = np.zeros((batch_size, h, w, 2), dtype=np.float32)
        
        image1_np = image1_batch.permute(0, 2, 3, 1).numpy()
        image2_np = image2_batch.permute(0, 2, 3, 1).numpy()
        
        for b in range(batch_size):
            # Convert RGB to Grayscale (Note: our tensor was [0, 255] float)
            gray1 = cv2.cvtColor(image1_np[b].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2_np[b].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow_out[b] = flow
            
        # Compute uncertainty on the CPU
        flow_tensor = torch.from_numpy(flow_out).permute(0, 3, 1, 2)
        uncert_out = compute_uncertainty_batch(image1_batch, image2_batch, flow_tensor, window_size=32)
            
        return flow_out, uncert_out
