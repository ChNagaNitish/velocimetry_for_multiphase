import torch
import numpy as np
import cv2
from scipy.interpolate import griddata

from openpiv.windef import PIVSettings, first_pass, multipass_img_deform
from openpiv.validation import sig2noise_val, global_val, global_std

from .base import BaseOpticalFlowModel
from pipeline_utils.uncertainty import compute_uncertainty_batch

class OpenPIVModel(BaseOpticalFlowModel):
    def __init__(self, **kwargs):
        # OpenPIV natively runs on CPU.
        super().__init__(device='cpu')
        
        # Match pivLabrun.m settings
        self.settings = PIVSettings()
        self.settings.windowsizes = (64, 32, 16, 8)
        self.settings.overlap = (32, 16, 8, 4)
        self.settings.num_iterations = 4
        
        # Advanced configurations matching MATLAB script 's' and 'p' parameters
        self.settings.subpixel_method = 'gaussian'
        self.settings.deformation_method = 'symmetric'
        self.settings.interpolation_order = 1 # Linear to avoid small-roi Spline crashes
        self.settings.sig2noise_method = 'peak2mean'
        self.settings.sig2noise_threshold = 1.0 # Standard weak peak removal
        
        # Outlier removal options
        self.settings.replace_vectors = True
        self.settings.smoothn = False
        self.settings.filter_method = 'localmean'
        self.settings.max_filter_iteration = 4
        self.settings.filter_kernel_size = 2

    def process_pair(self, img1, img2):
        """
        Processes a single pair of grayscale images through OpenPIV.
        """
        # (CLAHE preprocessing is now globally handled natively through `data.py`)
        
        # Iteration 1
        x, y, u, v, s2n = first_pass(img1, img2, self.settings)
        
        # openpiv.windef.multipass_img_deform expects masked arrays for tracking validations over iterations
        u = np.ma.masked_array(u, np.isnan(u))
        v = np.ma.masked_array(v, np.isnan(v))
        
        # Iterations 2..N (Deformation)
        for i in range(1, self.settings.num_iterations):
            x, y, u, v, grid_mask, flags = multipass_img_deform(
                img1, img2, i, x, y, u, v, self.settings
            )
            
    def _process_single_frame(self, b, image1_np, image2_np, h, w):
        gray1 = cv2.cvtColor(image1_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        x_pts, y_pts, u_pts, v_pts = self.process_pair(gray1, gray2)
        
        points = np.column_stack((x_pts.flatten(), y_pts.flatten()))
        u_flat = u_pts.flatten()
        v_flat = v_pts.flatten()
        
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        
        u_dense = griddata(points, u_flat, (grid_x, grid_y), method='linear', fill_value=0)
        v_dense = griddata(points, v_flat, (grid_x, grid_y), method='linear', fill_value=0)
        
        return b, u_dense, v_dense
    def predict_batch(self, image1_batch, image2_batch):
        """
        Runs OpenPIV optical flow sequentially over the batch.
        
        Args:
            image1_batch: PyTorch tensor (B, 3, H, W) normalized [0, 255]
            image2_batch: PyTorch tensor (B, 3, H, W) normalized [0, 255]
            
        Returns:
            numpy.ndarray of dense flow shape (B, H, W, 2)
        """
        batch_size, _, h, w = image1_batch.shape
        flow_out = np.zeros((batch_size, h, w, 2), dtype=np.float32)
        
        # ----- HPC / SLURM Safe CPU Parallelization -----
        import concurrent.futures
        import os
        
        # os.cpu_count() returns the total physical node cores (e.g., 128),
        # which ignores SLURM's --cpus-per-task allocation and leads to oversubscription.
        # os.sched_getaffinity(0) returns the exact CPU cores allocated to this process/cgroup.
        if hasattr(os, 'sched_getaffinity'):
            available_cores = len(os.sched_getaffinity(0))
        else:
            available_cores = os.cpu_count() or 4
            
        workers = min(batch_size, available_cores)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._process_single_frame, b, image1_np[b], image2_np[b], h, w)
                for b in range(batch_size)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                b_idx, u_dense, v_dense = future.result()
                flow_out[b_idx, :, :, 0] = u_dense
                flow_out[b_idx, :, :, 1] = v_dense
            
        # Compute uncertainty on the CPU
        flow_tensor = torch.from_numpy(flow_out).permute(0, 3, 1, 2)
        uncert_out = compute_uncertainty_batch(image1_batch, image2_batch, flow_tensor, window_size=32)
            
        return flow_out, uncert_out
