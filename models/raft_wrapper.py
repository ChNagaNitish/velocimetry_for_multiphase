import torch
import os
import sys
import numpy as np

# Add local core directory so we can load RAFT
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, '..', 'core')
sys.path.append(core_path)

try:
    from raft import RAFT
except ImportError:
    print("Warning: RAFT module not found. Check path dependencies.")

from .base import BaseOpticalFlowModel
from pipeline_utils.uncertainty import compute_uncertainty_batch

class RAFTOpticalFlow(BaseOpticalFlowModel):
    def __init__(self, model_path, device='cuda', small=False, mixed_precision=False,
                 alternate_corr=False, **kwargs):
        super().__init__(device=device)
        self.device = device
        
        # Build argparse namespace required by original RAFT constructor
        class Args:
            def __contains__(self, item):
                return hasattr(self, item)
            
            def __iter__(self):
                return iter(vars(self).keys())
                
        args = Args()
        args.small = small
        args.mixed_precision = mixed_precision
        args.alternate_corr = alternate_corr
        
        # Load the PyTorch Module via DataParallel (Natives the 'module.' prefix)
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        
        self.model.to(self.device)
        self.model.eval()

    def _compute_pad_params(self, ht, wd):
        """Compute padding to make dimensions multiples of 8."""
        pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
        pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
        return [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def predict_batch(self, image1_batch, image2_batch):
        """
        Args:
            image1_batch: (B, 3, H, W) on CPU
            image2_batch: (B, 3, H, W) on CPU
            
        Returns: 
            (flow_batch, uncert_batch) as numpy arrays (B, H, W, 2)
        """
        ht, wd = image1_batch.shape[-2:]
        pad_params = self._compute_pad_params(ht, wd)
        
        b_image1 = image1_batch.to(self.device)
        b_image2 = image2_batch.to(self.device)
        
        b_image1_pad = torch.nn.functional.pad(b_image1, pad_params, mode='replicate')
        b_image2_pad = torch.nn.functional.pad(b_image2, pad_params, mode='replicate')
        
        with torch.no_grad():
            _, flow_up_pad = self.model(b_image1_pad, b_image2_pad, iters=20, test_mode=True)
            
            # Unpad back to original dims
            c = [pad_params[2], flow_up_pad.shape[-2] - pad_params[3],
                 pad_params[0], flow_up_pad.shape[-1] - pad_params[1]]
            flow_up = flow_up_pad[..., c[0]:c[1], c[2]:c[3]]
            
            # Compute dense uncertainty directly on the GPU using the batched inputs and flow
            uncert_out = compute_uncertainty_batch(b_image1, b_image2, flow_up, window_size=32)
            
        # (B, 2, H, W) -> (B, H, W, 2)
        flow_up = flow_up.permute(0, 2, 3, 1).cpu().numpy()
        return flow_up, uncert_out
