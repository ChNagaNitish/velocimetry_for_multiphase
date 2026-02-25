from abc import ABC, abstractmethod

class BaseOpticalFlowModel(ABC):
    """
    Abstract base interface for all dense optical flow models.
    """
    
    @abstractmethod
    def __init__(self, device='cpu', **kwargs):
        self.device = device
        
    @abstractmethod
    def predict_batch(self, image1_batch, image2_batch):
        """
        Receives a batch of images and returns optical flow maps.
        
        Args:
            image1_batch: PyTorch tensor (B, 3, H, W)
            image2_batch: PyTorch tensor (B, 3, H, W)
            
        Returns:
            Tuple of (flow_batch, uncertainty_batch) 
            where each is a numpy.ndarray of shape (B, H, W, 2)
        """
        pass
