import h5py
import numpy as np
import cv2

class HDF5Writer:
    """
    Handles window-averaging and chunked writing of velocity grids to an HDF5 file.
    """
    def __init__(self, output_path, frame_count, h, w, window_height, window_width, metadata):
        self.output_path = output_path
        self.window_height = window_height
        self.window_width = window_width
        
        self.pad_height = window_height // 2
        self.pad_width = window_width // 2
        self.kernel = np.ones([window_height, window_width]) / (window_height * window_width)
        
        data_shape = (frame_count - 1, h // window_height, w // window_width, 2)
        chunk_t = min(10, frame_count - 1)
        chunk_shape = (chunk_t, h // window_height, w // window_width, 2)
        
        self.file = h5py.File(output_path, 'w')
        self.velData = self.file.create_dataset('velocity', shape=data_shape, chunks=chunk_shape, dtype='float32')
        self.uncertData = self.file.create_dataset('uncertainty', shape=data_shape, chunks=chunk_shape, dtype='float32')
        
        # Save attributes
        for key, value in metadata.items():
            self.velData.attrs[key] = value

    def write_batch(self, start_idx, flow_batch, uncert_batch):
        """
        Takes a batch of flow arrays and uncertainty arrays (B, H, W, 2) from the GPU, applies averaging if requested,
        and saves them to the HDF5 file.
        """
        B, H, W, C = flow_batch.shape
        
        if self.window_width > 1 or self.window_height > 1:
            for i in range(B):
                flow = flow_batch[i]
                uncert = uncert_batch[i]
                
                padded_arr = np.pad(flow, ((self.pad_height, self.pad_height), (self.pad_width, self.pad_width), (0, 0)), mode='reflect')
                padded_uncert = np.pad(uncert, ((self.pad_height, self.pad_height), (self.pad_width, self.pad_width), (0, 0)), mode='reflect')
                
                averaged_arr = np.zeros((H // self.window_height, W // self.window_width, 2), dtype=flow.dtype)
                averaged_uncert = np.zeros((H // self.window_height, W // self.window_width, 2), dtype=uncert.dtype)
                
                for channel in range(2):
                    averaged_arr[:, :, channel] = cv2.filter2D(
                        padded_arr[:, :, channel], -1, self.kernel
                    )[self.pad_height:-self.pad_height:self.window_height, self.pad_width:-self.pad_width:self.window_width]
                    
                    averaged_uncert[:, :, channel] = cv2.filter2D(
                        padded_uncert[:, :, channel], -1, self.kernel
                    )[self.pad_height:-self.pad_height:self.window_height, self.pad_width:-self.pad_width:self.window_width]
                
                self.velData[start_idx + i, :, :, :] = averaged_arr
                self.uncertData[start_idx + i, :, :, :] = averaged_uncert
        else:
            self.velData[start_idx:start_idx+B, :, :, :] = flow_batch
            self.uncertData[start_idx:start_idx+B, :, :, :] = uncert_batch

    def close(self):
        self.file.close()
