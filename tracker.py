import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VideoPairDataset
from pipeline_utils.io import HDF5Writer

from models.raft_wrapper import RAFTOpticalFlow
from models.farneback_wrapper import FarnebackOpticalFlow
from models.openpiv_wrapper import OpenPIVModel

def main(args):
    # 1. Setup Data Loader
    print("Initializing Video DataLoader...")
    dataset = VideoPairDataset(
        video_path=args.path,
        roi=args.roi,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
        max_frames=args.frames,
        rotate_angle=args.rotate_angle,
        rotate_center=tuple(args.rotate_center) if args.rotate_center else None
    )
    
    # 2. Setup Model
    print(f"Initializing {args.method.upper()} model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_path, ext = os.path.splitext(args.path)
    base_name = f"{base_path}_{ext[1:].lower()}"
    
    if args.method == 'raft':
        model_path = args.model
        if not os.path.exists(model_path):
            # Auto-resolve missing weights to the new /weights directory
            alt_path = os.path.join('weights', os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
            elif not model_path.endswith('.pth'):
                alt_path_pth = os.path.join('weights', f"{os.path.basename(model_path)}.pth")
                if os.path.exists(alt_path_pth):
                    model_path = alt_path_pth
        
        if not os.path.exists(model_path):
            print(f"Warning: RAFT model weights not found at {model_path}!")
            
        model = RAFTOpticalFlow(
            model_path=model_path,
            device=device,
            small=args.small,
            mixed_precision=args.mixed_precision,
            alternate_corr=args.alternate_corr
        )
        # Use the resolved base name for output file generation
        output_path = base_name + '_' + model_path.split("/")[-1][:-4] + '.h5'
    elif args.method == 'farneback':
        model = FarnebackOpticalFlow()
        output_path = base_name + '_fb_' + args.model + '.h5'
    elif args.method == 'openpiv':
        model = OpenPIVModel()
        output_path = base_name + '_piv_' + args.model + '.h5'
    else:
        raise ValueError(f"Unknown Method: {args.method}")

    # DataLoader Batch Size
    # OpenPIV runs multi-process chunks of 16. RAFT uses DataParallel batches.
    dl_batch_size = 16 if args.method == 'openpiv' else args.batch_size
    
    dataloader = DataLoader(
        dataset, 
        batch_size=dl_batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )

    # 4. Get expected dimensions from a sample
    sample_img1, sample_img2, _ = dataset[0]
    _, h, w = sample_img1.shape

    # 5. Setup HDF5 Output Writer
    print(f"Initializing HDF5 output at: {output_path}")
    metadata = {
        'window_height': args.win_h,
        'window_width': args.win_w,
        'mm_per_px': args.imgScale,
        'fps_capture': args.fpsCam,
        'roi': args.roi
    }
    if args.throat_loc is not None:
        metadata['throat_loc_px'] = args.throat_loc
    if args.rotate_angle != 0.0:
        metadata['rotate_angle'] = args.rotate_angle
    if args.rotate_center is not None:
        metadata['rotate_center'] = args.rotate_center
    
    writer = HDF5Writer(
        output_path=output_path,
        frame_count=dataset.frame_count,
        h=h,
        w=w,
        window_height=args.win_h,
        window_width=args.win_w,
        metadata=metadata
    )

    # 5. Core Execution Loop
    print(f"Starting Optical Flow Execution on {device.upper()}...")
    
    for image1_batch, image2_batch, indices in tqdm(dataloader, desc="Processing Batches"):
        # Inference
        flow_batch, uncert_batch = model.predict_batch(image1_batch, image2_batch)
        
        # Determine HDF5 start index 
        # (Since shuffle=False, indices are ordered. We use the first index in the batch.)
        start_idx = indices[0].item()
        
        # Save
        writer.write_batch(start_idx, flow_batch, uncert_batch)

    writer.close()
    print("Processing Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modular High-Speed PIV and Optical Flow Pipeline")
    
    # Core settings
    parser.add_argument('--method', default='farneback', choices=['raft', 'farneback', 'openpiv'])
    parser.add_argument('--model', default='c1', help="For RAFT: model path (.pth) or name in weights/ directory. For Farneback/OpenPIV: ID string.")
    parser.add_argument('--path', required=True, help="Input video file")
    parser.add_argument('-n', '--frames', type=int, default=-1, help="Max frames to process. -1 for entire video.")
    
    # Physical/Grid Settings
    parser.add_argument('--win_h', type=int, default=4, help="Averaging window height")
    parser.add_argument('--win_w', type=int, default=4, help="Averaging window width")
    parser.add_argument('--imgScale', type=float, default=0.001, help="Image Scale (mm/px)")
    parser.add_argument('--fpsCam', type=int, default=130000, help="FPS of camera capture")
    parser.add_argument('--roi', type=int, nargs=4, default=[0, -1, 0, -1], help="ROI: y_start y_end x_start x_end (-1 to end)")
    
    # Multi-Processing / Multi-GPU settings
    parser.add_argument('--batch_size', type=int, default=1, help="Number of frames per PyTorch batch")
    parser.add_argument('--num_workers', type=int, default=4, help="CPU threads for video decoding")
    
    # RAFT specific
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')

    # CLAHE specifics
    parser.add_argument('--use_clahe', action='store_true')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0)
    parser.add_argument('--clahe_tile_size', type=int, default=8)
    
    # Physical reference
    parser.add_argument('--throat_loc', type=int, nargs=2, default=None, help='Throat location in pixels (y, x)')

    # Pre-rotation
    parser.add_argument('--rotate_angle', type=float, default=0.0, help='Pre-rotation angle in degrees (positive = CCW)')
    parser.add_argument('--rotate_center', type=int, nargs=2, default=None, help='Rotation center in pixels (y, x). Required if rotate_angle != 0.')

    args = parser.parse_args()
    main(args)
