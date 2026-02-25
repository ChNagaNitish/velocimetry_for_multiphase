"""
Command Line Interface for visualizing optical flow results.
Designed for research group members to use without modifying code.
"""

import argparse
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Import the newly created postprocess package
from postprocess import (
    create_quiver_video,
    create_quiver_image,
    create_contour_video,
    create_spacetime_diagram,
    compute_shedding_frequency,
    compute_cavity_metrics,
    extract_line_profiles,
    plot_profile_comparison,
    compute_time_average_fields,
    plot_average_contours
)

def main():
    parser = argparse.ArgumentParser(
        description="Optical Flow Visualization Tool. Generate videos and plots from raw videos & HDF5 results."
    )
    
    # Base inputs (optional for standalone comparison tasks)
    parser.add_argument('--video', help="Path to the original raw video (.avi, .mp4)")
    parser.add_argument('--h5', help="Path to the velocity/uncertainty .h5 file from tracking.py")
    
    # Optional outputs
    parser.add_argument('--out_dir', default='results', help="Directory to save generated visualizations")
    
    # Quiver features
    parser.add_argument('--quiver', action='store_true', help="Generate a quiver (arrow) velocity overlay video")
    parser.add_argument('--quiver_scale', type=float, default=4.0, help="Velocity arrow length multiplier. Default: 4.0")
    parser.add_argument('--quiver_skip', type=int, default=16, help="Pixel spacing between arrows. Default: 16")
    
    # Contour features
    parser.add_argument('--contour', action='store_true', help="Generate a translucent heat-map contour overlay video")
    parser.add_argument('--dataset', type=str, default='uncertainty', choices=['velocity', 'uncertainty'], 
                        help="Which dataset from the HDF5 to render as a contour (velocity or uncertainty)")
    parser.add_argument('--alpha', type=float, default=0.5, help="Heatmap transparency. 0.0=invisible, 1.0=opaque. Default: 0.5")
    
    # Spacetime plot features
    parser.add_argument('--spacetime', action='store_true', help="Generate an average Space-Time (x-t) tracking plot")
    
    # Analytics features
    parser.add_argument('--average', action='store_true', help="Compute and output full 2D time-averaged velocity and Reynolds Stresses")
    parser.add_argument('--frequency', action='store_true', help="Compute and output the cavity shedding frequency (PSD)")
    parser.add_argument('--cavity', action='store_true', help="Compute and output the time-averaged cavity variance and length limits (in mm)")
    
    # Quantitative Data Extraction
    parser.add_argument('--profiles', type=float, nargs='+', help="List of x-locations (in mm) from throat to extract mean velocity and Reynolds stresses.")
    parser.add_argument('--angle', type=float, default=0.0, help="Angle (degrees) to rotate the velocity field about the throat before extraction.")
    parser.add_argument('--frame', type=int, default=None, help="Extract profiles at a specific instantaneous frame instead of time-averaged.")
    
    # Quantitative Comparison
    parser.add_argument('--compare', nargs='+', help="List of _lines.h5 files to compare")
    parser.add_argument('--labels', nargs='+', help="List of legends for the compared files")
    parser.add_argument('--prop', type=str, default='u', choices=['u', 'v', 'uu', 'uv', 'vv'], help="Property to compare (e.g. u, v, uu, uv, vv). Default: u")
    
    parser.add_argument('--plot', action='store_true', help="Generate and save PNG plots for analytical flags like --frequency, --cavity, or --profiles")

    args = parser.parse_args()
    
    if not (args.compare or (args.video and args.h5)):
        print("Error: You must either provide --video and --h5 for single-file processing, or --compare for multi-file comparisons.")
        sys.exit(1)
        
    if args.compare and not args.labels:
        print("Error: You must provide --labels corresponding to the --compare files.")
        sys.exit(1)
        
    if args.compare and len(args.compare) != len(args.labels):
        print("Error: The number of --compare files and --labels must match.")
        sys.exit(1)
        
    # Prepare the output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")
        
    # Get a base name for the output files
    if args.compare:
        base_name = "comparison"
    elif args.h5:
        base_name = os.path.splitext(os.path.basename(args.h5))[0]
    elif args.video:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
    else:
        base_name = "output"
    
    print("\n--- Starting Post-Processing Visualization ---")

    if args.compare:
        out_path = os.path.join(args.out_dir, f"{base_name}_{args.prop}_profiles.png")
        print(f"\n[C] Comparing {args.prop.upper()} profiles across {len(args.compare)} files --> {out_path}")
        plot_profile_comparison(args.compare, args.labels, prop=args.prop, output_path=out_path)
    
    if args.quiver:
        if args.frame is not None:
            out_path = os.path.join(args.out_dir, f"{base_name}_frame_{args.frame}_quiver.png")
            print(f"\n[1] Generating Instantaneous Quiver Image (Frame {args.frame}) --> {out_path}")
            create_quiver_image(args.video, args.h5, out_path, frame_idx=args.frame, scale=args.quiver_scale, skip=args.quiver_skip)
        else:
            out_path = os.path.join(args.out_dir, f"{base_name}_quiver.mp4")
            print(f"\n[1] Generating Quiver Video --> {out_path}")
            create_quiver_video(args.video, args.h5, out_path, scale=args.quiver_scale, skip=args.quiver_skip)
        
    if args.contour:
        out_path = os.path.join(args.out_dir, f"{base_name}_{args.dataset}_contour.mp4")
        print(f"\n[2] Generating {args.dataset.capitalize()} Contour Video --> {out_path}")
        create_contour_video(args.video, args.h5, dataset_name=args.dataset, output_path=out_path, alpha=args.alpha)
        
    if args.spacetime:
        # Load ROI automatically from HDF5 metadata
        with h5py.File(args.h5, 'r') as f:
            if 'roi' in f['velocity'].attrs:
                roi = tuple(f['velocity'].attrs['roi'])
                print(f"Loaded ROI from HDF5: {roi}")
            else:
                roi = [0, -1, 0, -1]
                print(f"ROI not found in HDF5, using full frame: {roi}")

        out_path = os.path.join(args.out_dir, f"{base_name}_spacetime.png")
        print(f"\n[3] Generating Space-Time (x-t) Diagram --> {out_path}")
        xt_image = create_spacetime_diagram(args.video, roi)
        
        # Save diagram using matplotlib
        plt.figure(figsize=(12, 6), layout='constrained')
        
        plt.imshow(xt_image.T, aspect='auto', cmap='gray')
        plt.title('Cavity Shedding Space-Time (x-t) Diagram')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Space (Pixels)')
        plt.colorbar(label='Averaged Brightness')
        plt.savefig(out_path, dpi=300, transparent=True)
        plt.close()
        
    if args.frequency:
        # Load ROI automatically from HDF5 metadata
        with h5py.File(args.h5, 'r') as f:
            if 'roi' in f['velocity'].attrs:
                roi = tuple(f['velocity'].attrs['roi'])
            else:
                roi = None
                
        signal, f_arr, Pxx, dom_freq = compute_shedding_frequency(args.video, roi=roi)
        print(f"\n[4] Dominant Shedding Frequency: {dom_freq:.2f} Hz")
        
        if args.plot:
            out_path = os.path.join(args.out_dir, f"{base_name}_frequency.png")
            print(f"    --> Saving Frequency Plot to {out_path}")
            
            plt.figure(figsize=(10, 5), layout='constrained')
            plt.plot(f_arr, Pxx)
            
            # Marking the peak value
            peak_idx = np.argmax(Pxx)
            plt.scatter(f_arr[peak_idx], Pxx[peak_idx], color='red', zorder=5)
            plt.annotate(f"{dom_freq:.2f} Hz", (f_arr[peak_idx], Pxx[peak_idx]), 
                         textcoords="offset points", xytext=(10,10), ha='left', color='red', weight='bold')
                         
            plt.title(f'Power Spectral Density (Dominant Frequency: {dom_freq:.1f} Hz)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.xlim(0, max(f_arr)/10) # Zoom into lower frequencies usually
            plt.grid(True)
            plt.savefig(out_path, dpi=300, transparent=True)
            plt.close()
            
    if args.cavity:
        # Get calibration factor from HDF5
        mm_per_px = 1.0
        with h5py.File(args.h5, 'r') as f:
            if 'mm_per_px' in f['velocity'].attrs:
                mm_per_px = f['velocity'].attrs['mm_per_px']
                print(f"\n[5] Calculating cavity length (Calibration: px = {mm_per_px:.6f} mm)")
            else:
                print(f"\n[5] Calculating cavity length (No calibration found, using pixels)")
                
        avg, var, cavity_mask, max_indx = compute_cavity_metrics(args.video)
        
        if max_indx is not None:
            length_px = max_indx[1]
            length_mm = length_px * mm_per_px
            print(f"    Cavity Length: {length_mm:.2f} mm ({length_px} px)")
        else:
            print("    Could not determine cavity length (variance threshold not met).")
        
        if args.plot and max_indx is not None:
            out_path = os.path.join(args.out_dir, f"{base_name}_cavity.png")
            print(f"    --> Saving Cavity Plot to {out_path}")
            
            plt.figure(figsize=(10, 4), layout='constrained')
            
            # Show mean plot
            plt.imshow(avg, cmap='gray')
            
            # Contour representing the cavity length
            if cavity_mask is not None:
                # Calculate coordinates for drawing the contour line
                plt.contour(cavity_mask, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.8)
                
            # Location of max std shown on mean plot
            plt.scatter(max_indx[1], max_indx[0], color='red', marker='x', s=100, label='Location of Max Standard Deviation')
            plt.legend(loc='upper right')
            
            plt.title(f'Time-Averaged Brightness with Cavity Extent ({length_mm:.2f} mm)')
            plt.axis('off')
            plt.savefig(out_path, dpi=300, transparent=True)
            plt.close()
            
    if args.average:
        print(f"\n[7] Computing Full 2D Time-Averaged Fields")
        avg_fields = compute_time_average_fields(args.h5)
        
        out_path_h5 = os.path.join(args.out_dir, f"{base_name}_average.h5")
        
        with h5py.File(out_path_h5, 'w') as fout:
            for key, data in avg_fields.items():
                fout.create_dataset(key, data=data)
        
        print(f"    --> Successfully saved numerical 2D averages to: {out_path_h5}")
        
        if args.plot:
            print(f"    --> Generating and saving contour plots to {args.out_dir}")
            saved_plots = plot_average_contours(avg_fields, args.out_dir, base_name)
            for path in saved_plots:
                print(f"        Saved: {path}")
        
    if args.profiles:
        if args.frame is not None:
            print(f"\n[6] Extracting Instantaneous Velocity Profiles at X = {args.profiles} mm (Frame {args.frame}, Rotated {args.angle}°)")
        else:
            print(f"\n[6] Extracting Time-Averaged Velocity Profiles at X = {args.profiles} mm (Rotated {args.angle}°)")
            
        results, lines_h5_path = extract_line_profiles(args.h5, args.profiles, angle_deg=args.angle, frame_idx=args.frame)
        
        print(f"    --> Successfully saved line extractions to: {lines_h5_path}")
        for x_wall, data in results.items():
            print(f"        X = {x_wall} mm | Extracted {data['valid_counts'][0]} valid spatial points")
        
    if not (args.quiver or args.contour or args.spacetime or args.frequency or args.cavity or args.profiles or args.compare or args.average):
        print("\nNo visualization flags were provided!")
        print("Please run with --quiver, --contour, --spacetime, --frequency, --cavity, --profiles, --average, and/or --compare.")
        print("Example: python3 visualize_results.py --video myvid.avi --h5 results.h5 --quiver")
    else:
        print("\n--- Processing Complete ---")

if __name__ == '__main__':
    main()
