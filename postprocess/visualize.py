import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_quiver_video(video_path, h5_filepath, output_path, scale=2.0, skip=16):
    """
    Overlays a sparse velocity vector field (quiver plot) onto the original video frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        frames = velData.shape[0]
        
        roi = list(velData.attrs.get('roi', [0, -1, 0, -1]))
        if roi[1] == -1: roi[1] = height
        if roi[3] == -1: roi[3] = width
        y0, y1, x0, x1 = roi[0], roi[1], roi[2], roi[3]
        
        win_w = int(velData.attrs.get('window_width', 1))
        win_h = int(velData.attrs.get('window_height', 1))
        
        # Read first frame to match indexing (N frames -> N-1 velocity fields)
        ret, _ = cap.read() 
        
        for k in tqdm(range(frames), desc="Rendering Quiver Video"):
            ret, frame = cap.read()
            if not ret:
                break
                
            u = velData[k, :, :, 0]
            v = velData[k, :, :, 1]
            Ny, Nx = u.shape
            
            # Map directly from HDF5 matrix indices to physical block centers
            for j in range(0, Ny, skip):
                for i in range(0, Nx, skip):
                    du = int(u[j, i] * scale)
                    dv = int(v[j, i] * scale)
                    
                    if du != 0 or dv != 0:
                        # Global frame coordinates relative to ROI starting point and window center
                        y_glob = y0 + (j * win_h) + (win_h // 2)
                        x_glob = x0 + (i * win_w) + (win_w // 2)
                        
                        cv2.arrowedLine(frame, (x_glob, y_glob), (x_glob + du, y_glob + dv), (0, 0, 255), 1, tipLength=0.3)
                        
            out.write(frame)
            
    cap.release()
    out.release()

def create_quiver_image(video_path, h5_filepath, output_path, frame_idx=0, scale=2.0, skip=16):
    """
    Overlays a sparse velocity vector field (quiver plot) onto a single video frame and saves as an image.
    """
    cap = cv2.VideoCapture(video_path)
    
    # +1 because optical flow for index k corresponds to motion between frame k and k+1.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error reading frame {frame_idx+1} from video.")
        cap.release()
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        if frame_idx >= velData.shape[0]:
            print(f"Error: frame_idx {frame_idx} is out of bounds for velocity data with {velData.shape[0]} frames.")
            cap.release()
            return
            
        roi = list(velData.attrs.get('roi', [0, -1, 0, -1]))
        if roi[1] == -1: roi[1] = height
        if roi[3] == -1: roi[3] = width
        y0, y1, x0, x1 = roi[0], roi[1], roi[2], roi[3]
        
        win_w = int(velData.attrs.get('window_width', 1))
        win_h = int(velData.attrs.get('window_height', 1))
            
        u = velData[frame_idx, :, :, 0]
        v = velData[frame_idx, :, :, 1]
        Ny, Nx = u.shape
        
    # Map directly from HDF5 matrix indices to physical block centers
    for j in range(0, Ny, skip):
        for i in range(0, Nx, skip):
            du = int(u[j, i] * scale)
            dv = int(v[j, i] * scale)
            
            if du != 0 or dv != 0:
                y_glob = y0 + (j * win_h) + (win_h // 2)
                x_glob = x0 + (i * win_w) + (win_w // 2)
                
                cv2.arrowedLine(frame, (x_glob, y_glob), (x_glob + du, y_glob + dv), (0, 0, 255), 1, tipLength=0.3)
                
    cv2.imwrite(output_path, frame)
    cap.release()

def create_spacetime_diagram(video_path, roi):
    """
    Creates an x-t (space-time) diagram by extracting a specific row (or averaged rows)
    from a video over time to track cavity shedding.
    
    Args:
        video_path: Path to the raw high-speed video.
        roi: Tuple of (y1, y2, x1, x2) defining the horizontal strip to extract and average.
    Returns:
        numpy.ndarray representing the (Time, Space) image.
    """
    cap = cv2.VideoCapture(video_path)
    y1, y2, x1, x2 = roi
    
    xt_image = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        y_slice = slice(y1, y2 if y2 != -1 else None)
        x_slice = slice(x1, x2 if x2 != -1 else None)
        
        strip = gray[y_slice, x_slice]
        
        # Average vertically across the strip thickness to get a 1D intensity profile
        intensity_profile = np.mean(strip, axis=0)
        xt_image.append(intensity_profile)
        
    cap.release()
    return np.array(xt_image)

def create_contour_video(video_path, h5_filepath, dataset_name, output_path, alpha=0.5):
    """
    Overlays a translucent scalar field (e.g., uncertainty, divergence) as a heatmap
    onto the original grayscale video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with h5py.File(h5_filepath, 'r') as f:
        data = f[dataset_name][:]
        
        # Determine strict global colormap bounds (e.g. 5th to 95th percentile)
        flat_data = data.flatten()
        flat_data = flat_data[np.isfinite(flat_data)]
        vmin, vmax = np.percentile(flat_data, [5, 95])
        
        frames = data.shape[0]
        
        # Advance 1 video frame to align with flow arrays
        ret, _ = cap.read()
        
        for k in tqdm(range(frames), desc=f"Rendering {dataset_name} Contour Video"):
            ret, frame = cap.read()
            if not ret:
                break
                
            scalar_field = data[k]
            # Handle norm vector fields like uncertainty (u, v) -> magnitude
            if scalar_field.ndim == 3 and scalar_field.shape[-1] == 2:
                scalar_field = np.linalg.norm(scalar_field, axis=2)
                
            scalar_resized = cv2.resize(scalar_field, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 255]
            normed = np.clip((scalar_resized - vmin) / (vmax - vmin), 0, 1) * 255
            normed = normed.astype(np.uint8)
            
            # Apply JET colormap
            heatmap = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
            
            # Blend
            blended = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
            out.write(blended)
            
    cap.release()
    cap.release()
    out.release()

def create_profile_video(video_path, h5_filepath, output_path, num_profiles=10, fps=10):
    """
    Renders an mp4 video overlaying 10 equidistant instantaneous U-velocity profiles 
    on top of the velocity magnitude heatmap and raw camera frames.
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        attrs = dict(velData.attrs)
        
        mm_per_px = attrs.get('mm_per_px', 1.0)
        fps_capture = attrs.get('fps_capture', 1.0)
        win_w = int(attrs.get('window_width', 1))
        win_h = int(attrs.get('window_height', 1))
        roi = list(attrs.get('roi', [0, -1, 0, -1]))
        
        n_frames = velData.shape[0]
        Ny, Nx = velData.shape[1:3]
        
    scale_fac = mm_per_px * 1e-3 * fps_capture
    
    if roi[1] == -1: roi[1] = height
    if roi[3] == -1: roi[3] = width
    y0, x0 = roi[0], roi[2]
    
    # Pre-calculate spatial grids
    extent = [0, width, height, 0]
    
    # 10 equidistant indices along Nx
    x_indices = np.linspace(0, Nx - 1, num_profiles, dtype=int)
    
    # Construct y pixel locations matching velocity columns
    y_px = y0 + (np.arange(Ny) * win_h) + (win_h / 2.0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    profile_scale_px = 2.0  # 2px for 1m/s scale
    
    # Advance 1 video frame to align with flow arrays
    ret, _ = cap.read()
    
    print(f"Generating profile intersection animation at {fps} fps...")
    for k in tqdm(range(n_frames), desc="Rendering Profile Video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with h5py.File(h5_filepath, 'r') as f:
            vel = f['velocity'][k]
            uncert = f['uncertainty'][k] if 'uncertainty' in f else None
            
        u = vel[..., 0] * scale_fac
        v = -vel[..., 1] * scale_fac
        mag = np.sqrt(u**2 + v**2)
        
        if uncert is not None:
            sigma_u = uncert[..., 0] * scale_fac
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        ax.imshow(img_rgb)
        contour = ax.imshow(mag, cmap='jet', alpha=0.4, extent=extent, vmin=0, vmax=13.0)
        
        # Add colorbar 
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity Magnitude (m/s)')
        
        for idx_x in x_indices:
            x_base_px = x0 + (idx_x * win_w) + (win_w / 2.0)
            u_prof = u[:, idx_x]
            
            # Baseline zero reference
            ax.axvline(x=x_base_px, color='white', linestyle='--', alpha=0.4)
            
            if uncert is not None:
                su_prof = sigma_u[:, idx_x]
                ax.fill_betweenx(y_px, 
                                 x_base_px + (u_prof - su_prof) * profile_scale_px,
                                 x_base_px + (u_prof + su_prof) * profile_scale_px,
                                 color='red', alpha=0.3)
                
            # Extracted profile at instantaneous frame
            ax.plot(x_base_px + u_prof * profile_scale_px, y_px, color='red', linewidth=1.5)
            
        ax.set_title(f"Instantaneous Profile Flow field (Frame {k})")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        
        # Convert matplotlib canvas to CV2 BGR array
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, fps, (img_bgr.shape[1], img_bgr.shape[0]))
            
        out.write(img_bgr)
        plt.close(fig)
        
    cap.release()
    if out is not None:
        out.release()

def set_journal_style(column_type='single'):
    """
    Applies professional style and sets precise dimensions for papers.
    column_type: 'single' (3.5"), 'double' (7.0"), or 'half'
    """
    if column_type == 'double':
        width = 7.0
    elif column_type == 'half':
        width = 4.5
    else: 
        width = 3.5 

    height = width / 1.618 
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "figure.figsize": (width, height),
        "font.family": "serif", 
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.usetex": True,
        "text.latex.preamble": r'\usepackage{amsmath} \usepackage{amssymb}',
        "axes.linewidth": 0.6,
        "lines.linewidth": 1,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 1.0, 
        "legend.fancybox": False, 
        "legend.edgecolor": "black",
        "savefig.dpi": 300,
        "savefig.bbox": "tight", 
        "savefig.pad_inches": 0.05
    })

def plot_profile_comparison(paths, legends, prop='u', output_path=None, show_uncertainty=False):
    """
    Plots comparison profiles from multiple _lines.h5 files.

    Creates a grid of subplots (one per x-location). Each subplot shows
    the chosen property vs. y-coordinate for every file, labelled with
    the supplied legends. By default, time-averaged profile plots will hide
    uncertainty bounds unless show_uncertainty is True.

    Args:
        paths:   List of paths to _lines.h5 files produced by extract_line_profiles.
        legends: Display label for each file (same order as *paths*).
        prop:    Property to compare - 'u', 'v', 'uu', 'uv', or 'vv'.
        output_path: Optional path to save the figure. If None, plt.show().
        show_uncertainty: Whether to shade uncertainty bounds along the profiles.
    """
    prop_map = {
        'u':  ('mean_velocity', 0),
        'v':  ('mean_velocity', 1),
        'uu': ('reynolds_stresses', 0),
        'uv': ('reynolds_stresses', 1),
        'vv': ('reynolds_stresses', 2),
    }
    if prop not in prop_map:
        raise ValueError(f"Unknown property '{prop}'. Valid: {list(prop_map.keys())}")
    ds_name, col_idx = prop_map[prop]

    # Labels for axes
    prop_labels = {
        'u':  r'$\bar{u}$ (m/s)',
        'v':  r'$\bar{v}$ (m/s)',
        'uu': r"$\overline{u'u'}$ (m$^2$/s$^2$)",
        'uv': r"$\overline{u'v'}$ (m$^2$/s$^2$)",
        'vv': r"$\overline{v'v'}$ (m$^2$/s$^2$)",
    }

    # Discover all x-location groups across files
    all_x_locs = set()
    for p in paths:
        with h5py.File(p, 'r') as f:
            all_x_locs.update(k for k in f.keys() if k.startswith('x_location_'))

    def _parse_x(name):
        return float(name.replace('x_location_', '').replace('_', '.', 1))

    x_locs = sorted(list(all_x_locs), key=_parse_x)
    if not x_locs:
        print("No x-location groups found in provided files.")
        return

    n = len(x_locs)
    
    set_journal_style('double')
    
    # Re-apply the user's specific constrained layout pattern from OldavgPlot
    fig, axes = plt.subplots(1, n, figsize=(5.0, 3.5), sharey=True, squeeze=False, layout='constrained')

    uncert_map = {
        'u':  ('mean_velocity_uncertainty', 0),
        'v':  ('mean_velocity_uncertainty', 1),
        'uu': ('reynolds_stresses_uncertainty', 0),
        'uv': ('reynolds_stresses_uncertainty', 1),
        'vv': ('reynolds_stresses_uncertainty', 2),
    }
    
    # Safe lookup incase it doesn't match
    unc_ds_name, unc_col_idx = uncert_map.get(prop, (None, None))

    for idx, loc_name in enumerate(x_locs):
        ax = axes[0][idx]
        x_val = _parse_x(loc_name)

        for p, label in zip(paths, legends):
            try:
                with h5py.File(p, 'r') as f:
                    if loc_name not in f:
                        continue
                    grp = f[loc_name]
                    y = np.array(grp['y_coordinates'])
                    data = np.array(grp[ds_name][:, col_idx])

                    npts = min(len(y), len(data))
                    y = y[:npts]
                    data = data[:npts]

                    line, = ax.plot(data, y, label=label, linewidth=1.5)

                    if show_uncertainty and (unc_ds_name in grp):
                        sigma = np.array(grp[unc_ds_name][:npts, unc_col_idx])
                        ax.fill_betweenx(y, data - sigma, data + sigma, alpha=0.2, color=line.get_color())
            except Exception as e:
                print(f"Error reading {p}: {e}")

        ax.set_title(f'X = {x_val} mm', pad=10)
        ax.set_xlabel(prop_labels.get(prop, prop.upper()))
        ax.grid(True, linestyle='--', alpha=0.7)
        if idx == 0:
            ax.set_ylabel('Wall-Normal Distance (mm)')

    # Exact match of oldavgPlot.py legend extraction and rendering rules
    handles, labels = [], []
    for ax_flat in axes.flatten():
        h, l = ax_flat.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc='outside lower center', ncol=len(by_label), frameon=False)

    fig.suptitle(f'Comparison: {prop.upper()} Profile')

    if output_path:
        fig.savefig(output_path, transparent=True)
    else:
        plt.show()
    plt.close()

def plot_average_contours(avg_fields, output_dir, base_name):
    """
    Plots the time-averaged 2D spatial contours for Mean Velocities and 
    Reynolds Stresses computed from `compute_time_average_fields`.
    """
    import os
    
    props = {
        'mean_u': r'$\bar{U}$ (m/s)',
        'mean_v': r'$\bar{V}$ (m/s)',
        'uu': r"$\overline{u'u'}$ (m$^2$/s$^2$)",
        'vv': r"$\overline{v'v'}$ (m$^2$/s$^2$)",
        'uv': r"$\overline{u'v'}$ (m$^2$/s$^2$)"
    }
    
    saved_paths = []
    
    for key, label in props.items():
        if key in avg_fields:
            data = avg_fields[key]
            out_path = os.path.join(output_dir, f"{base_name}_avg_contour_{key}.png")
            
            # Create a 2D physical map
            plt.figure(figsize=(10, 5), layout='constrained')
            
            # Using contourf with inverted layout since row 0 is the top of the image mathematically
            # 'data' is structured as (Ny, Nx).
            contour = plt.contourf(data, levels=30, cmap='jet')
            plt.gca().invert_yaxis() # Ensure image renders geometrically top-down
            
            plt.colorbar(contour, label=label)
            plt.title(f'Time-Averaged Contour: {label}')
            plt.axis('off')
            
            plt.savefig(out_path, dpi=300, transparent=True)
            plt.close()
            saved_paths.append(out_path)
            
    return saved_paths
