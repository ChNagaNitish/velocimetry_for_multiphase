import numpy as np
import h5py

def compute_spatial_derivatives(u, v, dx, dy):
    """
    Computes standard physical spatial derivatives for a 2D velocity field.
    Args:
        u: 2D numpy array of U velocities
        v: 2D numpy array of V velocities
        dx: Spatial resolution in x (m/px)
        dy: Spatial resolution in y (m/px)
    Returns:
        dictionary containing 'divergence', 'vorticity', 'strain_rate'
    """
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    strain_rate = 0.5 * (du_dy + dv_dx)
    
    return {
        'divergence': divergence,
        'vorticity': vorticity,
        'strain_rate': strain_rate,
        'du_dx': du_dx,
        'du_dy': du_dy,
        'dv_dx': dv_dx,
        'dv_dy': dv_dy
    }

def extract_profile_slice(x_query, x_grid, field_data):
    """
    Extracts a 1D column (y-profile) from a 2D field closest to the requested x-location.
    """
    # Find closest column index
    idx = (np.abs(x_grid - x_query)).argmin()
    return field_data[:, idx]

def rotate_velocity(u, v, angle_deg):
    """
    Rotates a velocity vector field by a given angle (e.g., to align with a throat wall).
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    u_rot = u * c - v * s
    v_rot = u * s + v * c
    return u_rot, v_rot

def compute_time_average_fields(h5_filepath, frame_start=None, frame_end=None):
    """
    Reads an HDF5 velocity file and computes the full 2D time-average of the mean
    velocities and Reynolds Stresses over the entire spatial grid.
    
    Returns a dictionary of mean_u, mean_v, uu, vv, and uv arrays.
    """
    from tqdm import tqdm
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        
        N = velData.shape[0]
        start = max(0, frame_start) if frame_start is not None else 0
        end = min(N, frame_end) if frame_end is not None else N
        
        attrs = dict(velData.attrs)
        mm_per_px = float(attrs.get('mm_per_px', 1.0))
        fps_capture = float(attrs.get('fps_capture', 1.0))
        
        # Scale factor from pixel-displacement per frame to m/s
        vel_scale_fac = mm_per_px * 1e-3 * fps_capture
        
        h, w = velData.shape[1:3]
        sum_u = np.zeros((h, w), dtype=np.float64)
        sum_v = np.zeros((h, w), dtype=np.float64)
        sum_uu = np.zeros((h, w), dtype=np.float64)
        sum_vv = np.zeros((h, w), dtype=np.float64)
        sum_uv = np.zeros((h, w), dtype=np.float64)
        
        # Uncertainty running sums
        sum_var_u = np.zeros((h, w), dtype=np.float64)
        sum_var_v = np.zeros((h, w), dtype=np.float64)
        
        sum_u_var_u = np.zeros((h, w), dtype=np.float64)
        sum_u2_var_u = np.zeros((h, w), dtype=np.float64)
        sum_v_var_v = np.zeros((h, w), dtype=np.float64)
        sum_v2_var_v = np.zeros((h, w), dtype=np.float64)
        
        sum_u_var_v = np.zeros((h, w), dtype=np.float64)
        sum_u2_var_v = np.zeros((h, w), dtype=np.float64)
        sum_v_var_u = np.zeros((h, w), dtype=np.float64)
        sum_v2_var_u = np.zeros((h, w), dtype=np.float64)
        
        chunk_size = 100
        count = 0
        
        for t0 in tqdm(range(start, end, chunk_size), desc='Computing Average 2D Fields'):
            t1 = min(t0 + chunk_size, end)
            chunk = velData[t0:t1]
            
            u_raw = chunk[..., 0] * vel_scale_fac
            v_raw = -chunk[..., 1] * vel_scale_fac
            
            # Uncertainty stored as std in pixels. Convert to physical variance (m^2/s^2)
            # Check if uncertainty dataset exists
            if 'uncertainty' in f:
                unc_chunk = f['uncertainty'][t0:t1]
                var_u_raw = (unc_chunk[..., 0] * vel_scale_fac) ** 2
                var_v_raw = (unc_chunk[..., 1] * vel_scale_fac) ** 2
            else:
                var_u_raw = np.zeros_like(u_raw)
                var_v_raw = np.zeros_like(v_raw)
            
            sum_u += np.sum(u_raw, axis=0)
            sum_v += np.sum(v_raw, axis=0)
            sum_uu += np.sum(u_raw**2, axis=0)
            sum_vv += np.sum(v_raw**2, axis=0)
            sum_uv += np.sum(u_raw * v_raw, axis=0)
            
            sum_var_u += np.sum(var_u_raw, axis=0)
            sum_var_v += np.sum(var_v_raw, axis=0)
            
            sum_u_var_u += np.sum(u_raw * var_u_raw, axis=0)
            sum_u2_var_u += np.sum(u_raw**2 * var_u_raw, axis=0)
            
            sum_v_var_v += np.sum(v_raw * var_v_raw, axis=0)
            sum_v2_var_v += np.sum(v_raw**2 * var_v_raw, axis=0)
            
            sum_u_var_v += np.sum(u_raw * var_v_raw, axis=0)
            sum_u2_var_v += np.sum(u_raw**2 * var_v_raw, axis=0)
            sum_v_var_u += np.sum(v_raw * var_u_raw, axis=0)
            sum_v2_var_u += np.sum(v_raw**2 * var_u_raw, axis=0)
            
            count += (t1 - t0)
            
        with np.errstate(invalid='ignore'):
            umean = sum_u / count
            vmean = sum_v / count
            uu = sum_uu / count - umean**2
            vv = sum_vv / count - vmean**2
            uv = sum_uv / count - umean * vmean
            
            # 1. Uncertainty of Mean Velocities (Time-averaged instantaneous uncertainty)
            umean_uncert = np.sqrt(sum_var_u / count)
            vmean_uncert = np.sqrt(sum_var_v / count)
            
            # 2. Uncertainty of Reynolds Stresses (Taylor Expansion of Variance)
            # Var(uu) = (4 / N^2) * SUM [ (u_i - umean)^2 * var_u_i ]
            # Expanding: SUM [ (u_i^2 - 2*umean*u_i + umean^2) * var_u_i ]
            var_uu = (4.0 / (count ** 2)) * (sum_u2_var_u - 2 * umean * sum_u_var_u + umean**2 * sum_var_u)
            var_vv = (4.0 / (count ** 2)) * (sum_v2_var_v - 2 * vmean * sum_v_var_v + vmean**2 * sum_var_v)
            
            # Var(uv) = (1 / N^2) * SUM [ (v_i - vmean)^2 * var_u_i  +  (u_i - umean)^2 * var_v_i ]
            var_uv_part1 = (sum_v2_var_u - 2 * vmean * sum_v_var_u + vmean**2 * sum_var_u)
            var_uv_part2 = (sum_u2_var_v - 2 * umean * sum_u_var_v + umean**2 * sum_var_v)
            var_uv = (1.0 / (count ** 2)) * (var_uv_part1 + var_uv_part2)
            
            uu_uncert = np.sqrt(np.maximum(var_uu, 0))
            vv_uncert = np.sqrt(np.maximum(var_vv, 0))
            uv_uncert = np.sqrt(np.maximum(var_uv, 0))
            
        return {
            'mean_u': umean,
            'mean_v': vmean,
            'uu': uu,
            'vv': vv,
            'uv': uv,
            'mean_u_uncert': umean_uncert,
            'mean_v_uncert': vmean_uncert,
            'uu_uncert': uu_uncert,
            'vv_uncert': vv_uncert,
            'uv_uncert': uv_uncert
        }

from scipy.ndimage import map_coordinates

def extract_line_profiles(h5_filepath, x_positions_mm, angle_deg=0.0, frame_idx=None):
    """
    Extracts mean velocities and Reynolds stresses along vertical (wall-normal) columns.
    Since the flow is assumed to be already rotated/aligned in tracking, angle_deg is ignored,
    and extraction is performed via highly optimized pure NumPy column slicing over the chunk.
    """
    import numpy as np
    import h5py
    from tqdm import tqdm
    import os
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        attrs = dict(velData.attrs)
        
        mm_per_px = float(attrs.get('mm_per_px', 1.0))
        fps_capture = float(attrs.get('fps_capture', 1.0))
        window_width = int(attrs.get('window_width', 1))
        window_height = int(attrs.get('window_height', 1))
        
        vel_scale_fac = mm_per_px * 1e-3 * fps_capture
        var_scale_fac = vel_scale_fac ** 2
        
        roi = list(attrs.get('roi', [0, -1, 0, -1] ))
        nFrames, Ny, Nx = velData.shape[:3]
        
        if roi[-1] == -1: roi[-1] = Nx * window_width
        if roi[1] == -1: roi[1] = Ny * window_height
        
        if 'throat_loc_px' in attrs:
            throat_loc_px = list(attrs['throat_loc_px'])
        else:
            throat_loc_px = [roi[1], 0]
            
        # 1D physical grids
        xmm = np.arange(roi[2] + window_width // 2, roi[-1], window_width) * mm_per_px - throat_loc_px[1] * mm_per_px
        ymm = throat_loc_px[0] * mm_per_px - np.arange(roi[0] + window_height // 2, roi[1], window_height) * mm_per_px
        xmm = xmm[:Nx]
        ymm = ymm[:Ny]
        
        line_info = []
        for x_wall in x_positions_mm:
            # Get 1D fractional horizontal index for column extraction
            j_frac = np.interp(x_wall, xmm, np.arange(Nx))
            j0 = int(np.floor(j_frac))
            j1 = min(j0 + 1, Nx - 1)
            dx = j_frac - j0
            
            line_info.append({
                'x_wall': x_wall,
                'j0': j0,
                'j1': j1,
                'w0': 1.0 - dx,
                'w1': dx
            })
            
        n_lines = len(line_info)
        n_s = Ny # The profile covers the entire Y column height automatically
        
        sum_u  = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_v  = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_uu = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_vv = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_uv = np.zeros((n_lines, n_s), dtype=np.float64)
        
        sum_var_u = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_var_v = np.zeros((n_lines, n_s), dtype=np.float64)
        
        sum_u_var_u = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_u2_var_u = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_v_var_v = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_v2_var_v = np.zeros((n_lines, n_s), dtype=np.float64)
        
        sum_u_var_v = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_u2_var_v = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_v_var_u = np.zeros((n_lines, n_s), dtype=np.float64)
        sum_v2_var_u = np.zeros((n_lines, n_s), dtype=np.float64)
        
        count = 0
        
        if frame_idx is not None:
            start_f = max(0, frame_idx)
            end_f   = min(nFrames, frame_idx + 1)
        else:
            start_f = 0
            end_f   = nFrames

        chunk_size = 100
        for t0 in tqdm(range(start_f, end_f, chunk_size), desc='Extracting Fast 1D Profiles'):
            t1 = min(t0 + chunk_size, end_f)
            chunk = velData[t0:t1]
            
            has_uncert = 'uncertainty' in f
            if has_uncert:
                unc_chunk = f['uncertainty'][t0:t1]

            B = chunk.shape[0]
            count += B
            
            for li, info in enumerate(line_info):
                j0, j1, w0, w1 = info['j0'], info['j1'], info['w0'], info['w1']
                
                # Slicing the full Y-column instantly over the whole 100-frame chunk
                u_chunk = (chunk[:, :, j0, 0] * w0 + chunk[:, :, j1, 0] * w1) * vel_scale_fac
                
                # In origin code, v raw was scaled by -vel_scale_fac to flip axis
                v_chunk = -(chunk[:, :, j0, 1] * w0 + chunk[:, :, j1, 1] * w1) * vel_scale_fac 

                if has_uncert:
                    var_u_chunk = (unc_chunk[:, :, j0, 0]**2 * w0 + unc_chunk[:, :, j1, 0]**2 * w1) * var_scale_fac
                    var_v_chunk = (unc_chunk[:, :, j0, 1]**2 * w0 + unc_chunk[:, :, j1, 1]**2 * w1) * var_scale_fac
                else:
                    var_u_chunk = np.zeros_like(u_chunk)
                    var_v_chunk = np.zeros_like(v_chunk)
                    
                sum_u[li] += np.sum(u_chunk, axis=0)
                sum_v[li] += np.sum(v_chunk, axis=0)
                sum_uu[li] += np.sum(u_chunk**2, axis=0)
                sum_vv[li] += np.sum(v_chunk**2, axis=0)
                sum_uv[li] += np.sum(u_chunk * v_chunk, axis=0)
                
                sum_var_u[li] += np.sum(var_u_chunk, axis=0)
                sum_var_v[li] += np.sum(var_v_chunk, axis=0)
                
                sum_u_var_u[li] += np.sum(u_chunk * var_u_chunk, axis=0)
                sum_u2_var_u[li] += np.sum(u_chunk**2 * var_u_chunk, axis=0)
                sum_v_var_v[li] += np.sum(v_chunk * var_v_chunk, axis=0)
                sum_v2_var_v[li] += np.sum(v_chunk**2 * var_v_chunk, axis=0)
                
                sum_u_var_v[li] += np.sum(u_chunk * var_v_chunk, axis=0)
                sum_u2_var_v[li] += np.sum(u_chunk**2 * var_v_chunk, axis=0)
                sum_v_var_u[li] += np.sum(v_chunk * var_u_chunk, axis=0)
                sum_v2_var_u[li] += np.sum(v_chunk**2 * var_u_chunk, axis=0)

        with np.errstate(invalid='ignore'):
            count = float(count)
            umean = sum_u / count
            vmean = sum_v / count
            
            if frame_idx is None:
                uu = sum_uu / count - umean ** 2
                vv = sum_vv / count - vmean ** 2
                uv = sum_uv / count - umean * vmean
                
                umean_uncert = np.sqrt(sum_var_u / count)
                vmean_uncert = np.sqrt(sum_var_v / count)
                
                var_uu = (4.0 / (count ** 2)) * (sum_u2_var_u - 2 * umean * sum_u_var_u + umean**2 * sum_var_u)
                var_vv = (4.0 / (count ** 2)) * (sum_v2_var_v - 2 * vmean * sum_v_var_v + vmean**2 * sum_var_v)
                
                var_uv_part1 = (sum_v2_var_u - 2 * vmean * sum_v_var_u + vmean**2 * sum_var_u)
                var_uv_part2 = (sum_u2_var_v - 2 * umean * sum_u_var_v + umean**2 * sum_var_v)
                var_uv = (1.0 / (count ** 2)) * (var_uv_part1 + var_uv_part2)
                
                uu_uncert = np.sqrt(np.maximum(var_uu, 0))
                vv_uncert = np.sqrt(np.maximum(var_vv, 0))
                uv_uncert = np.sqrt(np.maximum(var_uv, 0))
            else:
                uu = np.zeros_like(umean)
                vv = np.zeros_like(vmean)
                uv = np.zeros_like(umean)
                
                umean_uncert = np.sqrt(sum_var_u / count) # same logic for 1 frame
                vmean_uncert = np.sqrt(sum_var_v / count)
                
                uu_uncert = np.zeros_like(umean)
                vv_uncert = np.zeros_like(vmean)
                uv_uncert = np.zeros_like(umean)
            
        if frame_idx is not None:
            out_path = os.path.splitext(h5_filepath)[0] + f'_frame_{frame_idx}_lines.h5'
        else:
            out_path = os.path.splitext(h5_filepath)[0] + '_lines.h5'
        
        with h5py.File(out_path, 'w') as fout:
            fout.attrs['rotation_deg'] = 0.0 # Ignored by this fast extraction technique
            fout.attrs['coordinate_type'] = 'wall_normal'
            for li, info in enumerate(line_info):
                x_wall = info['x_wall']
                mean_vel = np.stack([umean[li], vmean[li]], axis=1)
                mean_vel_uncert = np.stack([umean_uncert[li], vmean_uncert[li]], axis=1)
                
                rey = np.stack([uu[li], uv[li], vv[li]], axis=1)
                rey_uncert = np.stack([uu_uncert[li], uv_uncert[li], vv_uncert[li]], axis=1)
                
                grp_name = f'x_location_{str(x_wall).replace(".", "_")}'
                grp = fout.create_group(grp_name)
                grp.create_dataset('y_coordinates', data=ymm)
                grp.attrs['coordinate_type'] = 'wall_normal_distance'
                
                vel_ds_out = grp.create_dataset('mean_velocity', data=mean_vel)
                vel_ds_out.attrs['components'] = ['u_parallel', 'v_normal']
                
                vel_unc_out = grp.create_dataset('mean_velocity_uncertainty', data=mean_vel_uncert)
                vel_unc_out.attrs['components'] = ['sigma_u_parallel', 'sigma_v_normal']
                
                rey_ds_out = grp.create_dataset('reynolds_stresses', data=rey)
                rey_ds_out.attrs['components'] = ['uu', 'uv', 'vv']
                
                rey_unc_out = grp.create_dataset('reynolds_stresses_uncertainty', data=rey_uncert)
                rey_unc_out.attrs['components'] = ['sigma_uu', 'sigma_uv', 'sigma_vv']
            
        results = {}
        valid_counts_arr = np.full_like(ymm, int(count)) # Kept for API compatibility
        for li, info in enumerate(line_info):
            x_wall = info['x_wall']
            results[x_wall] = {
                'y_coords': ymm,
                'mean_u': umean[li],
                'mean_v': vmean[li],
                'uu': uu[li],
                'vv': vv[li],
                'uv': uv[li],
                'valid_counts': valid_counts_arr
            }
            
        return results, out_path
