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
            
            # 1. Uncertainty of Mean Velocities (Standard Error of Mean = sqrt(sum_var) / N)
            umean_uncert = np.sqrt(sum_var_u) / count
            vmean_uncert = np.sqrt(sum_var_v) / count
            
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
    Rotates the velocity field around the throat location and extracts mean
    velocities and Reynolds stresses along specified wall-normal x-locations.
    
    Args:
        h5_filepath: path to the velocity .h5 file.
        x_positions_mm: List of x locations (in mm) to extract profiles.
        angle_deg: Angle to rotate the coordinate system (e.g., throat angle).
    
    Returns:
        dict containing 'y_coords', 'mean_u', 'mean_v', 'uu', 'vv', 'uv' for each x.
    """
    import os
    from tqdm import tqdm
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        attrs = dict(velData.attrs)
        
        mm_per_px = float(attrs.get('mm_per_px', 1.0))
        fps_capture = float(attrs.get('fps_capture', 1.0))
        window_width = int(attrs.get('window_width', 1))
        window_height = int(attrs.get('window_height', 1))
        
        # In tracking.py, displacement is in pixels per frame.
        # Scale factor to physical velocity (m/s)
        vel_scale_fac = mm_per_px * 1e-3 * fps_capture
        
        roi = list(attrs.get('roi', [0, -1, 0, -1] ))
        nFrames, Ny, Nx = velData.shape[:3]
        
        if roi[-1] == -1: roi[-1] = Nx * window_width
        if roi[1] == -1: roi[1] = Ny * window_height
        
        # Read throat location. Set default Y origin to bottom of ROI if unspecified.
        if 'throat_loc_px' in attrs:
            throat_loc_px = list(attrs['throat_loc_px'])
        else:
            throat_loc_px = [roi[1], 0]
            
        # Physical grids relative to throat
        xmm = np.arange(roi[2] + window_width // 2, roi[-1], window_width) * mm_per_px - throat_loc_px[1] * mm_per_px
        ymm = throat_loc_px[0] * mm_per_px - np.arange(roi[0] + window_height // 2, roi[1], window_height) * mm_per_px
        
        # Limit grid to actual data shapes in case of rounding errors
        xmm = xmm[:Nx]
        ymm = ymm[:Ny]

        theta = np.radians(angle_deg)
        c_th, s_th = np.cos(theta), np.sin(theta)
        
        s_values = ymm # Wall-normal distance array
        n_s = len(s_values)
        
        line_info = []
        for x_wall in x_positions_mm:
            # Rotate physical coordinates to find sampling locations
            x_c = x_wall * c_th
            y_c = -x_wall * s_th
            x_pts = x_c + s_values * s_th
            y_pts = y_c + s_values * c_th
            
            # Convert physical mm to fractional grid indices for map_coordinates
            j_frac = np.interp(x_pts, xmm, np.arange(Nx))
            
            # ymm is monotonically decreasing, so reverse it for np.interp
            i_frac = np.interp(y_pts, ymm[::-1], np.arange(Ny)[::-1])
            
            # Keep array coordinates for sampling (y evaluates to axis 0, x to axis 1)
            coords = np.vstack((i_frac, j_frac))
            line_info.append((x_wall, coords))
            
        n_lines = len(line_info)
        shape_r = (n_lines, n_s)
        
        sum_u  = np.zeros(shape_r, dtype=np.float64)
        sum_v  = np.zeros(shape_r, dtype=np.float64)
        sum_uu = np.zeros(shape_r, dtype=np.float64)
        sum_vv = np.zeros(shape_r, dtype=np.float64)
        sum_uv = np.zeros(shape_r, dtype=np.float64)
        count  = np.zeros(shape_r, dtype=np.int64)
        
        sum_var_u = np.zeros(shape_r, dtype=np.float64)
        sum_var_v = np.zeros(shape_r, dtype=np.float64)
        
        sum_u_var_u = np.zeros(shape_r, dtype=np.float64)
        sum_u2_var_u = np.zeros(shape_r, dtype=np.float64)
        sum_v_var_v = np.zeros(shape_r, dtype=np.float64)
        sum_v2_var_v = np.zeros(shape_r, dtype=np.float64)
        
        sum_u_var_v = np.zeros(shape_r, dtype=np.float64)
        sum_u2_var_v = np.zeros(shape_r, dtype=np.float64)
        sum_v_var_u = np.zeros(shape_r, dtype=np.float64)
        sum_v2_var_u = np.zeros(shape_r, dtype=np.float64)
        
        if frame_idx is not None:
            start_f = max(0, frame_idx)
            end_f   = min(nFrames, frame_idx + 1)
        else:
            start_f = 0
            end_f   = nFrames

        chunk_size = 100
        for t0 in tqdm(range(start_f, end_f, chunk_size), desc='Extracting Line Profiles'):
            t1 = min(t0 + chunk_size, end_f)
            chunk = velData[t0:t1]
            
            for b in range(chunk.shape[0]):
                u_raw = chunk[b, :, :, 0] * vel_scale_fac
                v_raw = -chunk[b, :, :, 1] * vel_scale_fac
                
                if 'uncertainty' in f:
                    unc_chunk = f['uncertainty'][t0:t1]
                    var_u_raw = (unc_chunk[b, :, :, 0] * vel_scale_fac) ** 2
                    var_v_raw = (unc_chunk[b, :, :, 1] * vel_scale_fac) ** 2
                else:
                    var_u_raw = np.zeros_like(u_raw)
                    var_v_raw = np.zeros_like(v_raw)
                
                u_rot, v_rot = rotate_velocity(u_raw, v_raw, angle_deg)
                
                # Rotate variances (assuming u and v orthogonal independent axes errors)
                c2, s2 = c_th**2, s_th**2
                var_u_rot = var_u_raw * c2 + var_v_raw * s2
                var_v_rot = var_u_raw * s2 + var_v_raw * c2
                
                for li, (x_wall, coords) in enumerate(line_info):
                    # Sample U, V, and Uncertainties at rotated line coordinates
                    u_s = map_coordinates(u_rot, coords, order=1, mode='constant', cval=np.nan)
                    v_s = map_coordinates(v_rot, coords, order=1, mode='constant', cval=np.nan)
                    var_u_s = map_coordinates(var_u_rot, coords, order=1, mode='constant', cval=0.0)
                    var_v_s = map_coordinates(var_v_rot, coords, order=1, mode='constant', cval=0.0)
                    
                    valid = np.isfinite(u_s) & np.isfinite(v_s)
                    
                    # Accumulate valid points
                    u_c = np.where(valid, u_s, 0.0)
                    v_c = np.where(valid, v_s, 0.0)
                    vu_c = np.where(valid, var_u_s, 0.0)
                    vv_c = np.where(valid, var_v_s, 0.0)
                    
                    sum_u[li]  += u_c
                    sum_v[li]  += v_c
                    sum_uu[li] += u_c ** 2
                    sum_vv[li] += v_c ** 2
                    sum_uv[li] += u_c * v_c
                    
                    sum_var_u[li] += vu_c
                    sum_var_v[li] += vv_c
                    
                    sum_u_var_u[li] += u_c * vu_c
                    sum_u2_var_u[li] += u_c**2 * vu_c
                    sum_v_var_v[li] += v_c * vv_c
                    sum_v2_var_v[li] += v_c**2 * vv_c
                    
                    sum_u_var_v[li] += u_c * vv_c
                    sum_u2_var_v[li] += u_c**2 * vv_c
                    sum_v_var_u[li] += v_c * vu_c
                    sum_v2_var_u[li] += v_c**2 * vu_c
                    
                    count[li]  += valid.astype(np.int64)

        # Average and compute Reynolds Stresses (Variance/Covariance)
        with np.errstate(invalid='ignore'):
            umean = sum_u / count
            vmean = sum_v / count
            
            if frame_idx is None:
                uu = sum_uu / count - umean ** 2
                vv = sum_vv / count - vmean ** 2
                uv = sum_uv / count - umean * vmean
                
                # Uncertainty Propagations
                umean_uncert = np.sqrt(sum_var_u) / count
                vmean_uncert = np.sqrt(sum_var_v) / count
                
                var_uu = (4.0 / (count ** 2)) * (sum_u2_var_u - 2 * umean * sum_u_var_u + umean**2 * sum_var_u)
                var_vv = (4.0 / (count ** 2)) * (sum_v2_var_v - 2 * vmean * sum_v_var_v + vmean**2 * sum_var_v)
                
                var_uv_part1 = (sum_v2_var_u - 2 * vmean * sum_v_var_u + vmean**2 * sum_var_u)
                var_uv_part2 = (sum_u2_var_v - 2 * umean * sum_u_var_v + umean**2 * sum_var_v)
                var_uv = (1.0 / (count ** 2)) * (var_uv_part1 + var_uv_part2)
                
                uu_uncert = np.sqrt(np.maximum(var_uu, 0))
                vv_uncert = np.sqrt(np.maximum(var_vv, 0))
                uv_uncert = np.sqrt(np.maximum(var_uv, 0))
            else:
                # Instantaneous frame: no temporal variance -> no Reynolds Stresses.
                uu = np.zeros_like(umean)
                vv = np.zeros_like(vmean)
                uv = np.zeros_like(umean)
                
                # Instantaneous uncertainty is simply the propagated pixel uncertainty.
                # (sum_var_u contains the single frame squared uncertainty since count=1)
                umean_uncert = np.sqrt(sum_var_u)
                vmean_uncert = np.sqrt(sum_var_v)
                
                uu_uncert = np.zeros_like(umean)
                vv_uncert = np.zeros_like(vmean)
                uv_uncert = np.zeros_like(umean)
            
        if frame_idx is not None:
            out_path = os.path.splitext(h5_filepath)[0] + f'_frame_{frame_idx}_lines.h5'
        else:
            out_path = os.path.splitext(h5_filepath)[0] + '_lines.h5'
        
        with h5py.File(out_path, 'w') as fout:
            fout.attrs['rotation_deg'] = angle_deg
            fout.attrs['coordinate_type'] = 'wall_normal'
            for li, (x_wall, _) in enumerate(line_info):
                mean_vel = np.stack([umean[li], vmean[li]], axis=1)
                mean_vel_uncert = np.stack([umean_uncert[li], vmean_uncert[li]], axis=1)
                
                rey = np.stack([uu[li], uv[li], vv[li]], axis=1)
                rey_uncert = np.stack([uu_uncert[li], uv_uncert[li], vv_uncert[li]], axis=1)
                
                grp_name = f'x_location_{str(x_wall).replace(".", "_")}'
                grp = fout.create_group(grp_name)
                grp.create_dataset('y_coordinates', data=s_values)
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
        for li, (x_wall, _) in enumerate(line_info):
            results[x_wall] = {
                'y_coords': s_values,
                'mean_u': umean[li],
                'mean_v': vmean[li],
                'uu': uu[li],
                'vv': vv[li],
                'uv': uv[li],
                'valid_counts': count[li]
            }
            
        return results, out_path
