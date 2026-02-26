import torch
import torch.nn.functional as F


def smooth_field(f, window_size=32):
    """
    Applies a spatial box filter (average pooling) to a tensor, 
    preserving the original dimensions via reflection padding.
    """
    B, C, H, W = f.shape
    kernel = torch.ones(C, 1, window_size, window_size, device=f.device) / (window_size * window_size)
    
    # Pad symmetrically so output is exactly H, W
    p_left = window_size // 2
    p_right = window_size - p_left - 1
    p_top = window_size // 2
    p_bottom = window_size - p_top - 1
    
    f_pad = F.pad(f, (p_left, p_right, p_top, p_bottom), mode='reflect')
    return F.conv2d(f_pad, kernel, groups=C)

def get_smoothed_ncc(img1, img2, window_size=32):
    """
    Calculates the Normalized Cross-Correlation between two image tensors
    where the local mean and variance are taken over a sliding window.
    """
    mean1 = smooth_field(img1, window_size)
    mean2 = smooth_field(img2, window_size)
    
    var1 = smooth_field(img1 * img1, window_size) - mean1 * mean1
    var2 = smooth_field(img2 * img2, window_size) - mean2 * mean2
    
    cov = smooth_field(img1 * img2, window_size) - mean1 * mean2
    
    std1 = torch.sqrt(torch.clamp(var1, min=1e-8))
    std2 = torch.sqrt(torch.clamp(var2, min=1e-8))
    ncc = cov / (std1 * std2)
    return ncc

def warp_img(img, flow):
    """
    Warps an image tensor according to a flow field.
    flow: (B, 2, H, W) in pixel displacement.
    """
    B, C, H, W = img.shape
    
    # Create meshgrid of base coordinates
    xx = torch.arange(0, W, device=img.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=img.device).view(-1, 1).repeat(1, W)
    
    # Add batch and channel dimensions: (B, 1, H, W) -> (B, 2, H, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
    
    # Displace coordinates by the flow field
    grid = torch.cat((xx, yy), dim=1) + flow
    
    # Normalize coordinates to [-1, 1] for grid_sample
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    grid = grid.permute(0, 2, 3, 1) # (B, H, W, 2)
    img_warped = F.grid_sample(img, grid, align_corners=True, padding_mode='reflection')
    return img_warped

def shift_img(img, dx, dy):
    """
    Shifts an image tensor by integer pixel amounts using reflection padding.
    """
    B, C, H, W = img.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    
    padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    
    start_y = pad_top - dy
    start_x = pad_left - dx
    return padded[:, :, start_y:start_y+H, start_x:start_x+W]

def compute_uncertainty_batch(image1, image2, flow, window_size=8):
    """
    Computes dense optical flow uncertainty via NCC Peak Shift analysis.
    
    For each pixel, warps frame2 by the predicted flow, then searches a local 
    window of shifts to find the NCC peak. The distance between the peak and 
    the zero-shift position (where the tracker placed it) gives the uncertainty.
    
    Args:
        image1: (B, C, H, W) — first frame tensor
        image2: (B, C, H, W) — second frame tensor
        flow: (B, 2, H, W) in pixel displacement
        window_size: NCC correlation window size (default 8)
    Returns:
        uncertainty: (B, H, W, 2) containing (sigma_u, sigma_v) in pixels.
    """
    # Convert to grayscale if needed
    if image1.shape[1] == 3:
        weight = torch.tensor([0.299, 0.587, 0.114], device=image1.device).view(1, 3, 1, 1)
        image1 = (image1 * weight).sum(dim=1, keepdim=True)
        image2 = (image2 * weight).sum(dim=1, keepdim=True)
    
    # Warp frame2 by predicted flow
    img2_w = warp_img(image2, flow)
    
    eps = 1e-6
    max_shift = window_size // 2
    shifts = range(-max_shift, max_shift + 1)
    n_shifts = len(shifts)
    B, C, H, W = image1.shape
    device = image1.device
    
    # Build NCC map over all (dx, dy) shift combinations
    ncc_map = torch.zeros((B, n_shifts, n_shifts, H, W), device=device)
    for i, dx in enumerate(shifts):
        for j, dy in enumerate(shifts):
            shifted_img2 = shift_img(img2_w, dx, dy)
            ncc = get_smoothed_ncc(image1, shifted_img2, window_size=window_size)
            ncc_map[:, i, j, :, :] = torch.clamp(ncc, -1.0, 1.0)[:, 0, :, :]
    
    # Find peak in flattened shift space
    ncc_flat = ncc_map.view(B, n_shifts * n_shifts, H, W)
    max_idx = torch.argmax(ncc_flat, dim=1, keepdim=True)
    max_idx_x = max_idx // n_shifts
    max_idx_y = max_idx % n_shifts
    
    # Clamp to allow 3-point Gaussian fitting
    center_x = torch.clamp(max_idx_x, 1, n_shifts - 2)
    center_y = torch.clamp(max_idx_y, 1, n_shifts - 2)
    
    def get_1d_peak_shift(cx, cy, is_x):
        """Sub-pixel peak via 3-point Gaussian fit along one axis."""
        if is_x:
            idx_m = (cx - 1) * n_shifts + cy
            idx_0 = cx * n_shifts + cy
            idx_p = (cx + 1) * n_shifts + cy
        else:
            idx_m = cx * n_shifts + (cy - 1)
            idx_0 = cx * n_shifts + cy
            idx_p = cx * n_shifts + (cy + 1)
        
        Rm = torch.clamp(torch.gather(ncc_flat, 1, idx_m).squeeze(1), eps, 1.0)
        R0 = torch.clamp(torch.gather(ncc_flat, 1, idx_0).squeeze(1), eps, 1.0)
        Rp = torch.clamp(torch.gather(ncc_flat, 1, idx_p).squeeze(1), eps, 1.0)
        
        den = 2 * (torch.log(Rm) + torch.log(Rp) - 2 * torch.log(R0)) - eps
        x0 = (torch.log(Rm) - torch.log(Rp)) / den
        
        discrete_shift = (cx if is_x else cy).squeeze(1).float() - max_shift
        delta = torch.clamp(discrete_shift + x0, -max_shift * 2.0, max_shift * 2.0)
        return torch.abs(delta)
    
    sigma_u = get_1d_peak_shift(center_x, center_y, is_x=True)   # (B, H, W)
    sigma_v = get_1d_peak_shift(center_x, center_y, is_x=False)  # (B, H, W)
    
    # Clamp extreme values
    sigma_u = torch.clamp(sigma_u, 0.0, max_shift * 2.0)
    sigma_v = torch.clamp(sigma_v, 0.0, max_shift * 2.0)
    
    # (B, H, W) -> (B, H, W, 2)
    uncert = torch.stack((sigma_u, sigma_v), dim=-1)
    
    return uncert.cpu().numpy()
