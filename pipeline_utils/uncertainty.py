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

def compute_uncertainty_batch(image1, image2, flow, window_size=4):
    """
    Computes dense optical flow uncertainty metrics matching the 
    classical PIV 1D Gaussian Correlation Peak Fit format, fully batched.
    
    Args:
        image1: (B, C, H, W) -> will be converted to grayscale if C=3
        image2: (B, C, H, W)
        flow: (B, 2, H, W) in pixel displacement.
    Returns:
        uncertainty: (B, H, W, 2) containing (sigma_u, sigma_v) in pixels.
    """
    # Convert RGB to grayscale so correlation is single-channel
    if image1.shape[1] == 3:
        weight = torch.tensor([0.299, 0.587, 0.114], device=image1.device).view(1, 3, 1, 1)
        image1 = (image1 * weight).sum(dim=1, keepdim=True)
        image2 = (image2 * weight).sum(dim=1, keepdim=True)
        
    # Warp image2 backwards using the calculated flow
    img2_w = warp_img(image2, flow)
    eps = 1e-8
    
    # Use a 4x4 localized window over the 32x32 global region
    win_size = 4
    
    def get_ncc_shift(dx, dy):
        # Calculates 4x4 localized NCC for a given integer shift
        R = get_smoothed_ncc(image1, shift_img(img2_w, dx, dy), window_size=win_size)
        return torch.clamp(R.mean(dim=1, keepdim=True), eps, 1.0)

    # We shift Frame 2 by [-2, -1, 0, 1, 2] in X and Y to sweep an 8x8 region 
    # matching a 4x4 template natively.
    
    # Center
    R_0 = get_ncc_shift(0, 0)
    
    # X axis sweep (dx shifts)
    R_x_m2 = get_ncc_shift(2, 0)
    R_x_m1 = get_ncc_shift(1, 0)
    R_x_p1 = get_ncc_shift(-1, 0)
    R_x_p2 = get_ncc_shift(-2, 0)
    
    # Y axis sweep (dy shifts)
    R_y_m2 = get_ncc_shift(0, 2)
    R_y_m1 = get_ncc_shift(0, 1)
    R_y_p1 = get_ncc_shift(0, -1)
    R_y_p2 = get_ncc_shift(0, -2)

    # Stack to find max along the 5 shifts: (B, 1, H, W, 5)
    R_X = torch.stack([R_x_m2, R_x_m1, R_0, R_x_p1, R_x_p2], dim=-1)
    R_Y = torch.stack([R_y_m2, R_y_m1, R_0, R_y_p1, R_y_p2], dim=-1)

    # Autodetect the correlation peak index [0 to 4] for every pixel 
    max_idx_X = torch.argmax(R_X, dim=-1, keepdim=True)
    max_idx_Y = torch.argmax(R_Y, dim=-1, keepdim=True)
    
    # Ensure the center of our 3-point fit is not on the absolute edges 
    # so we can always extract a left/right neighbor for the curve fit.
    center_idx_X = torch.clamp(max_idx_X, 1, 3)
    center_idx_Y = torch.clamp(max_idx_Y, 1, 3)
    
    # Gather Rm (left), R0 (center), Rp (right)
    Rm_X = torch.gather(R_X, -1, center_idx_X - 1).squeeze(-1)
    R0_X = torch.gather(R_X, -1, center_idx_X).squeeze(-1)
    Rp_X = torch.gather(R_X, -1, center_idx_X + 1).squeeze(-1)
    
    Rm_Y = torch.gather(R_Y, -1, center_idx_Y - 1).squeeze(-1)
    R0_Y = torch.gather(R_Y, -1, center_idx_Y).squeeze(-1)
    Rp_Y = torch.gather(R_Y, -1, center_idx_Y + 1).squeeze(-1)
    
    def get_sigma(Rm, R0, Rp, center_idx):
        # 1. Compute Base Variance of the peak (Texture Sharpness)
        den = 2 * (torch.log(Rm) + torch.log(Rp) - 2 * torch.log(R0))
        den = den - eps 
        
        var = -2.0 / den
        
        # Clamp mathematical explosions to physical threshold (10.0px variance)
        var = torch.where(var < 0, torch.tensor(10.0, device=var.device), var)
        
        # 2. Compute sub-pixel shift delta from the prediction origin
        num = torch.log(Rm) - torch.log(Rp)
        x0 = num / den
        
        # Center index 2 corresponds to 0 shift.
        discrete_shift = center_idx.squeeze(-1).float() - 2.0
        delta = discrete_shift + x0
        
        # 3. Explicit requested isolation: Distance-based uncertainty ONLY.
        total_var = (delta ** 2)
        
        sigma = torch.sqrt(torch.clamp(total_var, min=0.0, max=100.0))
        return sigma

    sigma_u = get_sigma(Rm_X, R0_X, Rp_X, center_idx_X)
    sigma_v = get_sigma(Rm_Y, R0_Y, Rp_Y, center_idx_Y)
    
    # (B, 2, H, W) -> (B, H, W, 2)
    uncert = torch.cat((sigma_u, sigma_v), dim=1).permute(0, 2, 3, 1)
    
    return uncert.cpu().numpy()
