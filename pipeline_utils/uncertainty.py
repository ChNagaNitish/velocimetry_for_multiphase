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

def compute_uncertainty_batch(image1, image2, flow, window_size=32):
    """
    Computes dense optical flow uncertainty metrics matching the 
    Window-Smoothed Correlation format, fully batched in PyTorch.
    
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
    
    # Base correlation
    R_0 = get_smoothed_ncc(image1, img2_w, window_size)
    
    # +/- 1 pixel shifts to sample the correlation peak surface
    R_px = get_smoothed_ncc(image1, shift_img(img2_w, -1, 0), window_size)
    R_mx = get_smoothed_ncc(image1, shift_img(img2_w, 1, 0), window_size)
    R_py = get_smoothed_ncc(image1, shift_img(img2_w, 0, -1), window_size)
    R_my = get_smoothed_ncc(image1, shift_img(img2_w, 0, 1), window_size)
    
    # Clamp to prevent log(0) or log(negative)
    eps = 1e-8
    R_0 = torch.clamp(R_0, eps, 1.0)
    R_px = torch.clamp(R_px, eps, 1.0)
    R_mx = torch.clamp(R_mx, eps, 1.0)
    R_py = torch.clamp(R_py, eps, 1.0)
    R_my = torch.clamp(R_my, eps, 1.0)
    
    def get_sigma(Rm, R0, Rp):
        # 3-point Gaussian Fit second derivative approximation
        den = 2 * (torch.log(Rm) + torch.log(Rp) - 2 * torch.log(R0))
        # Add epsilon to prevent div by zero
        den = den - eps 
        
        var = -2.0 / den
        sigma = torch.sqrt(torch.clamp(var, 0.0, 100.0))
        return sigma

    sigma_u = get_sigma(R_mx, R_0, R_px)
    sigma_v = get_sigma(R_my, R_0, R_py)
    
    # (B, 2, H, W) -> (B, H, W, 2)
    uncert = torch.cat((sigma_u, sigma_v), dim=1).permute(0, 2, 3, 1)
    
    return uncert.cpu().numpy()
