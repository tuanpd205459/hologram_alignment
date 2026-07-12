import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

def extract_plus_one_order(holo, sigma=5.0):
    """
    (1) & (2) & (3): Trích xuất phổ bậc +1 từ ảnh hologram off-axis.
    
    Parameters:
    - holo: Mảng numpy 2D, ảnh hologram đầu vào.
    - sigma: Độ lệch chuẩn cho bộ lọc Gaussian làm mượt cửa sổ.
    
    Returns:
    - F_filtered: Phổ đã được lọc bởi mask (chưa dịch).
    - centroid: Tọa độ tâm (y, x) của phổ bậc +1.
    - smoothed_mask: Cửa sổ lọc đã được làm mượt.
    """
    F = np.fft.fftshift(np.fft.fft2(holo))
    I = np.abs(F)
    I_log = np.log1p(I)
    
    GTL = threshold_otsu(I_log)
    thresh = GTL
    
    step = 0.01 * GTL
    max_thresh = np.max(I_log)
    
    regions = []
    bw = None
    
    while thresh < max_thresh:
        bw_current = I_log > thresh
        labeled = label(bw_current)
        current_regions = regionprops(labeled)
        
        valid_regions = [r for r in current_regions if r.area > 5]
        
        if len(valid_regions) == 3:
            regions = valid_regions
            bw = bw_current
            break
        elif len(valid_regions) < 3:
            if len(regions) == 0:
                regions = valid_regions
                bw = bw_current
            break
            
        regions = valid_regions
        bw = bw_current
        thresh += step
        
    if len(regions) < 2:
        raise ValueError("Không tìm thấy đủ các vùng phổ (DC, +1, -1). Vui lòng kiểm tra lại chất lượng ảnh hologram.")
        
    H, W = holo.shape
    center_y, center_x = H // 2, W // 2
    
    def dist_from_center(r):
        return (r.centroid[0] - center_y)**2 + (r.centroid[1] - center_x)**2
        
    regions.sort(key=dist_from_center, reverse=True)
    
    r_plus = regions[0]
    for r in regions:
        if r.centroid[0] > center_y or (r.centroid[0] == center_y and r.centroid[1] > center_x):
            r_plus = r
            break
            
    min_row, min_col, max_row, max_col = r_plus.bbox
    centroid_y, centroid_x = r_plus.centroid
    
    mask = np.zeros((H, W), dtype=float)
    mask[min_row:max_row, min_col:max_col] = bw[min_row:max_row, min_col:max_col]
    
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    F_filtered = F * smoothed_mask
    
    return F_filtered, (centroid_y, centroid_x), smoothed_mask

def shift_spectrum_to_center(F_filtered, centroid):
    H, W = F_filtered.shape
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    shift_y = H // 2 - cy
    shift_x = W // 2 - cx
    F_centered = np.roll(F_filtered, shift=(shift_y, shift_x), axis=(0, 1))
    return F_centered

def apply_subpixel_shift(F_centered, k1, k2):
    H, W = F_centered.shape
    y, x = np.mgrid[0:H, 0:W]
    
    complex_field = np.fft.ifft2(np.fft.ifftshift(F_centered))
    phase_ramp = np.exp(1j * 2 * np.pi * (k1 * y / H + k2 * x / W))
    
    shifted_field = complex_field * phase_ramp
    return shifted_field

def phase_alignment_cost(k, F_centered_1, F_centered_2):
    k1, k2 = k
    field_1 = np.fft.ifft2(np.fft.ifftshift(F_centered_1))
    field_2_shifted = apply_subpixel_shift(F_centered_2, k1, k2)
    
    phase_diff = np.angle(field_1 * np.conj(field_2_shifted))
    cost = np.var(phase_diff)
    return cost

def align_two_holograms(holo_1, holo_2):
    F1, centroid_1, mask_1 = extract_plus_one_order(holo_1)
    F2, centroid_2, mask_2 = extract_plus_one_order(holo_2)
    
    F1_c = shift_spectrum_to_center(F1, centroid_1)
    F2_c = shift_spectrum_to_center(F2, centroid_2)
    
    k1_init = (centroid_2[0] - round(centroid_2[0])) - (centroid_1[0] - round(centroid_1[0]))
    k2_init = (centroid_2[1] - round(centroid_2[1])) - (centroid_1[1] - round(centroid_1[1]))
    
    res = minimize(phase_alignment_cost, [k1_init, k2_init], args=(F1_c, F2_c), method='Nelder-Mead')
    k1_opt, k2_opt = res.x
    
    field_1 = np.fft.ifft2(np.fft.ifftshift(F1_c))
    field_2_aligned = apply_subpixel_shift(F2_c, k1_opt, k2_opt)
    
    return {
        'field_1': field_1,
        'field_2_aligned': field_2_aligned,
        'k_shifts': (k1_opt, k2_opt),
        'centroids': (centroid_1, centroid_2),
        'optimization_result': res
    }
