import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

def extract_plus_one_order(holo, sigma=5.0):
    F = np.fft.fftshift(np.fft.fft2(holo))
    # Sử dụng abs()^2 giống với nguyên bản cường độ trong Matlab
    I = np.abs(F) ** 2
    I_log = np.log1p(I)
    
    # Chuẩn hóa về [0, 1] để giống hàm graythresh của MATLAB nhất có thể
    I_norm = (I_log - np.min(I_log)) / (np.max(I_log) - np.min(I_log))
    
    GTL = threshold_otsu(I_norm)
    thresh = GTL
    step = 0.01 * GTL
    
    regions = []
    bw = None
    
    # Lặp theo thuật toán: Tăng ngưỡng cho đến khi số regions == 3
    # Ở đây ta sẽ lấy top các regions to nhất để tránh đếm nhiễu 1 pixel
    while thresh < 1.0:
        bw_current = I_norm > thresh
        labeled = label(bw_current)
        current_regions = regionprops(labeled)
        
        # Chỉ xét các regions có diện tích > 2 pixel để loại nhiễu rác
        valid_regions = [r for r in current_regions if r.area >= 2]
        
        if len(valid_regions) == 3:
            regions = valid_regions
            bw = bw_current
            break
        elif len(valid_regions) < 3:
            # Nếu ngưỡng quá cao làm mất order, lấy kết quả gần nhất trước đó
            if len(regions) == 0:
                regions = valid_regions
                bw = bw_current
            break
            
        regions = valid_regions
        bw = bw_current
        thresh += step
        
    if len(regions) < 2:
        # Nếu thuật toán GTL thất bại, thử fallback: chọn top 3 regions lớn nhất từ GTL ban đầu
        bw_current = I_norm > GTL
        labeled = label(bw_current)
        all_regions = regionprops(labeled)
        all_regions.sort(key=lambda r: r.area, reverse=True)
        regions = all_regions[:3]
        bw = bw_current

    H, W = holo.shape
    center_y, center_x = H // 2, W // 2
    
    # Lấy ra vùng bậc +1 (chắc chắn không phải là vùng DC ở trung tâm)
    # Sắp xếp theo khoảng cách từ tâm để loại DC (thường DC sẽ ở gần tâm nhất)
    def dist_from_center(r):
        return np.sqrt((r.centroid[0] - center_y)**2 + (r.centroid[1] - center_x)**2)
        
    regions.sort(key=dist_from_center, reverse=True)
    
    # Ưu tiên lấy vùng ở nửa dưới (hoặc nửa trên) một cách nhất quán
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
    
    # MỚI: Tối ưu bằng Correlation Pha (tránh lỗi phase wrapping của variance)
    # Tìm k1, k2 sao cho độ lệch pha giữa 2 ảnh là hằng số phẳng nhất
    # exp(i * phase_diff) sẽ cộng hưởng (cùng hướng) nếu phase_diff là phẳng.
    complex_diff = field_1 * np.conj(field_2_shifted)
    phase_only = complex_diff / (np.abs(complex_diff) + 1e-12)
    
    # Lấy giá trị trung bình của các vector pha. Nếu pha đồng nhất -> độ dài vector sẽ tiến về 1
    metric = np.abs(np.mean(phase_only))
    
    # minimize của scipy cần tìm giá trị nhỏ nhất, nên ta nhân với -1
    return -metric

def align_two_holograms(holo_1, holo_2):
    F1, centroid_1, mask_1 = extract_plus_one_order(holo_1)
    F2, centroid_2, mask_2 = extract_plus_one_order(holo_2)
    
    F1_c = shift_spectrum_to_center(F1, centroid_1)
    F2_c = shift_spectrum_to_center(F2, centroid_2)
    
    k1_init = (centroid_2[0] - round(centroid_2[0])) - (centroid_1[0] - round(centroid_1[0]))
    k2_init = (centroid_2[1] - round(centroid_2[1])) - (centroid_1[1] - round(centroid_1[1]))
    
    # Sử dụng thuật toán Powell hoặc Nelder-Mead với giới hạn vòng lặp
    res = minimize(phase_alignment_cost, [k1_init, k2_init], args=(F1_c, F2_c), method='Powell', options={'maxiter': 100})
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
