import numpy as np
import cv2
from scipy.optimize import minimize

def _detect_sideband(amp, H, W, min_area=5, margin=5.0):
    """
    Thuật toán nhận diện vùng Fourier tự động (Otsu + Morphology nâng ngưỡng 1%)
    được copy chính xác từ fourier_region_recognition của multiangle_phase_retrieval,
    chỉ thay đổi hướng chọn búp phổ sang bên PHẢI (centroid_x > cx).
    """
    cx, cy = W // 2, H // 2
    
    # 1. Tính ngưỡng Otsu khởi đầu (GTL) trên RAW amplitude
    amp_flat = amp.ravel()
    nbins = 256
    hist, bin_edges = np.histogram(amp_flat, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    total = hist.sum()
    sum_total = np.sum(bin_centers * hist)
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = bin_centers[0]
    
    for i in range(nbins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = bin_centers[i]
            
    gtl = best_threshold
    
    T = gtl
    step = 0.01 * gtl
    max_iter = 200
    best_components = None
    num_labels_prev = 0
    
    # Kernel morphology
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def _threshold_and_regionprops(threshold):
        bw = (amp > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_area = np.zeros_like(bw)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(bw_area, [cnt], -1, 255, -1)
        bw_close = cv2.morphologyEx(bw_area, cv2.MORPH_CLOSE, kernel_close)
        contours_fill, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_filled = np.zeros_like(bw_close)
        cv2.drawContours(bw_filled, contours_fill, -1, 255, -1)
        bw_open = cv2.morphologyEx(bw_filled, cv2.MORPH_OPEN, kernel_open)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_open)
        comps = []
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                comps.append({
                    'centroid': centroids[label_idx].copy(),
                    'bbox': (stats[label_idx, cv2.CC_STAT_LEFT],
                             stats[label_idx, cv2.CC_STAT_TOP],
                             stats[label_idx, cv2.CC_STAT_WIDTH],
                             stats[label_idx, cv2.CC_STAT_HEIGHT]),
                    'area': int(area)
                })
        return bw_open, comps

    # Bước 1: Lưu ảnh nhị phân tại GTL
    step1_binary, step1_components = _threshold_and_regionprops(gtl)
    
    # Bước 2: Tăng ngưỡng 1% GTL cho đến khi số vùng bằng 3
    for _ in range(max_iter):
        _, components = _threshold_and_regionprops(T)
        if len(components) == 3:
            best_components = components
            break
        elif len(components) == 2 and (best_components is None or num_labels_prev != 3):
            best_components = components
        elif len(components) == 1 and best_components is None:
            best_components = components
            
        num_labels_prev = len(components)
        T += step
        if T >= amp.max():
            break
            
    if best_components is None:
        best_components = components if len(components) > 0 else []
        
    # Phân loại: DC gần tâm nhất
    if len(best_components) > 0:
        dc_comp = min(best_components, key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)
        sidebands = [c for c in best_components if c is not dc_comp]
    else:
        sidebands = []
        
    if len(sidebands) > 0:
        # Chọn búp ở nửa PHẢI (centroid_x > cx)
        right_sidebands = [c for c in sidebands if c['centroid'][0] > cx]
        if len(right_sidebands) > 0:
            target_comp = max(right_sidebands, key=lambda c: c['area'])
        else:
            target_comp = max(sidebands, key=lambda c: c['centroid'][0])
            
        target_cx, target_cy = target_comp['centroid']
        
        # Bước 3: Tìm vùng lớn ở Bước 1
        best_step1_dist = float('inf')
        best_step1_comp = None
        for c in step1_components:
            d = np.sqrt((c['centroid'][0] - target_cx)**2 + (c['centroid'][1] - target_cy)**2)
            if d < best_step1_dist:
                best_step1_dist = d
                best_step1_comp = c
                
        if best_step1_comp is not None:
            left, top, w, h = best_step1_comp['bbox']
            px, py = best_step1_comp['centroid']
        else:
            left, top, w, h = target_comp['bbox']
            px, py = target_cx, target_cy
            
        rx = w / 2.0 + margin
        ry = h / 2.0 + margin
    else:
        # Fallback peak search ở nửa bên PHẢI
        search_amp = amp.copy()
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        dist_from_dc = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
        search_amp[dist_from_dc < 15] = 0
        search_amp[:, :cx] = 0  # Chỉ tìm ở nửa bên phải
        max_idx = np.argmax(search_amp)
        py_peak, px_peak = np.unravel_index(max_idx, search_amp.shape)
        px, py = float(px_peak), float(py_peak)
        rx, ry = 20.0, 20.0
        
    # Tinh chỉnh sub-pixel
    y_coords = np.arange(H)
    x_coords = np.arange(W)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    local_r = 7
    local_mask = (np.sqrt((X - px)**2 + (Y - py)**2) <= local_r)
    weights = amp[local_mask]
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        kx = float(np.sum(X[local_mask] * weights) / total_weight - cx)
        ky = float(np.sum(Y[local_mask] * weights) / total_weight - cy)
    else:
        kx = float(px - cx)
        ky = float(py - cy)
        
    return kx, ky, rx, ry


def detect_carrier(holo, min_area=5, margin=5.0):
    """
    Tiện ích gọi nội bộ để trả về các tham số sóng mang, dùng cho việc debug.
    Trả về: (kx, ky, rx, ry, abs_y, abs_x)
    """
    try:
        H, W = holo.shape
        # Áp dụng Hanning window để triệt tiêu nhiễu chữ thập (cross artifacts) ở trục ngang/dọc
        window = np.outer(np.hanning(H), np.hanning(W))
        holo_windowed = holo * window
        
        F = np.fft.fftshift(np.fft.fft2(holo_windowed))
        amp = np.abs(F)
        
        kx, ky, rx, ry = _detect_sideband(amp, H, W, min_area, margin)
        return kx, ky, rx, ry, float(H // 2 + ky), float(W // 2 + kx)
    except Exception as e:
        print(f"[WARN] detect_carrier failed: {e}")
        H, W = holo.shape
        # Default fallback: 1/4 khoảng cách bên phải
        return W // 4, 0.0, 30.0, 30.0, H // 2, W // 2 + W // 4


def extract_plus_one_order(holo, temperature=0.5, min_area=5, margin=5.0):
    """
    Trích xuất phổ bậc +1 từ hologram off-axis, SỬ DỤNG KIẾN TRÚC GIẢI ĐIỀU CHẾ KHẢ VI 
    (Differentiable Demodulator) tương tự như mạng nơ-ron:
    
    1. Xác định tần số sóng mang (kx, ky).
    2. Dịch tần số trong miền KHÔNG GIAN bằng exp(-j*2π*(kx*x/W + ky*y/H)).
       Lưu ý: Dấu âm (-) vì ta lấy búp bên PHẢI (kx > 0), muốn dời nó về tâm (0,0).
    3. Chuyển sang miền FFT, búp phổ +1 giờ đã nằm CHÍNH XÁC tại tâm (H/2, W/2).
    4. Áp dụng Sigmoid Soft Mask (hàm truyền đạt mượt) xung quanh tâm.
    5. Trả về phổ đã lọc.
    
    Returns:
        F_filtered: Phổ Fourier (đã dời về tâm và được lọc)
        centroid:   Tâm của phổ, LÀ HẰNG SỐ (H//2, W//2) vì đã dời bằng giải điều chế.
        mask:       Mặt nạ sigmoid
    """
    H, W = holo.shape
    cx_f, cy_f = W // 2, H // 2
    
    # 1. Phát hiện tần số mang (sóng mang)
    kx, ky, rx, ry, _, _ = detect_carrier(holo, min_area, margin)
    
    # 2. Giải điều chế (Dịch dải nền - baseband shift) trong không gian
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    phase_shift = -2.0 * np.pi * (kx * x_grid / W + ky * y_grid / H)
    exp_shift = np.exp(1j * phase_shift)
    
    I_shifted = holo.astype(np.complex128) * exp_shift
    
    # 3. FFT (Lúc này búp +1 đã nằm ở chính giữa)
    F_centered = np.fft.fftshift(np.fft.fft2(I_shifted))
    
    # 4. Mặt nạ Sigmoid mượt (Soft Mask)
    x_dist = np.abs(x_grid - cx_f)
    y_dist = np.abs(y_grid - cy_f)
    
    mask_x = 1.0 / (1.0 + np.exp(-(rx - x_dist) / temperature))
    mask_y = 1.0 / (1.0 + np.exp(-(ry - y_dist) / temperature))
    mask = mask_x * mask_y
    
    # 5. Lọc thông thấp (Low-pass filter)
    F_filtered = F_centered * mask
    
    # Trả về centroid luôn là tâm do đã dùng kỹ thuật demodulation
    return F_filtered, (float(cy_f), float(cx_f)), mask


def shift_spectrum_to_center(F_filtered, centroid):
    """
    Do đã áp dụng Differentiable Demodulator, phổ đã LUÔN LUÔN nằm ở chính giữa.
    Hàm này được giữ lại chỉ để đảm bảo tính tương thích với code cũ, 
    không thực hiện thay đổi gì (shift = 0).
    """
    return F_filtered


def apply_subpixel_shift(F_centered, k1, k2):
    """
    Bù phần dư sub-pixel (k1, k2) cho phổ F_centered, trả về trường sóng phức.
    IFFT(F)[y,x] * exp(+j2π*(k1*y/H + k2*x/W)) ↔ dịch phổ (k1, k2) pixel.
    Optimizer tìm (k1, k2) sao cho hai field khớp nhau.
    """
    H, W  = F_centered.shape
    y, x  = np.mgrid[0:H, 0:W]
    field = np.fft.ifft2(np.fft.ifftshift(F_centered))
    ramp  = np.exp(1j * 2 * np.pi * (k1 * y / H + k2 * x / W))
    return field * ramp


def phase_alignment_cost(k, F_centered_1, F_centered_2):
    """
    Hàm mục tiêu: Tối ưu hóa ĐỒNG THỜI lượng dịch cho cả ảnh 1 (k1) và ảnh 2 (k2).
    - Ảnh 1 thực hiện shift với k1 = (k1x, k1y)
    - Ảnh 2 thực hiện shift với k2 = (k2x, k2y)
    - Vòng lặp minimize sẽ chạy hàm này liên tục để tìm k làm cho Pha 1 ~ Pha 2.
    """
    k1x, k1y, k2x, k2y = k
    
    # Shift cả 2 ảnh
    field_1 = apply_subpixel_shift(F_centered_1, k1x, k1y)
    field_2 = apply_subpixel_shift(F_centered_2, k2x, k2y)
    
    # Tính sai lệch pha (Pha 1 vs Pha 2)
    diff = field_1 * np.conj(field_2)
    phase_only = diff / (np.abs(diff) + 1e-12)
    
    # Tính điểm khớp pha. Khi Pha 1 ~ Pha 2, điểm số np.abs(np.mean) sẽ hội tụ về 1.
    # Thêm lượng regularization siêu nhỏ (1e-6) để giữ (k1, k2) không bị trôi đi vô cực.
    reg = 1e-6 * (k1x**2 + k1y**2 + k2x**2 + k2y**2)
    
    return -np.abs(np.mean(phase_only)) + reg


def align_two_holograms(holo_1, holo_2):
    """
    Quy trình hoàn chỉnh với Differentiable Demodulator và Sub-pixel alignment.
    """
    # 1. Trích xuất (Phổ trả về đã được căn giữa hoàn toàn)
    F1_c, centroid_1, _ = extract_plus_one_order(holo_1)
    F2_c, centroid_2, _ = extract_plus_one_order(holo_2)

    # 2. Vòng lặp tối ưu hóa (Khởi tạo k1 và k2 đều bằng 0)
    k_init = [0.0, 0.0, 0.0, 0.0]

    # Lặp đến khi nào Pha 1 ~ Pha 2 thì dừng
    res = minimize(
        phase_alignment_cost,
        k_init,
        args=(F1_c, F2_c),
        method='Powell',
        options={'maxiter': 500, 'ftol': 1e-9}
    )
    k1x_opt, k1y_opt, k2x_opt, k2y_opt = res.x

    # 3. Xuất ra Pha 1 và Pha 2 sau khi shift với k1 và k2 tối ưu
    field_1_aligned = apply_subpixel_shift(F1_c, k1x_opt, k1y_opt)
    field_2_aligned = apply_subpixel_shift(F2_c, k2x_opt, k2y_opt)

    # 4. Bù hằng số pha toàn cục (Global Phase Offset) 
    # Ép hiệu pha hội tụ chính xác về mốc 0.0 rad
    diff_complex = field_1_aligned * np.conj(field_2_aligned)
    mean_phase_offset = np.angle(np.mean(diff_complex / (np.abs(diff_complex) + 1e-12)))
    field_2_aligned = field_2_aligned * np.exp(1j * mean_phase_offset)

    return {
        'field_1':             field_1_aligned,
        'field_2_aligned':     field_2_aligned,
        'k1_shifts':           (k1x_opt, k1y_opt),
        'k2_shifts':           (k2x_opt, k2y_opt),
        'centroids':           (centroid_1, centroid_2),
        'optimization_result': res,
        'phase_offset':        mean_phase_offset
    }
