import numpy as np
import cv2
from scipy.optimize import minimize

def _detect_sideband(amp, H, W, min_area=5, margin=5.0):
    """
    Thuật toán nhận diện vùng Fourier tự động (Otsu + Morphology).
    Trả về (kx, ky) tương đối so với tâm và bán kính (rx, ry) của bộ lọc.
    """
    cx, cy = W // 2, H // 2
    
    # ── Custom Otsu trên float data (tránh mất dải động khi ép uint8) ────────
    amp_flat        = amp.ravel()
    hist, bin_edges = np.histogram(amp_flat, bins=256)
    bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total           = hist.sum()
    sum_total       = np.sum(bin_centers * hist)
    sum_bg, w_bg    = 0.0, 0
    max_var, gtl    = 0.0, bin_centers[0]

    for i in range(256):
        w_bg += hist[i]
        if w_bg == 0: continue
        w_fg = total - w_bg
        if w_fg == 0: break
        sum_bg  += bin_centers[i] * hist[i]
        m_bg     = sum_bg / w_bg
        m_fg     = (sum_total - sum_bg) / w_fg
        variance = w_bg * w_fg * (m_bg - m_fg) ** 2
        if variance > max_var:
            max_var = variance
            gtl     = bin_centers[i]

    # ── Kernel morphology (kích thước vừa phải, ít hung hăng hơn) ───────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _threshold_and_regionprops(threshold):
        bw = (amp > threshold).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_area = np.zeros_like(bw)
        for cnt in cnts:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(bw_area, [cnt], -1, 255, -1)
        if cv2.countNonZero(bw_area) == 0:
            bw_area = bw
        bw_close = cv2.morphologyEx(bw_area, cv2.MORPH_CLOSE, kernel_close)
        cnts_f, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_fill   = np.zeros_like(bw_close)
        cv2.drawContours(bw_fill, cnts_f, -1, 255, -1)
        bw_open   = cv2.morphologyEx(bw_fill, cv2.MORPH_OPEN, kernel_open)
        if cv2.countNonZero(bw_open) == 0:
            bw_open = bw_area

        n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(bw_open)
        comps = []
        for idx in range(1, n_lbl):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                comps.append({
                    'centroid': centroids[idx].copy(),
                    'bbox': (stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP],
                             stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]),
                    'area': int(area)
                })
        return bw_open, comps

    _, step1_comps = _threshold_and_regionprops(gtl)

    T, step = gtl, 0.01 * gtl
    best_comps = None

    for _ in range(200):
        _, comps = _threshold_and_regionprops(T)
        if len(comps) == 3:
            best_comps = comps
            break
        elif len(comps) < 3:
            if best_comps is None:
                best_comps = comps if comps else (step1_comps if step1_comps else None)
            break
        best_comps = comps
        T += step
        if T >= amp.max(): break

    if not best_comps and step1_comps:
        best_comps = step1_comps

    if not best_comps:
        Y0, X0 = np.ogrid[0:H, 0:W]
        search = amp.copy()
        search[np.sqrt((X0 - cx)**2 + (Y0 - cy)**2) < 20] = 0

        def _pick_peak_as_comp(search_amp, suppress_r=35):
            idx = np.argmax(search_amp)
            py_, px_ = np.unravel_index(idx, search_amp.shape)
            bw_peak = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(bw_peak, (int(px_), int(py_)), suppress_r // 2, 255, -1)
            _, _, stats_, centroids_ = cv2.connectedComponentsWithStats(bw_peak)
            comp = {
                'centroid': np.array([float(px_), float(py_)]),
                'bbox': (max(0, int(px_) - suppress_r // 2), max(0, int(py_) - suppress_r // 2),
                         suppress_r, suppress_r),
                'area': int(np.pi * (suppress_r // 2) ** 2)
            }
            y1s = max(0, py_ - suppress_r)
            y2s = min(H, py_ + suppress_r)
            x1s = max(0, px_ - suppress_r)
            x2s = min(W, px_ + suppress_r)
            search_amp[y1s:y2s, x1s:x2s] = 0
            return comp

        best_comps = [_pick_peak_as_comp(search), _pick_peak_as_comp(search)]

    dc_comp = min(best_comps, key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)
    sidebands = [c for c in best_comps if c is not dc_comp]
    if not sidebands:
        sidebands = best_comps

    right_sb = [c for c in sidebands if c['centroid'][0] > cx]
    if right_sb:
        target = max(right_sb, key=lambda c: c['centroid'][0])
    else:
        target = max(sidebands, key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)

    target_cx, target_cy = target['centroid']

    best_dist, best_step1 = float('inf'), None
    for c in step1_comps:
        d = np.sqrt((c['centroid'][0] - target_cx)**2 + (c['centroid'][1] - target_cy)**2)
        if d < best_dist:
            best_dist  = d
            best_step1 = c

    if best_step1 is not None:
        left, top, w, h = best_step1['bbox']
        px, py          = best_step1['centroid']
    else:
        left, top, w, h = target['bbox']
        px, py          = target_cx, target_cy

    Y_grid, X_grid = np.mgrid[0:H, 0:W]
    local_mask = np.sqrt((X_grid - px)**2 + (Y_grid - py)**2) <= 7
    weights    = amp[local_mask]
    total_w    = weights.sum()

    if total_w > 0:
        refined_cx = float(np.sum(X_grid[local_mask] * weights) / total_w)
        refined_cy = float(np.sum(Y_grid[local_mask] * weights) / total_w)
    else:
        refined_cx, refined_cy = float(px), float(py)

    rx = w / 2.0 + margin
    ry = h / 2.0 + margin
    kx = refined_cx - cx
    ky = refined_cy - cy

    return kx, ky, rx, ry


def detect_carrier(holo, min_area=5, margin=5.0):
    """
    Tiện ích gọi nội bộ để trả về các tham số sóng mang, dùng cho việc debug.
    Trả về: (kx, ky, rx, ry, abs_y, abs_x)
    """
    try:
        F = np.fft.fftshift(np.fft.fft2(holo))
        amp = np.abs(F)
        H, W = holo.shape
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
