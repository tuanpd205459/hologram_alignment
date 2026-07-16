import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


def extract_plus_one_order(holo, sigma=5.0, min_area=5, margin=5.0):
    """
    Trích xuất phổ bậc +1 từ hologram off-axis.

    Thuật toán theo paper:
    'Automated Fourier space region-recognition filtering for off-axis DHM'

    Quy trình:
    (1) Biên độ phổ FFT thô (raw, KHÔNG log) → Custom Otsu float (GTL).
    (2) Lưu ảnh nhị phân tại GTL (bước 1) để lấy bbox lớn ở cuối.
    (3) Tăng ngưỡng 1% GTL + morphology cv2 (close→fill→open) lặp
        đến khi đúng 3 vùng (DC, +1, -1).
    (4) Lấy bbox từ bước 1 (đúng theo paper) để xây dựng cửa sổ lọc.
    (5) Tinh chỉnh sub-pixel bằng weighted centroid bán kính 7px.
    (6) Chọn nhất quán búp bên PHẢI tâm (centroid_x > center_x).
    Fallback: nếu không phát hiện được 3 vùng → tìm đỉnh cực đại
              trực tiếp trên biên độ sau khi loại bỏ vùng DC.

    Returns:
        F_filtered:    Phổ đã lọc (chưa dịch về tâm)
        centroid:      (centroid_y, centroid_x) = (row, col) của búp +1
        smoothed_mask: Mặt nạ Gaussian đã làm mượt
    """
    import cv2

    H, W = holo.shape
    cx, cy = W // 2, H // 2        # tâm DC trong ảnh phổ

    # ── FFT và biên độ thô ───────────────────────────────────────────────────
    F   = np.fft.fftshift(np.fft.fft2(holo))
    amp = np.abs(F)

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
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
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
        """Phân ngưỡng → lọc nhiễu nhỏ → morphology → connected components."""
        bw = (amp > threshold).astype(np.uint8) * 255

        # Loại contour quá nhỏ
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_area = np.zeros_like(bw)
        for cnt in cnts:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(bw_area, [cnt], -1, 255, -1)

        # Nếu không còn gì sau lọc nhiễu → dùng raw binary
        if cv2.countNonZero(bw_area) == 0:
            bw_area = bw

        # Close → fill → open
        bw_close = cv2.morphologyEx(bw_area, cv2.MORPH_CLOSE, kernel_close)
        cnts_f, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_fill   = np.zeros_like(bw_close)
        cv2.drawContours(bw_fill, cnts_f, -1, 255, -1)
        bw_open   = cv2.morphologyEx(bw_fill, cv2.MORPH_OPEN, kernel_open)

        # Nếu morphology xóa sạch → dùng lại bw_area
        if cv2.countNonZero(bw_open) == 0:
            bw_open = bw_area

        n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(bw_open)
        comps = []
        for idx in range(1, n_lbl):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                comps.append({
                    'centroid': centroids[idx].copy(),   # cv2: (x, y)
                    'bbox': (stats[idx, cv2.CC_STAT_LEFT],
                             stats[idx, cv2.CC_STAT_TOP],
                             stats[idx, cv2.CC_STAT_WIDTH],
                             stats[idx, cv2.CC_STAT_HEIGHT]),
                    'area': int(area)
                })
        return bw_open, comps

    # ── BƯỚC 1: Binary tại GTL gốc (dùng lấy bbox lớn ở bước 4) ────────────
    _, step1_comps = _threshold_and_regionprops(gtl)

    # ── BƯỚC 2–3: Tăng ngưỡng 1% GTL đến khi đúng 3 vùng ───────────────────
    T, step    = gtl, 0.01 * gtl
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
        T         += step
        if T >= amp.max():
            break

    # ── FALLBACK A: Dùng step1_comps nếu vòng lặp trả về rỗng ──────────────
    if not best_comps and step1_comps:
        best_comps = step1_comps

    # ── FALLBACK B: Tìm đỉnh cực đại trực tiếp (loại bỏ vùng DC) ───────────
    # Áp dụng khi thuật toán Otsu thất bại hoàn toàn (DC quá sáng so với búp ±1)
    if not best_comps:
        print("    [WARN] Otsu thất bại, dùng fallback: tìm đỉnh biên độ trực tiếp...")
        Y0, X0    = np.ogrid[0:H, 0:W]
        search    = amp.copy()
        # Che DC (vùng tâm bán kính 20px)
        search[np.sqrt((X0 - cx)**2 + (Y0 - cy)**2) < 20] = 0

        def _pick_peak_as_comp(search_amp, suppress_r=35):
            idx     = np.argmax(search_amp)
            py_, px_ = np.unravel_index(idx, search_amp.shape)
            bw_peak = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(bw_peak, (int(px_), int(py_)), suppress_r // 2, 255, -1)
            _, _, stats_, centroids_ = cv2.connectedComponentsWithStats(bw_peak)
            comp = {
                'centroid': np.array([float(px_), float(py_)]),
                'bbox': (max(0, int(px_) - suppress_r // 2),
                         max(0, int(py_) - suppress_r // 2),
                         suppress_r, suppress_r),
                'area': int(np.pi * (suppress_r // 2) ** 2)
            }
            # Che lân cận để tìm peak tiếp theo
            y1s = max(0, py_ - suppress_r)
            y2s = min(H, py_ + suppress_r)
            x1s = max(0, px_ - suppress_r)
            x2s = min(W, px_ + suppress_r)
            search_amp[y1s:y2s, x1s:x2s] = 0
            return comp

        peak1 = _pick_peak_as_comp(search)
        peak2 = _pick_peak_as_comp(search)
        best_comps = [peak1, peak2]

    # ── Phân loại DC và sideband ─────────────────────────────────────────────
    # cv2 centroid = (x, y): [0]=x (cột), [1]=y (hàng)
    dc_comp   = min(best_comps,
                    key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)
    sidebands = [c for c in best_comps if c is not dc_comp]

    if not sidebands:
        sidebands = best_comps    # edge case: chỉ 1 vùng

    # ── BƯỚC 6: Chọn búp bên PHẢI (centroid_x > cx) ─────────────────────────
    right_sb = [c for c in sidebands if c['centroid'][0] > cx]
    if right_sb:
        target = max(right_sb, key=lambda c: c['centroid'][0])
    else:
        # Fallback: búp xa tâm nhất
        target = max(sidebands,
                     key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)

    target_cx, target_cy = target['centroid']   # cv2: (x, y)

    # ── BƯỚC 4: Lấy bbox từ bước 1 (rộng hơn, đúng theo paper) ─────────────
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

    # ── BƯỚC 5: Sub-pixel weighted centroid bán kính 7px ────────────────────
    Y_grid, X_grid = np.mgrid[0:H, 0:W]
    local_mask = np.sqrt((X_grid - px)**2 + (Y_grid - py)**2) <= 7
    weights    = amp[local_mask]
    total_w    = weights.sum()

    if total_w > 0:
        refined_cx = float(np.sum(X_grid[local_mask] * weights) / total_w)
        refined_cy = float(np.sum(Y_grid[local_mask] * weights) / total_w)
    else:
        refined_cx, refined_cy = float(px), float(py)

    # ── Tạo mặt nạ hình chữ nhật + Gaussian smoothing biên ─────────────────
    rx = w / 2.0 + margin
    ry = h / 2.0 + margin

    x1 = max(0, int(refined_cx - rx))
    x2 = min(W, int(refined_cx + rx))
    y1 = max(0, int(refined_cy - ry))
    y2 = min(H, int(refined_cy + ry))

    mask          = np.zeros((H, W), dtype=float)
    mask[y1:y2, x1:x2] = 1.0
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    F_filtered    = F * smoothed_mask

    # Trả về (row, col) = (y, x) cho nhất quán với phần còn lại của code
    return F_filtered, (refined_cy, refined_cx), smoothed_mask


def shift_spectrum_to_center(F_filtered, centroid):
    """
    Dịch nguyên pixel búp phổ +1 về tâm (H/2, W/2) bằng np.roll.
    Phần dư sub-pixel xử lý tiếp bởi apply_subpixel_shift.
    """
    H, W = F_filtered.shape
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    shift_y = H // 2 - cy
    shift_x = W // 2 - cx
    return np.roll(F_filtered, shift=(shift_y, shift_x), axis=(0, 1))


def apply_subpixel_shift(F_centered, k1, k2):
    """
    Bù phần dư sub-pixel (k1, k2), trả về trường sóng phức.

    IFFT(F)[y,x] * exp(+j2π*(k1*y/H + k2*x/W))  ↔  dịch phổ (k1, k2) pixel.
    Optimizer tìm (k1, k2) sao cho hai field khớp nhau.
    """
    H, W  = F_centered.shape
    y, x  = np.mgrid[0:H, 0:W]
    field = np.fft.ifft2(np.fft.ifftshift(F_centered))
    ramp  = np.exp(1j * 2 * np.pi * (k1 * y / H + k2 * x / W))
    return field * ramp


def phase_alignment_cost(k, field_1, F_centered_2):
    """
    Hàm mục tiêu: tối đa hoá |mean(U1 * conj(U2_shifted) / |...|)|.

    Khi pha đồng nhất → tất cả unit phasor cùng hướng → metric → 1.
    field_1 được tính TRƯỚC 1 lần, không tính lại trong mỗi bước lặp.
    """
    k1, k2     = k
    field_2    = apply_subpixel_shift(F_centered_2, k1, k2)
    diff       = field_1 * np.conj(field_2)
    phase_only = diff / (np.abs(diff) + 1e-12)
    return -np.abs(np.mean(phase_only))


def align_two_holograms(holo_1, holo_2):
    """
    Quy trình hoàn chỉnh: trích xuất búp +1, dịch về tâm, tối ưu sub-pixel.
    """
    F1, centroid_1, _ = extract_plus_one_order(holo_1)
    F2, centroid_2, _ = extract_plus_one_order(holo_2)

    F1_c = shift_spectrum_to_center(F1, centroid_1)
    F2_c = shift_spectrum_to_center(F2, centroid_2)

    # Tính field_1 MỘT LẦN — tránh IFFT lặp lại trong optimizer
    field_1 = np.fft.ifft2(np.fft.ifftshift(F1_c))

    # Khởi tạo từ phần dư sub-pixel sau làm tròn
    k1_init = (centroid_2[0] - round(centroid_2[0])) - (centroid_1[0] - round(centroid_1[0]))
    k2_init = (centroid_2[1] - round(centroid_2[1])) - (centroid_1[1] - round(centroid_1[1]))

    res = minimize(
        phase_alignment_cost,
        [k1_init, k2_init],
        args=(field_1, F2_c),
        method='Powell',
        options={'maxiter': 200, 'ftol': 1e-9}
    )
    k1_opt, k2_opt = res.x

    field_2_aligned = apply_subpixel_shift(F2_c, k1_opt, k2_opt)

    return {
        'field_1':             field_1,
        'field_2_aligned':     field_2_aligned,
        'k_shifts':            (k1_opt, k2_opt),
        'centroids':           (centroid_1, centroid_2),
        'optimization_result': res
    }
