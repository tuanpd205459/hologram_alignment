import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


def extract_plus_one_order(holo, sigma=5.0, min_area=30, margin=5.0):
    """
    Trích xuất phổ bậc +1 từ hologram off-axis.

    Thuật toán theo paper:
    'Automated Fourier space region-recognition filtering for off-axis DHM'

    Quy trình:
    (1) Tính biên độ phổ FFT thô (KHÔNG log) → Custom Otsu float (GTL).
    (2) Lưu ảnh nhị phân tại GTL (bước 1) → dùng lấy bbox lớn ở bước 3.
    (3) Tăng ngưỡng 1% GTL mỗi bước + morphology cv2 (close → open)
        → lặp cho đến khi đếm được đúng 3 vùng (DC, búp +1, búp -1).
    (4) Lấy bbox TỪ BƯỚC 1 (lớn hơn, đúng theo paper) cho cửa sổ lọc.
    (5) Tinh chỉnh sub-pixel centroid bằng weighted centroid bán kính 7px.
    (6) Chọn nhất quán búp nằm bên PHẢI tâm (centroid_x > center_x).

    Returns:
        F_filtered:    Phổ đã lọc (chưa dịch về tâm)
        centroid:      Tọa độ (centroid_y, centroid_x) = (row, col) của búp +1
        smoothed_mask: Mặt nạ Gaussian đã làm mượt
    """
    import cv2

    H, W = holo.shape
    cx, cy = W // 2, H // 2         # tâm DC trong ảnh phổ (pixel)

    # ── FFT và biên độ thô (raw amplitude, KHÔNG log) ───────────────────────
    F   = np.fft.fftshift(np.fft.fft2(holo))
    amp = np.abs(F)

    # ── Custom Otsu trên float data ──────────────────────────────────────────
    # graythresh của MATLAB hoạt động trực tiếp trên float, tránh mất dải động
    amp_flat           = amp.ravel()
    nbins              = 256
    hist, bin_edges    = np.histogram(amp_flat, bins=nbins)
    bin_centers        = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total     = hist.sum()
    sum_total = np.sum(bin_centers * hist)
    sum_bg, weight_bg = 0.0, 0
    max_variance, gtl = 0.0, bin_centers[0]

    for i in range(nbins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg   += bin_centers[i] * hist[i]
        mean_bg   = sum_bg / weight_bg
        mean_fg   = (sum_total - sum_bg) / weight_fg
        variance  = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            gtl          = bin_centers[i]

    # ── Kernel morphology ────────────────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))

    def _threshold_and_regionprops(threshold):
        """Phân ngưỡng → loại nhiễu nhỏ → close → fill → open → connected components."""
        bw = (amp > threshold).astype(np.uint8) * 255

        # Loại contour có diện tích quá nhỏ (nhiễu điểm)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_area = np.zeros_like(bw)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(bw_area, [cnt], -1, 255, -1)

        # Close: lấp lỗ hổng bên trong từng vùng
        bw_close = cv2.morphologyEx(bw_area, cv2.MORPH_CLOSE, kernel_close)

        # Fill contour sau close để tránh lỗ nội tâm
        contours_fill, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_filled = np.zeros_like(bw_close)
        cv2.drawContours(bw_filled, contours_fill, -1, 255, -1)

        # Open: tách các vùng dính nhau
        bw_open = cv2.morphologyEx(bw_filled, cv2.MORPH_OPEN, kernel_open)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bw_open)
        comps = []
        for idx in range(1, num_labels):     # bỏ nhãn 0 (nền)
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                comps.append({
                    'centroid': centroids[idx].copy(),    # cv2: (x, y)
                    'bbox': (stats[idx, cv2.CC_STAT_LEFT],
                             stats[idx, cv2.CC_STAT_TOP],
                             stats[idx, cv2.CC_STAT_WIDTH],
                             stats[idx, cv2.CC_STAT_HEIGHT]),
                    'area': int(area)
                })
        return bw_open, comps

    # ── BƯỚC 1: Lưu ảnh nhị phân tại GTL (dùng bbox ở bước 4) ──────────────
    _, step1_comps = _threshold_and_regionprops(gtl)

    # ── BƯỚC 2–3: Tăng ngưỡng 1% GTL đến khi đúng 3 vùng ───────────────────
    T          = gtl
    step       = 0.01 * gtl
    best_comps = None

    for _ in range(200):
        _, comps = _threshold_and_regionprops(T)

        if len(comps) == 3:
            best_comps = comps
            break
        elif len(comps) < 3:
            if best_comps is None:
                best_comps = comps if comps else step1_comps
            break

        best_comps = comps
        T         += step
        if T >= amp.max():
            break

    if not best_comps:
        best_comps = step1_comps

    # ── Phân loại DC và sideband ─────────────────────────────────────────────
    # DC = vùng gần tâm hình học nhất
    # cv2 centroid = (x, y), nên [0]=x (cột), [1]=y (hàng)
    dc_comp   = min(best_comps,
                    key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)
    sidebands = [c for c in best_comps if c is not dc_comp]

    if not sidebands:      # fallback nếu chỉ tìm được 1 vùng
        sidebands = best_comps

    # ── Chọn búp bên PHẢI (centroid_x > cx) ─────────────────────────────────
    right_sb = [c for c in sidebands if c['centroid'][0] > cx]
    if right_sb:
        target = max(right_sb, key=lambda c: c['centroid'][0])   # xa phải nhất
    else:
        # Fallback: búp xa tâm nhất bất kể bên nào
        target = max(sidebands,
                     key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)

    target_cx, target_cy = target['centroid']    # cv2: (x, y)

    # ── BƯỚC 4: Lấy bbox từ bước 1 (lớn hơn, đúng theo paper) ──────────────
    best_dist, best_step1 = float('inf'), None
    for c in step1_comps:
        d = np.sqrt((c['centroid'][0] - target_cx)**2 + (c['centroid'][1] - target_cy)**2)
        if d < best_dist:
            best_dist  = d
            best_step1 = c

    if best_step1 is not None:
        left, top, w, h = best_step1['bbox']
        px, py          = best_step1['centroid']   # cv2: (x, y)
    else:
        left, top, w, h = target['bbox']
        px, py          = target_cx, target_cy

    # ── BƯỚC 5: Tinh chỉnh sub-pixel — weighted centroid bán kính 7px ───────
    Y_grid, X_grid = np.mgrid[0:H, 0:W]
    local_mask = np.sqrt((X_grid - px)**2 + (Y_grid - py)**2) <= 7
    weights    = amp[local_mask]
    total_w    = weights.sum()

    if total_w > 0:
        refined_cx = float(np.sum(X_grid[local_mask] * weights) / total_w)
        refined_cy = float(np.sum(Y_grid[local_mask] * weights) / total_w)
    else:
        refined_cx, refined_cy = float(px), float(py)

    # ── Tạo mặt nạ hình chữ nhật từ bbox bước 1 + margin, Gaussian biên ─────
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

    # Trả về (row, col) = (y, x) để nhất quán với phần còn lại của code
    return F_filtered, (refined_cy, refined_cx), smoothed_mask


def shift_spectrum_to_center(F_filtered, centroid):
    """
    Dịch nguyên pixel búp phổ +1 về trung tâm (H/2, W/2) bằng np.roll.
    Phần dư sub-pixel được xử lý tiếp bởi apply_subpixel_shift.
    """
    H, W = F_filtered.shape
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    shift_y = H // 2 - cy
    shift_x = W // 2 - cx
    return np.roll(F_filtered, shift=(shift_y, shift_x), axis=(0, 1))


def apply_subpixel_shift(F_centered, k1, k2):
    """
    Bù phần dư sub-pixel (k1, k2) cho phổ F_centered, trả về trường sóng phức.

    Theo định lý dịch Fourier:
        IFFT(F)[y,x] * exp(+j2π*(k1*y/H + k2*x/W))  ↔  F[u-k1, v-k2]
    Nhân phase ramp vào miền không gian = dịch phổ về phải (k1, k2) pixel.
    Optimizer tìm (k1, k2) sao cho hai field khớp nhau.
    """
    H, W = F_centered.shape
    y, x  = np.mgrid[0:H, 0:W]
    field = np.fft.ifft2(np.fft.ifftshift(F_centered))
    ramp  = np.exp(1j * 2 * np.pi * (k1 * y / H + k2 * x / W))
    return field * ramp


def phase_alignment_cost(k, field_1, F_centered_2):
    """
    Hàm mục tiêu: tối đa hoá |mean(U1 * conj(U2_shifted) / |...|)|.

    Khi pha hai field đồng nhất → tất cả unit phasor cùng hướng → metric → 1.
    field_1 được tính TRƯỚC (1 lần) ở align_two_holograms, không tính lại ở đây.
    """
    k1, k2         = k
    field_2        = apply_subpixel_shift(F_centered_2, k1, k2)
    diff           = field_1 * np.conj(field_2)
    phase_only     = diff / (np.abs(diff) + 1e-12)
    return -np.abs(np.mean(phase_only))    # minimize → dùng dấu âm


def align_two_holograms(holo_1, holo_2):
    """
    Quy trình hoàn chỉnh: trích xuất búp +1, dịch về tâm, tối ưu sub-pixel.
    """
    # 1. Trích xuất và dịch nguyên pixel về tâm
    F1, centroid_1, mask_1 = extract_plus_one_order(holo_1)
    F2, centroid_2, mask_2 = extract_plus_one_order(holo_2)

    F1_c = shift_spectrum_to_center(F1, centroid_1)
    F2_c = shift_spectrum_to_center(F2, centroid_2)

    # 2. Tính field_1 MỘT LẦN — tránh tính lại IFFT(F1) trong mỗi bước lặp
    field_1 = np.fft.ifft2(np.fft.ifftshift(F1_c))

    # 3. Khởi tạo từ phần dư sub-pixel sau làm tròn centroid
    k1_init = (centroid_2[0] - round(centroid_2[0])) - (centroid_1[0] - round(centroid_1[0]))
    k2_init = (centroid_2[1] - round(centroid_2[1])) - (centroid_1[1] - round(centroid_1[1]))

    # 4. Tối ưu hóa Powell — truyền field_1 spatial, F2_c frequency
    res = minimize(
        phase_alignment_cost,
        [k1_init, k2_init],
        args=(field_1, F2_c),
        method='Powell',
        options={'maxiter': 200, 'ftol': 1e-9}
    )
    k1_opt, k2_opt = res.x

    # 5. Áp dụng shift tối ưu
    field_2_aligned = apply_subpixel_shift(F2_c, k1_opt, k2_opt)

    return {
        'field_1':            field_1,
        'field_2_aligned':    field_2_aligned,
        'k_shifts':           (k1_opt, k2_opt),
        'centroids':          (centroid_1, centroid_2),
        'optimization_result': res
    }
