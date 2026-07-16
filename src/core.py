import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

def extract_plus_one_order(holo, sigma=5.0):
    """
    Trích xuất phổ bậc +1 từ hologram off-axis.
    Búp phổ được chọn nhất quán là búp nằm ở bên PHẢI tâm tần số (centroid_x > center_x).

    Returns:
        F_filtered:  Phổ đã lọc (chưa dịch về tâm)
        centroid:    Tọa độ (centroid_y, centroid_x) của búp +1
        smoothed_mask: Mặt nạ Gaussian đã làm mượt
    """
    F = np.fft.fftshift(np.fft.fft2(holo))
    # Dùng abs^2 rồi log để tương thích với dải động cao của phổ Fourier
    I = np.abs(F) ** 2
    I_log = np.log1p(I)

    # Chuẩn hóa về [0, 1] cho ngưỡng Otsu nhất quán với MATLAB graythresh
    I_norm = (I_log - np.min(I_log)) / (np.max(I_log) - np.min(I_log))

    # Bước 1: Ngưỡng Otsu ban đầu (GTL)
    GTL = threshold_otsu(I_norm)
    thresh = GTL
    step = 0.01 * GTL

    regions = []
    bw = None

    # Bước 2: Tăng ngưỡng 1% GTL cho đến khi số vùng == 3 (DC, +1, -1)
    while thresh < 1.0:
        bw_current = I_norm > thresh
        labeled = label(bw_current)
        current_regions = regionprops(labeled)

        # Loại nhiễu rác < 2 pixel
        valid_regions = [r for r in current_regions if r.area >= 2]

        if len(valid_regions) == 3:
            regions = valid_regions
            bw = bw_current
            break
        elif len(valid_regions) < 3:
            # Ngưỡng đã vượt quá, giữ kết quả gần nhất trước đó
            if len(regions) == 0:
                regions = valid_regions
                bw = bw_current
            break

        regions = valid_regions
        bw = bw_current
        thresh += step

    # Fallback: nếu GTL thất bại hoàn toàn, lấy 3 vùng lớn nhất ở ngưỡng GTL gốc
    if len(regions) < 2:
        bw_current = I_norm > GTL
        labeled = label(bw_current)
        all_regions = regionprops(labeled)
        all_regions.sort(key=lambda r: r.area, reverse=True)
        regions = all_regions[:3]
        bw = bw_current

    H, W = holo.shape
    center_y, center_x = H // 2, W // 2

    def dist_from_center(r):
        return np.sqrt((r.centroid[0] - center_y)**2 + (r.centroid[1] - center_x)**2)

    # Sắp xếp từ xa đến gần tâm: 2 vùng đầu là sideband, vùng cuối là DC
    regions.sort(key=dist_from_center, reverse=True)

    # Lấy 2 vùng xa nhất (2 búp ±1), bỏ qua DC
    sidebands = regions[:2] if len(regions) >= 2 else regions

    # Chọn nhất quán: búp nằm bên PHẢI tâm (centroid_x > center_x)
    right_sidebands = [r for r in sidebands if r.centroid[1] > center_x]
    if right_sidebands:
        # Nếu cả 2 đều bên phải (ảnh bất thường), lấy cái xa bên phải nhất
        r_plus = max(right_sidebands, key=lambda r: r.centroid[1])
    else:
        # Fallback: lấy búp xa tâm nhất trong 2 sideband
        r_plus = sidebands[0]

    # Bước 3: Tạo mặt nạ từ bounding box và ảnh nhị phân, rồi làm mượt biên Gaussian
    min_row, min_col, max_row, max_col = r_plus.bbox
    centroid_y, centroid_x_r = r_plus.centroid

    mask = np.zeros((H, W), dtype=float)
    mask[min_row:max_row, min_col:max_col] = bw[min_row:max_row, min_col:max_col]

    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    F_filtered = F * smoothed_mask

    return F_filtered, (centroid_y, centroid_x_r), smoothed_mask


def shift_spectrum_to_center(F_filtered, centroid):
    """
    Dịch nguyên pixel búp phổ +1 về trung tâm (H/2, W/2) bằng np.roll.
    Phần dư sub-pixel sẽ được xử lý tiếp bởi apply_subpixel_shift.
    """
    H, W = F_filtered.shape
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    shift_y = H // 2 - cy
    shift_x = W // 2 - cx
    F_centered = np.roll(F_filtered, shift=(shift_y, shift_x), axis=(0, 1))
    return F_centered


def apply_subpixel_shift(F_centered, k1, k2):
    """
    Bù phần dư sub-pixel (k1, k2) cho phổ F_centered rồi trả về trường sóng phức.

    Theo định lý dịch Fourier:
        IFFT(F[u-k]) = ifft(F)[y,x] * exp(+j2π*(k*y/H))
    Nhân phase ramp (k1,k2) vào spatial field ↔ dịch phổ về phải (k1,k2) pixel.
    Optimizer sẽ tìm (k1,k2) tối ưu để hai field khớp nhau.

    Returns:
        shifted_field: Trường sóng phức [H, W] đã bù sub-pixel
    """
    H, W = F_centered.shape
    y, x = np.mgrid[0:H, 0:W]
    complex_field = np.fft.ifft2(np.fft.ifftshift(F_centered))
    phase_ramp = np.exp(1j * 2 * np.pi * (k1 * y / H + k2 * x / W))
    return complex_field * phase_ramp


def phase_alignment_cost(k, field_1, F_centered_2):
    """
    Hàm mục tiêu tối ưu hóa sub-pixel alignment.

    Nhận field_1 đã IFFT sẵn từ bên ngoài (KHÔNG tính lại trong mỗi bước lặp)
    và F_centered_2 để áp dụng phase ramp (k1, k2) trước khi so sánh.

    Metric: Maximise |mean( U1 * conj(U2_shifted) / |U1 * conj(U2_shifted)| )|
    Khi pha hai field đồng nhất (phẳng), tất cả unit phasor cùng hướng → metric → 1.
    """
    k1, k2 = k
    field_2_shifted = apply_subpixel_shift(F_centered_2, k1, k2)

    complex_diff = field_1 * np.conj(field_2_shifted)
    # Chia cho biên độ để chỉ giữ lại thông tin pha (unit phasor)
    phase_only = complex_diff / (np.abs(complex_diff) + 1e-12)

    # Giá trị metric càng gần 1 → hai trường pha càng đồng nhất
    metric = np.abs(np.mean(phase_only))
    return -metric   # scipy.minimize cần tìm giá trị nhỏ nhất


def align_two_holograms(holo_1, holo_2):
    """
    Quy trình hoàn chỉnh: Lọc phổ bậc +1 và tối ưu căn chỉnh sub-pixel giữa 2 ảnh.
    """
    # 1. Trích xuất và dịch nguyên pixel về tâm
    F1, centroid_1, mask_1 = extract_plus_one_order(holo_1)
    F2, centroid_2, mask_2 = extract_plus_one_order(holo_2)

    F1_c = shift_spectrum_to_center(F1, centroid_1)
    F2_c = shift_spectrum_to_center(F2, centroid_2)

    # 2. Tính trước field_1 MỘT LẦN DUY NHẤT ở đây.
    #    Tránh tính lại IFFT(F1) trong mỗi bước lặp của optimizer (lãng phí ~2x).
    field_1 = np.fft.ifft2(np.fft.ifftshift(F1_c))

    # 3. Khởi tạo (k1, k2) từ phần dư sub-pixel sau khi làm tròn centroid
    k1_init = (centroid_2[0] - round(centroid_2[0])) - (centroid_1[0] - round(centroid_1[0]))
    k2_init = (centroid_2[1] - round(centroid_2[1])) - (centroid_1[1] - round(centroid_1[1]))

    # 4. Tối ưu hóa: truyền field_1 spatial (đã IFFT) thay vì F_centered_1
    res = minimize(
        phase_alignment_cost,
        [k1_init, k2_init],
        args=(field_1, F2_c),       # field_1 = spatial, F2_c = frequency
        method='Powell',
        options={'maxiter': 200, 'ftol': 1e-9}
    )
    k1_opt, k2_opt = res.x

    # 5. Áp dụng shift tối ưu để lấy field_2 đã căn chỉnh
    field_2_aligned = apply_subpixel_shift(F2_c, k1_opt, k2_opt)

    return {
        'field_1': field_1,
        'field_2_aligned': field_2_aligned,
        'k_shifts': (k1_opt, k2_opt),
        'centroids': (centroid_1, centroid_2),
        'optimization_result': res
    }
