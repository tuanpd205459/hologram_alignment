import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.core import align_two_holograms, detect_carrier, extract_plus_one_order


def plot_fft_debug(holo1, holo2, prefix, output_dir):
    """
    Vẽ phổ Fourier của cả 2 hologram, đánh dấu tâm DC và vị trí búp phổ +1
    được thuật toán tự động phát hiện. Giúp chẩn đoán lỗi chọn sai sideband.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"[DEBUG] Phổ Fourier và Sideband Detection — Mẫu: {prefix}", fontsize=13)

    for row, (holo, label) in enumerate([(holo1, "Ảnh 1"), (holo2, "Ảnh 2")]):
        H, W = holo.shape
        cx, cy = W // 2, H // 2

        # --- Cột 1: Hologram gốc ---
        axes[row, 0].imshow(holo, cmap='gray')
        axes[row, 0].set_title(f"{label} — Hologram gốc")

        # --- Cột 2: Phổ Fourier (log scale) ---
        F = np.fft.fftshift(np.fft.fft2(holo))
        I_log = np.log1p(np.abs(F) ** 2)
        I_norm = (I_log - I_log.min()) / (I_log.max() - I_log.min() + 1e-12)

        axes[row, 1].imshow(I_norm, cmap='hot')
        axes[row, 1].set_title(f"{label} — Phổ Fourier (log)\nCyan = trục tâm | Xanh lá = búp được chọn")
        axes[row, 1].axhline(cy, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[row, 1].axvline(cx, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)

        # Gọi hàm phát hiện sideband để lấy centroid thực tế
        try:
            kx, ky, rx, ry, abs_y, abs_x = detect_carrier(holo)
            axes[row, 1].plot(abs_x, abs_y, 'g+', markersize=16, markeredgewidth=2.5,
                              label=f"Búp chọn: ({abs_x:.1f}, {abs_y:.1f})")
            axes[row, 1].plot(cx, cy, 'b+', markersize=12, markeredgewidth=2,
                              label=f"DC: ({cx}, {cy})")
            
            import matplotlib.patches as patches
            rect = patches.Rectangle((abs_x - rx, abs_y - ry), 2*rx, 2*ry,
                                      linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--')
            axes[row, 1].add_patch(rect)
            
            axes[row, 1].legend(loc='upper right', fontsize=8)

            # In thông tin ra console
            side = "PHẢI" if kx > 0 else "TRÁI"
            vert = "DƯỚI" if ky > 0 else "TRÊN"
            print(f"    [{label}] Búp chọn tại pixel ({abs_x:.1f}, {abs_y:.1f}) "
                  f"→ nằm ở {side}-{vert} so với DC ({cx}, {cy}) | rx={rx:.1f}, ry={ry:.1f}")
        except Exception as e:
            axes[row, 1].set_title(f"{label} — Phổ Fourier (log)\n❌ Lỗi phát hiện: {e}")
            print(f"    [{label}] ❌ Không phát hiện được sideband: {e}")

        # --- Cột 3: Mặt nạ lọc (sigmoid mask) ---
        try:
            _, _, mask = extract_plus_one_order(holo)
            axes[row, 2].imshow(mask, cmap='viridis')
            axes[row, 2].set_title(f"{label} — Sigmoid Mask (bộ lọc)\nSáng = lọc mạnh, Tối = bỏ")
            axes[row, 2].axhline(cy, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
            axes[row, 2].axvline(cx, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
        except Exception:
            axes[row, 2].set_title(f"{label} — Mask: lỗi")

    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{prefix}_fft_debug.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> [DEBUG] Đã lưu ảnh phổ Fourier tại {out_file}")


def process_folder(raw_dir, output_dir, debug=True):
    """
    Quét thư mục raw_dir, tìm các cặp ảnh '* (1).bmp' và '* (2).bmp',
    căn chỉnh và lưu kết quả vào output_dir.

    debug=True: Xuất thêm ảnh phổ Fourier để chẩn đoán lỗi chọn sideband.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pattern = os.path.join(raw_dir, "* (1).[bB][mM][pP]")
    file1_list = glob.glob(pattern)

    if not file1_list:
        print(f"Không tìm thấy file nào có định dạng '* (1).bmp' trong thư mục {raw_dir}")
        return

    print(f"Tìm thấy {len(file1_list)} ảnh (1).bmp. Đang tiến hành ghép cặp...")

    for file1_path in file1_list:
        base_name = os.path.basename(file1_path)

        match = re.match(r"(.*) \(1\)\.bmp", base_name, re.IGNORECASE)
        if not match:
            continue

        prefix = match.group(1)
        file2_name = f"{prefix} (2).bmp"
        file2_path = os.path.join(raw_dir, file2_name)

        if not os.path.exists(file2_path):
            print(f"[-] Cảnh báo: Thấy {base_name} nhưng thiếu {file2_name}. Bỏ qua.")
            continue

        print(f"\n[+] Đang xử lý cặp ảnh: '{prefix}'...")

        # Đọc ảnh an toàn hỗ trợ unicode path
        holo1_raw = cv2.imdecode(np.fromfile(file1_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        holo2_raw = cv2.imdecode(np.fromfile(file2_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if holo1_raw is None or holo2_raw is None:
            print(f"[-] Lỗi khi đọc file ảnh cho cặp {prefix}")
            continue

        holo1 = holo1_raw.astype(float)
        holo2 = holo2_raw.astype(float)

        # --- DEBUG: Xuất ảnh phổ Fourier và vị trí sideband được chọn ---
        if debug:
            print(f"  [DEBUG] Đang phân tích phổ Fourier và sideband detection...")
            plot_fft_debug(holo1, holo2, prefix, output_dir)

        try:
            results = align_two_holograms(holo1, holo2)
            k1_shifts = results['k1_shifts']
            k2_shifts = results['k2_shifts']
            opt_result = results['optimization_result']

            print(f"  -> Lượng dịch ảnh 1 (k1): {k1_shifts[0]:.4f}, {k1_shifts[1]:.4f}")
            print(f"  -> Lượng dịch ảnh 2 (k2): {k2_shifts[0]:.4f}, {k2_shifts[1]:.4f}")
            print(f"  -> Lặp đến khi hội tụ: {opt_result.success} | "
                  f"Điểm số: {opt_result.fun:.6f} | Số vòng lặp: {opt_result.nit}")

            field_1 = results['field_1']
            field_2 = results['field_2_aligned']

            phase1     = np.angle(field_1)
            phase2     = np.angle(field_2)
            phase_diff = np.angle(field_1 * np.conj(field_2))
            amplitude1 = np.abs(field_1)
            amplitude2 = np.abs(field_2)

            # --- Ảnh kết quả chính (5 subplot) ---
            fig, axes = plt.subplots(2, 3, figsize=(18, 11))
            fig.suptitle(
                f"Kết quả Alignment — Mẫu: {prefix} | "
                f"Shift k1: ({k1_shifts[0]:.2f}, {k1_shifts[1]:.2f}), k2: ({k2_shifts[0]:.2f}, {k2_shifts[1]:.2f}) | "
                f"Hội tụ: {opt_result.success}",
                fontsize=12
            )

            im0 = axes[0, 0].imshow(phase1,     cmap='gray')
            axes[0, 0].set_title("Pha ảnh 1")
            fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

            im1 = axes[0, 1].imshow(phase2,     cmap='gray')
            axes[0, 1].set_title("Pha ảnh 2 (đã căn chỉnh)")
            fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            im2 = axes[0, 2].imshow(phase_diff, cmap='RdBu')
            axes[0, 2].set_title("Hiệu pha (nên phẳng nếu alignment đúng)")
            fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

            im3 = axes[1, 0].imshow(amplitude1, cmap='inferno')
            axes[1, 0].set_title("Biên độ ảnh 1")
            fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

            im4 = axes[1, 1].imshow(amplitude2, cmap='inferno')
            axes[1, 1].set_title("Biên độ ảnh 2 (đã căn chỉnh)")
            fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # Histogram hiệu pha để đánh giá mức độ hội tụ
            axes[1, 2].hist(phase_diff.ravel(), bins=100, color='steelblue', edgecolor='none')
            axes[1, 2].set_title(
                f"Phân phối hiệu pha\n"
                f"Mean={np.mean(phase_diff):.3f} rad | Std={np.std(phase_diff):.3f} rad\n"
                f"(Std nhỏ = alignment tốt)"
            )
            axes[1, 2].set_xlabel("Phase difference (rad)")
            axes[1, 2].set_ylabel("Số pixel")

            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{prefix}_phase_results.png")
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  -> Đã lưu kết quả tại {out_file}")

        except Exception as e:
            import traceback
            print(f"[-] Lỗi khi tính toán thuật toán cho {prefix}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    RAW_FOLDER    = "raw"
    OUTPUT_FOLDER = "results"

    # debug=True: xuất thêm ảnh FFT debug để kiểm tra sideband selection
    # Sau khi xác nhận sideband đúng, đặt debug=False để chạy nhanh hơn
    print(f"Bắt đầu quy trình với thư mục '{RAW_FOLDER}'...")
    process_folder(RAW_FOLDER, OUTPUT_FOLDER, debug=True)
