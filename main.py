import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.core import align_two_holograms

def process_folder(raw_dir, output_dir):
    """
    Quét thư mục raw_dir, tìm các cặp ảnh '* (1).bmp' và '* (2).bmp',
    căn chỉnh và lưu kết quả vào output_dir.
    """
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Tìm tất cả các file có đuôi " (1).bmp" (không phân biệt hoa thường)
    pattern = os.path.join(raw_dir, "* (1).[bB][mM][pP]")
    file1_list = glob.glob(pattern)
    
    if not file1_list:
        print(f"Không tìm thấy file nào có định dạng '* (1).bmp' trong thư mục {raw_dir}")
        return
        
    print(f"Tìm thấy {len(file1_list)} ảnh (1).bmp. Đang tiến hành ghép cặp...")
    
    for file1_path in file1_list:
        base_name = os.path.basename(file1_path)
        
        # Dùng Regex để trích xuất phần tiền tố (prefix) trước chữ " (1).bmp"
        # Ví dụ: "78 (1).bmp" -> prefix = "78"
        match = re.match(r"(.*) \(1\)\.bmp", base_name, re.IGNORECASE)
        if not match:
            continue
            
        prefix = match.group(1)
        
        # Tạo tên file 2 dựa trên prefix
        file2_name = f"{prefix} (2).bmp"
        file2_path = os.path.join(raw_dir, file2_name)
        
        # Kiểm tra file 2 có tồn tại không
        if not os.path.exists(file2_path):
            print(f"[-] Cảnh báo: Thấy {base_name} nhưng thiếu {file2_name}. Bỏ qua.")
            continue
            
        print(f"\n[+] Đang xử lý cặp ảnh: '{prefix}'...")
        
        # Đọc ảnh (Grayscale)
        holo1 = cv2.imread(file1_path, cv2.IMREAD_GRAYSCALE)
        holo2 = cv2.imread(file2_path, cv2.IMREAD_GRAYSCALE)
        
        if holo1 is None or holo2 is None:
            print(f"[-] Lỗi khi đọc file ảnh cho cặp {prefix}")
            continue
            
        # Chuẩn hóa kiểu dữ liệu
        holo1 = holo1.astype(float)
        holo2 = holo2.astype(float)
        
        try:
            # Gọi thuật toán cốt lõi
            results = align_two_holograms(holo1, holo2)
            k_shifts = results['k_shifts']
            print(f"  -> Lượng dịch sub-pixel (k1, k2): {k_shifts[0]:.4f}, {k_shifts[1]:.4f}")
            
            # Tính toán phân bố pha
            field_1 = results['field_1']
            field_2 = results['field_2_aligned']
            
            phase1 = np.angle(field_1)
            phase2 = np.angle(field_2)
            phase_diff = np.angle(field_1 * np.conj(field_2))
            
            # Vẽ và xuất ảnh kết quả với 3 biểu đồ (Subplots)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            im0 = axes[0].imshow(phase1, cmap='gray')
            axes[0].set_title(f"Pha ảnh 1 - {prefix}")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(phase2, cmap='gray')
            axes[1].set_title(f"Pha ảnh 2 (đã dịch) - {prefix}")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(phase_diff, cmap='jet')
            axes[2].set_title(f"Hiệu pha - Mẫu {prefix}")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Tên file xuất ra: "78_phase_results.png"
            out_file = os.path.join(output_dir, f"{prefix}_phase_results.png")
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  -> Đã lưu kết quả đồ thị tại {out_file}")
            
        except Exception as e:
            print(f"[-] Lỗi khi tính toán thuật toán cho {prefix}: {e}")

if __name__ == "__main__":
    # Đặt tên folder raw và folder output ở đây
    # Ví dụ: Thư mục chứa ảnh gốc tên là "raw", folder xuất ra tên là "results"
    RAW_FOLDER = "raw"
    OUTPUT_FOLDER = "results"
    
    print(f"Bắt đầu quy trình với thư mục '{RAW_FOLDER}'...")
    process_folder(RAW_FOLDER, OUTPUT_FOLDER)
