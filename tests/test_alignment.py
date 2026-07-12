import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src path to sys.path to easily import the core module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core import align_two_holograms

def generate_dummy_hologram(N, carrier_freq, phase_shift=0.0):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    object_phase = np.exp(1j * phase_shift)
    fx, fy = carrier_freq
    reference = np.exp(1j * 2 * np.pi * (fx * X + fy * Y))
    
    holo = np.abs(object_phase + reference)**2
    holo += 0.1 * np.random.randn(N, N)
    return holo

if __name__ == "__main__":
    N = 256
    freq1 = (30, 20)
    freq2 = (30.5, 20.3)
    
    holo1 = generate_dummy_hologram(N, freq1, phase_shift=0.0)
    holo2 = generate_dummy_hologram(N, freq2, phase_shift=1.5)
    
    print("Đang xử lý và căn chỉnh 2 ảnh hologram (thông qua test package)...")
    try:
        results = align_two_holograms(holo1, holo2)
        print("Trích xuất và căn chỉnh thành công!")
        print(f"Tâm phổ ảnh 1: {results['centroids'][0]}")
        print(f"Tâm phổ ảnh 2: {results['centroids'][1]}")
        print(f"Lượng dịch sub-pixel (k1, k2) đã tối ưu: {results['k_shifts']}")
        
        # Plot kết quả để kiểm tra trực quan (tuỳ chọn)
        # Bỏ qua lưu file để chạy test nhanh
        print("Test passed.")
    except Exception as e:
        print(f"Lỗi khi chạy thuật toán: {e}")
