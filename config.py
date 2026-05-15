# config.py
# Cấu hình trung tâm cho Hệ thống Truy xuất Thông tin (IR System)

# 1. Cấu hình Dữ liệu (Dataset)
DATASET_NAME = "beir/scifact/test"

# 2. Cấu hình Thuật toán BM25 (Baseline)
BM25_K1 = 1.5
BM25_B = 0.75

# 3. Cấu hình Thuật toán BM25F (Multi-field Scoring)
BM25F_K1 = 1.5
BM25F_B_TITLE = 0.8
BM25F_B_TEXT = 0.75
BM25F_W_TITLE = 2.0  # Trọng số ưu tiên Title (x2)
BM25F_W_TEXT = 1.0

# 4. Cấu hình thuật toán Rocchio (Pseudo-Relevance Feedback - PRF)
ROCCHIO_TOP_DOCS = 1       # K: Lấy top K tài liệu đầu tiên làm Bằng chứng giả định
ROCCHIO_TOP_TERMS = 3     # T: Mở rộng thêm T từ khóa liên quan nhất từ Top K
ROCCHIO_ALPHA = 1.0      # Trọng số giữ lại câu query gốc
ROCCHIO_BETA = 0.5         # Trọng số cộng thêm từ các tài liệu liên quan

# 5. Cấu hình Đánh giá Hệ thống
EVAL_TOP_K = 10            # Ngưỡng đánh giá độ chính xác mP@10, MAP@10
