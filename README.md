# Hệ Thống Truy Xuất Thông Tin - Đánh giá SciFact (IR System)

Dự án này là hệ thống Information Retrieval quy mô nhỏ hỗ trợ so sánh, đánh giá hiệu năng các kỹ thuật tìm kiếm trên tập tài liệu dạng Fact-checking y khoa: **SciFact** (Từ thư viện `ir_datasets`).

## 1. Cấu Trúc Hệ Thống (The Architecture)
Bám sát kiến trúc 4 lớp chuẩn mực:

- **Giai đoạn 1: Lớp Tiền xử lý (Preprocessing):** Được đóng gói gọn gàng bên trong `core/indexer.py` xử lý Tokenization với Regex, Lowercasing, xóa Stopwords và thuật toán rút gọn từ gốc NLTK **PorterStemmer**. Điều này đảm bảo triệt tiêu các rào cản về hình thái từ (Morphology).

- **Giai đoạn 2: Lớp Data & Indexing:** Quản lý bởi `Indexer` và dataset loader. Đặc biệt triển khai Inverted Index cho văn bản trơn và Inverted Index đa phân vùng (Multi-field: Title, Abstract) để lưu trữ `term_frequency` và `doc_length` độc lập theo từng vùng. 
- **Giai đoạn 3: Lớp Thuật toán Xếp hạng (Hybrid Ranking):** Gồm bộ ba mô hình được triển khai trong folder `search/`:
  - `BM25 Standard`: Baseline đo điểm tổng hợp trên đoạn text dài.
  - `BM25F`: Nhận diện rõ Title mang giá trị thông tin cao hơn Abstract. Đã cấu hình nhân đôi trọng số (`W_TITLE = 2.0`).
  - `BM25 + PRF (Rocchio)`: Truy xuất lần 1 (BM25) lấy Pseudo-Docs, trích xuất lượng terms mở rộng và đắp ngược vào câu Query bằng trọng số Rocchio Feedback.
- **Giai đoạn 4: Lớp Đánh giá (Evaluation Suite):** `core/evaluator.py` hỗ trợ tính trực tiếp Precision@1, Precision@10, Recall@10 và Mean Average Precision (MAP).

## 2. Kết quả Thực nghiệm trên Dataset *SciFact*
Kết quả đối chiếu 3 cấu hình thuật toán trên 300 claims của tập Benchmarking:
| Cấu hình mô hình          | mP@1       | mP@10      | mRecall@10 | MAP        |
|---------------------------|------------|------------|------------|------------|
| C1: BM25 Standard         | 0.5567     | 0.0893     | 0.8122     | 0.6469     |
| C2: BM25F                 | 0.5733     | 0.0893     | 0.8122     | 0.6513     |
| C3: BM25 + PRF            | 0.3900     | 0.0883     | 0.8074     | 0.5499     |

**Nhận xét (Có giá trị cho báo cáo):**
1. Mô hình **BM25F** cho kết quả tốt nhất. Khẳng định chân lý trong IR khoa học: "Thông tin ở Tiêu đề đóng vai trò bao quát giá trị nhất". Bằng cách đặt $W_{title}=2$, thuật toán đã kéo những tài liệu chạm đúng keyword trên tiêu đề lên rank 1 (P@1 cao nhất đạt 0.5733), dẫn đến MAP đạt đỉnh (0.6513).


2. Sự thụt giảm của **Rocchio PRF** trên SciFact: SciFact là bộ Fact-Checking truy vấn bằng các "Câu khẳng định" ngắn gọn, xúc tích với các Entity y học đặc biệt (vd: Covid-19, Incubation period). Hậu quả của việc dùng PRF lấy thêm từ ở các tài liệu top đầu đắp vào là gây ra hiện tượng **Topic-Drift (Trôi dạt lãng đề)**, khiến Query mới bị loãng và giảm tỉ lệ hit đúng tài liệu đích tại top 1 (làm cho P@1 và MAP tụt đáng kể).
3. Tỉ lệ **Recall@10** rất cao (trên 81%): Vì mỗi claim thường chỉ có 1-2 tài liệu đúng, việc hệ thống của chúng ta quét được hơn 81% tổng lượng tài liệu đúng chỉ bằng Top 10 kết quả hiển thị cho thấy sức mạnh của PorterStemmer và BM25. Điểm số P@10 loanh quanh 0.089 cũng là giới hạn lý thuyết tối đa (~1/10) trên tập dữ liệu dạng "Sparse Judgments" này.

## 3. Quản Lý Tham Số Tại `config.py`
Tài liệu cung cấp tệp hằng số config thống nhất (Single source of truth) giúp quản lý mọi Hyper-parameters như Tuning K1 cho thuật toán (1.5), $B$ parameter, Trọng số phân cụm (Weights), Các trọng số Rocchio alpha/beta. Tất cả giúp mã nguồn Clean và chuyên nghiệp.

## 4. Hướng dẫn sử dụng
* Yêu cầu NLTK và `ir_datasets`:
  ```bash
  pip install nltk ir_datasets tqdm
  # (Vào python shell gõ import nltk; nltk.download('punkt')) nếu thấy thiếu punkt/porter
  ```
* Chạy Auto-Evaluation Pipeline để tái tạo bảng báo cáo:
  ```bash
  python main.py
  ```
* Chạy Demo Application (Tương tác Console gõ câu Query & Trực tiếp kiểm tra Rank):
  ```bash
  python demo.py
  ```