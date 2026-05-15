import streamlit as st
import sys
from config import *
from core.indexer import Indexer
from core.dataset_loader import SciFactLoader
from search.bm25_engine import BM25Ranker
from search.bm25f_engine import BM25FRanker
from search.rocchio_PRF import RocchioPRF

# Hàm để ẩn output khi khởi tạo, tránh in ra console quá nhiều
def clean_log_execution(func, *args, **kwargs):
    old_stdout = sys.stdout
    class DummyStream:
        def write(self, data): pass
        def flush(self): pass
    sys.stdout = DummyStream()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result

# Khởi tạo dữ liệu và mô hình dùng cache để tránh việc tải lại mỗi lần click
@st.cache_resource
def load_models_and_data():
    parser = SciFactLoader(DATASET_NAME)
    docs_standard = clean_log_execution(parser.parse_docs)
    docs_fields = clean_log_execution(parser.parse_docs_fields)
    
    indexer_standard = Indexer()
    clean_log_execution(indexer_standard.build_index_from_dict, docs_standard)
    
    indexer_fields = Indexer()
    clean_log_execution(indexer_fields.build_field_index_from_dict, docs_fields)
    
    bm25 = BM25Ranker(indexer_standard, k1=BM25_K1, b=BM25_B)
    bm25f = BM25FRanker(
        indexer_fields, 
        k1=BM25F_K1, 
        b_title=BM25F_B_TITLE, 
        b_text=BM25F_B_TEXT, 
        w_title=BM25F_W_TITLE, 
        w_text=BM25F_W_TEXT
    )
    rocchio = RocchioPRF(indexer_standard, alpha=ROCCHIO_ALPHA, beta=ROCCHIO_BETA)
    
    return docs_fields, bm25, bm25f, rocchio

def main():
    st.set_page_config(page_title="SciFact Search Demo", page_icon="🔍")
    st.title("🔍 Interactive Demo - Truy Xuất Thông Tin Y Khoa")
    st.write("Hệ thống tìm kiếm thông tin trên tập dữ liệu SciFact.")

    with st.spinner("Đang khởi tạo các Indexer, vui lòng chờ trong giây lát..."):
        docs_fields, bm25, bm25f, rocchio = load_models_and_data()

    # Nhập query
    query = st.text_input("Nhập câu truy vấn (Claim) của bạn:", placeholder="Gõ câu truy vấn vào đây...")
    
    # Lựa chọn thuật toán
    option = st.selectbox(
        "Chọn cấu hình xếp hạng:",
        (
            "1. Baseline - BM25 Standard",
            "2. Multi-field - BM25F",
            "3. Query Expansion - BM25 + PRF (Rocchio)"
        )
    )

    if st.button("Tìm kiếm") or query:
        if not query.strip():
            st.warning("Vui lòng nhập câu truy vấn!")
            return
            
        st.write(f"Đang tìm kiếm cho: **{query}**")
        results = []
        
        with st.spinner("Đang xử lý thuật toán..."):
            if option.startswith("1"):
                res = bm25.calculate_score(query)
                results = res[:10]  # Lấy top 10 cho web
            elif option.startswith("2"):
                res = bm25f.calculate_score(query)
                results = res[:10]
            elif option.startswith("3"):
                res_bm25 = bm25.calculate_score(query)
                expanded_query = rocchio.apply_feedback(query, res_bm25, ROCCHIO_TOP_DOCS, ROCCHIO_TOP_TERMS)
                
                # Hiển thị cho mượt mà, cắt các từ bị lặp do cơ chế boost điểm (Alpha)
                display_query = f"{query} " + " ".join(expanded_query.replace(query, "").split())
                st.info(f"**Query mở rộng:** {display_query}")
                
                res = bm25.calculate_score(expanded_query)
                results = res[:10]

        if not results:
            st.error("Không tìm thấy kết quả phù hợp.")
            return

        st.subheader("🏆 Top các văn bản có liên quan nhất")
        for idx, (doc_id, score) in enumerate(results):
            doc_info = docs_fields.get(doc_id, {})
            title = doc_info.get('title', 'N/A')
            abstract_full = doc_info.get('text', '')
            abstract_short = abstract_full[:200] + "..." if len(abstract_full) > 200 else abstract_full
            
            with st.expander(f"[{idx+1}] DocID: {doc_id} | Điểm số: {score:.4f} | {title}"):
                st.write(f"**Title:** {title}")
                st.write(f"**Abstract:** {abstract_full}")

if __name__ == "__main__":
    main()