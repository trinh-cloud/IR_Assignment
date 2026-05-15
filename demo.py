import sys
from config import *
from core.indexer import Indexer
from core.medline_loader import IrDatasetsParser
from search.bm25_engine import BM25Ranker
from search.bm25f_engine import BM25FRanker
from search.rocchio import RocchioPRF

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

def main():
    print("="*60)
    print(" INTERACTIVE DEMO - HỆ THỐNG TRUY XUẤT THÔNG TIN Y KHOA (SciFact)")
    print("="*60)
    print("Đang khởi tạo các Indexer, vui lòng chờ trong giây lát...")
    
    parser = IrDatasetsParser(DATASET_NAME)
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
    
    print("\n✅ Khởi tạo thành công!")
    while True:
        print("\n" + "-"*60)
        query = input("Nhập câu truy vấn (Claim) của bạn (hoặc gõ 'exit' để thoát): ")
        if query.lower() == 'exit':
            break
            
        print("\nChọn cấu hình xếp hạng:")
        print("1. Baseline - BM25 Standard")
        print("2. Multi-field - BM25F")
        print("3. Query Expansion - BM25 + PRF (Rocchio)")
        opt = input("Nhập lựa chọn của bạn (1-3): ")
        
        results = []
        if opt == "1":
            print(f"👉 Đang tìm kiếm với BM25 Standard...")
            res = bm25.calculate_score(query)
            results = res[:5]
        elif opt == "2":
            print(f"👉 Đang tìm kiếm với BM25F...")
            res = bm25f.calculate_score(query)
            results = res[:5]
        elif opt == "3":
            print(f"👉 Đang phân tích phản hồi giả định (Rocchio)...")
            res_bm25 = bm25.calculate_score(query)
            expanded_query = rocchio.apply_feedback(query, res_bm25, ROCCHIO_TOP_DOCS, ROCCHIO_TOP_TERMS)
            print(f"   [Query mở rộng]: {expanded_query}")
            res = bm25.calculate_score(expanded_query)
            results = res[:5]
        else:
            print("Lựa chọn không hợp lệ!")
            continue
            
        print("\n🏆 Top 5 văn bản có liên quan nhất:")
        for idx, (doc_id, score) in enumerate(results):
            doc_info = docs_fields.get(doc_id, {})
            title = doc_info.get('title', 'N/A')
            abstract = doc_info.get('text', '')[:150] + "..." # Trích 150 ký tự đầu
            print(f"\n[{idx+1}] DocID: {doc_id} | Điểm số: {score:.4f}")
            print(f"   Titl: {title}")
            print(f"   Abst: {abstract}")

if __name__ == "__main__":
    main()