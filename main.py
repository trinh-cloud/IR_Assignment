from config import *
from core.indexer import Indexer
from core.dataset_loader import SciFactLoader
from core.evaluator import Evaluator
from search.bm25_engine import BM25Ranker
from search.bm25f_engine import BM25FRanker
from search.rocchio_PRF import RocchioPRF

def run_evaluation_pipeline():
    parser = SciFactLoader(DATASET_NAME)
    docs_standard = parser.parse_docs()
    docs_fields = parser.parse_docs_fields()
    queries = parser.parse_queries()
    qrels = parser.parse_qrels()
    
    # Build Indexers
    indexer_standard = Indexer()
    indexer_standard.build_index_from_dict(docs_standard)
    
    indexer_fields = Indexer()
    indexer_fields.build_field_index_from_dict(docs_fields)
    
    # Build Engines
    bm25 = BM25Ranker(indexer_standard, k1=BM25_K1, b=BM25_B)
    bm25f = BM25FRanker(
        indexer_fields, 
        k1=BM25F_K1, 
        b_title=BM25F_B_TITLE, 
        b_text=BM25F_B_TEXT, 
        w_title=BM25F_W_TITLE, 
        w_text=BM25F_W_TEXT
    )
    rocchio = RocchioPRF(indexer_standard)
    evaluator = Evaluator()
    
    configs = ["C1: BM25 Standard", "C2: BM25F", "C3: BM25 + PRF"]
    results_sum = {c: {"p1": 0.0, "p10": 0.0, "recall10": 0.0, "ap": 0.0} for c in configs}
    valid_count = 0
    
    for q_id, query_text in queries.items():
        if q_id not in qrels:
            continue
            
        actual_docs = qrels[q_id]
        valid_count += 1
        
        # Predict
        ret1 = [r[0] for r in bm25.calculate_score(query_text)[:100]]
        ret2 = [r[0] for r in bm25f.calculate_score(query_text)[:100]]
        
        exp_q = rocchio.apply_feedback(query_text, bm25.calculate_score(query_text), ROCCHIO_TOP_DOCS, ROCCHIO_TOP_TERMS)
        ret3 = [r[0] for r in bm25.calculate_score(exp_q)[:100]]
        
        predictions = {"C1: BM25 Standard": ret1, "C2: BM25F": ret2, "C3: BM25 + PRF": ret3}
        
        # Đánh giá chung
        for c, ret in predictions.items():
            p1, p10, r10, ap = evaluator.evaluate_list(ret, actual_docs)
            results_sum[c]["p1"] += p1
            results_sum[c]["p10"] += p10
            results_sum[c]["recall10"] += r10
            results_sum[c]["ap"] += ap

    # Output kết quả tính toán dưới dạng bảng Markdown
    print("| Configuration       | mP@1       | mP@10      | mRecall@10 | MAP        |")
    print("|---------------------|------------|------------|------------|------------|")
    
    for c in configs:
        
        mp1 = results_sum[c]["p1"] / valid_count
        mp10 = results_sum[c]["p10"] / valid_count
        mr10 = results_sum[c]["recall10"] / valid_count
        map_score = results_sum[c]["ap"] / valid_count
        print(f"| {c:<19} | {mp1:<10.4f} | {mp10:<10.4f} | {mr10:<10.4f} | {map_score:<10.4f} |")

if __name__ == "__main__":
    run_evaluation_pipeline()