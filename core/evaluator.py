class Evaluator:
    """
    Module Đánh giá hiệu năng Retrieval tinh gọn (Evaluation Suite).
    Hỗ trợ tính toán trực tiếp: P@1, P@10, Recall@10, và MAP.
    """
    
    @staticmethod
    def evaluate_list(retrieved_ids, actual_ids):
        """
        Tính toán 4 chỉ số trả về trong cùng một tuple.
        """
        if not actual_ids:
            return 0.0, 0.0, 0.0, 0.0
            
        actual_set = set(actual_ids)
        
        # --- P@1 ---
        p_at_1 = 1.0 if retrieved_ids and retrieved_ids[0] in actual_set else 0.0
        
        # --- P@10 và Recall@10 ---
        top_10 = retrieved_ids[:10]
        relevant_in_top_10 = sum(1 for doc_id in top_10 if doc_id in actual_set)
        p_at_10 = relevant_in_top_10 / 10.0
        recall_at_10 = relevant_in_top_10 / len(actual_set)
        
        # --- MAP (Average Precision cho truy vấn hiện tại) ---
        ap = 0.0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in actual_set:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        ap = ap / len(actual_set)
        
        return p_at_1, p_at_10, recall_at_10, ap

