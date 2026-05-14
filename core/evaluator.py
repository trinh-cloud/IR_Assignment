class Evaluator:
    """
    Đánh giá hiệu năng Retrieval: P@1, P@10, Recall@10, MAP.
    """
    @staticmethod
    def evaluate_list(retrieved_ids, actual_ids):
        if not actual_ids:
            return 0.0, 0.0, 0.0, 0.0
        actual_set = set(actual_ids)
        p_at_1 = float(retrieved_ids and retrieved_ids[0] in actual_set)
        relevant_in_top_10 = sum(doc_id in actual_set for doc_id in retrieved_ids[:10])
        p_at_10 = relevant_in_top_10 / 10.0
        recall_at_10 = relevant_in_top_10 / len(actual_set)
        ap = 0.0
        rels = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in actual_set:
                rels += 1
                ap += rels / (i + 1)
        ap /= len(actual_set)
        return p_at_1, p_at_10, recall_at_10, ap
   
