from collections import defaultdict

class RocchioPRF:
    def __init__(self, indexer, alpha=1.0, beta=0.5):
        self.indexer = indexer
        self.alpha = alpha
        self.beta = beta
        
    def extract_relevant_terms(self, top_docs, top_k_terms):
        """Khai thác các từ khóa xuất hiện nhiều nhất trong Top K tài liệu giả định."""
        term_scores = defaultdict(float)
        
        for doc_id, _ in top_docs:
            for term, postings in self.indexer.inverted_index.items():
                if doc_id in postings:
                    term_scores[term] += postings[doc_id]
                    
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:top_k_terms]]

    def apply_feedback(self, query_text, initial_results, top_doc_count=3, top_term_count=2):
        """Mở rộng câu truy vấn gốc bằng các term lấy từ top documents giả định."""
        top_docs = initial_results[:top_doc_count]
        
        expanded_terms = self.extract_relevant_terms(top_docs, top_term_count)
        
        original_tokens = self.indexer.tokenize(query_text)
        original_query_processed = " ".join(original_tokens)
        added_query_processed = " ".join(expanded_terms)
        
        expanded_query = f"{original_query_processed} {added_query_processed}"
        return expanded_query