from collections import defaultdict
import math

class RocchioPRF:
    def __init__(self, indexer, alpha=1.0, beta=0.5):
        self.indexer = indexer
        self.alpha = alpha
        self.beta = beta
        
    def extract_relevant_terms(self, top_docs, top_k_terms, original_tokens):
        """Khai thác các từ khóa xuất hiện nhiều nhất trong Top K tài liệu giả định sử dụng phân tích TF-IDF."""
        term_scores = defaultdict(float)
        N = len(self.indexer.doc_lengths)
        
        # Tiêu chuẩn hóa các từ để chọn ra những từ có giá trị mang tính phân loại cao (dùng IDF kết hợp)
        for term, postings in self.indexer.inverted_index.items():
            doc_freq = len(postings)
            # Tính IDF tương tự như BM25 để trọng số hoá
            idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            if idf <= 0:
                continue
                
            for doc_id, _ in top_docs:
                if doc_id in postings:
                    tf = postings[doc_id]
                    # Cộng điểm = TF * IDF
                    term_scores[term] += tf * idf

        # Lọc bỏ các từ đã có trong câu truy vấn gốc
        for token in original_tokens:
            if token in term_scores:
                del term_scores[token]
                
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:top_k_terms]]

    def apply_feedback(self, query_text, initial_results, top_doc_count=1, top_term_count=2):
        """Mở rộng câu truy vấn gốc bằng các term lấy từ top documents giả định."""
        top_docs = initial_results[:top_doc_count]
        
        # Danh sách stopwords tiếng Anh để lọc riêng cho Rocchio PRF
        english_stopwords = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
            "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
            "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
        stemmed_stopwords = {self.indexer.stemmer.stem(w) for w in english_stopwords}
        
        original_tokens = self.indexer.tokenize(query_text)
        banned_tokens = set(original_tokens).union(self.indexer.stopwords).union(stemmed_stopwords)
        
        expanded_terms = self.extract_relevant_terms(top_docs, top_term_count, banned_tokens)

        # Lặp query gốc 3 lần để đảm bảo BM25 giữ độ ưu tiên rất cao cho câu hỏi ban đầu (Alpha lớn)
        boosted_original_query = " ".join([query_text] * 3) 
        added_query_processed = " ".join(expanded_terms)
        
        expanded_query = f"{boosted_original_query} {added_query_processed}"
        return expanded_query