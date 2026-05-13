import math

class BM25FRanker:
    def __init__(self, indexer, k1=1.5, b_title=0.8, b_text=0.75, w_title=2.0, w_text=1.0):
        self.indexer = indexer
        self.k1 = k1
        self.B = {'title': b_title, 'text': b_text}
        self.W = {'title': w_title, 'text': w_text}
        self.fields = ['title', 'text']
        self.avg_lengths = self._calculate_avg_len()
        self.doc_count = len(self.indexer.doc_lengths)
        
    def _calculate_avg_len(self):
        total_len = {'title': 0, 'text': 0}
        for lengths in self.indexer.doc_lengths.values():
            if sum(lengths.values()) == 0:
                continue
            for f in self.fields:
                total_len[f] += lengths.get(f, 0)
        
        doc_count = len(self.indexer.doc_lengths)
        if doc_count == 0: return {'title': 1, 'text': 1}
        return {f: total_len[f] / doc_count for f in self.fields}

    def _idf(self, df):
        # Tính IDF sử dụng công thức chuẩn của BM25
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        
    def calculate_score(self, query):
        tokens = self.indexer.tokenize(query)
        scores = {doc_id: 0.0 for doc_id in self.indexer.doc_lengths.keys()}
        
        for token in tokens:
            if token not in self.indexer.inverted_index:
                continue
                
            posting_list = self.indexer.inverted_index[token]
            df = len(posting_list)
            idf = self._idf(df)
            
            for doc_id, tf_fields in posting_list.items():
                doc_lens = self.indexer.doc_lengths[doc_id]
                w_td = 0.0
                
                # Tính trọng số w_{t,d} cho tất cả các field
                for f in self.fields:
                    tf_t_d_c = tf_fields.get(f, 0)
                    len_d_c = doc_lens.get(f, 0)
                    avg_len_c = self.avg_lengths[f] or 1
                    
                    B_c = self.B[f]
                    W_c = self.W[f]
                    
                    # Chuẩn hóa tần suất xuất hiện với độ dài field
                    norm_tf = tf_t_d_c / (1.0 + B_c * ((len_d_c / avg_len_c) - 1.0))
                    w_td += W_c * norm_tf
                
                # Điểm BM25
                score = (w_td / (self.k1 + w_td)) * idf
                scores[doc_id] += score
                
        # Sắp xếp kết quả
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs
