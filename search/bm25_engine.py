import math

class BM25Ranker:
    def __init__(self, indexer, k1=1.2, b=0.75):
        self.indexer = indexer
        self.k1 = k1
        self.b = b
        self.N = len(indexer.doc_lengths)
        if self.N > 0:
            self.avgdl = sum(indexer.doc_lengths.values()) / self.N
        else:
            self.avgdl = 0

    def calculate_score(self, query):
        tokens = self.indexer.tokenize(query)
        query_tokens = [t for t in tokens if t not in self.indexer.stopwords]
        
        doc_scores = {doc_id: 0.0 for doc_id in self.indexer.doc_lengths}
        
        for token in query_tokens:
            if token not in self.indexer.inverted_index:
                continue
                
            doc_freq = len(self.indexer.inverted_index[token])
            idf = math.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            
            for doc_id, tf in self.indexer.inverted_index[token].items():
                doc_len = self.indexer.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                term_score = idf * (numerator / denominator)
                doc_scores[doc_id] += term_score
                
        sorted_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores
