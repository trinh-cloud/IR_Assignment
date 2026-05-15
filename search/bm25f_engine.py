import math

class BM25FRanker:
    def __init__(self, indexer, k1=1.5, b_title=0.8, b_text=0.75, w_title=2.0, w_text=1.0):
        self.indexer = indexer
        self.k1 = k1
        self.b = {'title': b_title, 'text': b_text}
        self.w = {'title': w_title, 'text': w_text}
        self.fields = ['title', 'text']
        self.avg_lengths = self._get_avg_lengths()
        self.doc_count = len(self.indexer.doc_lengths)

    def _get_avg_lengths(self):
        avg_lengths = {}
        count = len(self.indexer.doc_lengths)
        total_title = 0
        total_text = 0
        for doc in self.indexer.doc_lengths.values():
            total_title += doc.get('title', 0)
            total_text += doc.get('text', 0)
        if count == 0:
            avg_lengths['title'] = 1
            avg_lengths['text'] = 1
        else:
            avg_lengths['title'] = total_title / count
            avg_lengths['text'] = total_text / count
        return avg_lengths

    def _idf(self, df):
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def calculate_score(self, query):
        tokens = self.indexer.tokenize(query)
        scores = {}
        for doc_id in self.indexer.doc_lengths.keys():
            scores[doc_id] = 0.0

        for token in tokens:
            if token not in self.indexer.inverted_index:
                continue

            posting = self.indexer.inverted_index[token]
            df = len(posting)
            idf = self._idf(df)

            for doc_id in posting:
                tf_fields = posting[doc_id]
                doc_lens = self.indexer.doc_lengths[doc_id]
                w_td = 0.0

                for field in self.fields:
                    tf = tf_fields.get(field, 0)
                    length = doc_lens.get(field, 0)
                    avg_length = self.avg_lengths[field]
                    b = self.b[field]
                    w = self.w[field]
                    div = 1.0 + b * (length / avg_length - 1.0)
                    if div == 0:
                        norm_tf = 0
                    else:
                        norm_tf = tf / div
                    w_td += w * norm_tf

                numerator = w_td
                denominator = self.k1 + w_td
                if denominator == 0:
                    score = 0
                else:
                    score = (numerator / denominator) * idf
                scores[doc_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
