from core.processor import TextProcessor

class Indexer:
    """
    Xây dựng chỉ mục (indexing), tiền xử lý văn bản bằng TextProcessor.
    """
    def __init__(self):
        self.inverted_index = {}
        self.doc_lengths = {}
        self.processor = TextProcessor()
        self.stopwords = self.processor.stopwords

    def tokenize(self, text):
        return self.processor.tokenize(text)

    def build_index_from_dict(self, docs_dict):
        for doc_id, content in docs_dict.items():
            filtered_tokens = self.processor.process_and_filter(content)
            
            self.doc_lengths[doc_id] = len(filtered_tokens)
            
            for token in filtered_tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = 0
                self.inverted_index[token][doc_id] += 1

    def build_field_index_from_dict(self, docs_fields_dict):
        self.doc_lengths = {}
        self.inverted_index = {}
        for doc_id, fields in docs_fields_dict.items():
            self.doc_lengths[doc_id] = {}
            for field, content in fields.items():
                filtered_tokens = self.processor.process_and_filter(content)
                
                self.doc_lengths[doc_id][field] = len(filtered_tokens)
                
                for token in filtered_tokens:
                    if token not in self.inverted_index:
                        self.inverted_index[token] = {}
                    if doc_id not in self.inverted_index[token]:
                        self.inverted_index[token][doc_id] = {}
                    if field not in self.inverted_index[token][doc_id]:
                        self.inverted_index[token][doc_id][field] = 0
                    self.inverted_index[token][doc_id][field] += 1