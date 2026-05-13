import os
import re
from nltk.stem import PorterStemmer

class Indexer:
    def __init__(self):
        # Inverted index: {term: {doc_id: term_frequency}}
        self.inverted_index = {}
        # Document lengths: {doc_id: length}
        self.doc_lengths = {}
        # Danh sách stopwords cơ bản
        self.stopwords = {"và", "là", "của", "trong", "cho", "các", "những", "một", "rất", "rằng", "thì", "mà", "có"}
        # Khởi tạo Porter Stemmer
        self.stemmer = PorterStemmer()
        
    def tokenize(self, text):
        """Tách từ, chuyển thành chữ thường, stem và lọc ký tự đặc biệt."""
        text = text.lower()
        # Lấy các từ chứa chữ cái và số
        tokens = re.findall(r'\b\w+\b', text)
        # Áp dụng Porter Stemmer
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens
        
    def build_index(self, data_dir):
        """Quét thư mục data, đọc file, tách từ và xây dựng Inverted Index."""
        if not os.path.exists(data_dir):
            return
            
        for filename in sorted(os.listdir(data_dir)):
            if not filename.endswith(".txt"): 
                continue
                
            filepath = os.path.join(data_dir, filename)
            doc_id = filename
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tokens = self.tokenize(content)
            # Lọc stopwords
            filtered_tokens = [t for t in tokens if t not in self.stopwords]
            
            self.doc_lengths[doc_id] = len(filtered_tokens)
            
            for token in filtered_tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = 0
                self.inverted_index[token][doc_id] += 1

    def build_index_from_dict(self, docs_dict):
        """Xây dựng Inverted Index trực tiếp từ Dictionary (Dùng cho SciFact)"""
        for doc_id, content in docs_dict.items():
            tokens = self.tokenize(content)
            filtered_tokens = [t for t in tokens if t not in self.stopwords]
            
            self.doc_lengths[doc_id] = len(filtered_tokens)
            
            for token in filtered_tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = 0
                self.inverted_index[token][doc_id] += 1

    def build_field_index_from_dict(self, docs_fields_dict):
        """
        Xây dựng Index hỗ trợ nhiều trường (cho BM25F).
        Cấu trúc inverted_index cho BM25F: {term: {doc_id: {'title': tf, 'text': tf}}}
        Cấu trúc doc_lengths cho BM25F: {doc_id: {'title': len, 'text': len}}
        """
        self.doc_lengths = {}
        self.inverted_index = {}
        
        for doc_id, fields in docs_fields_dict.items():
            self.doc_lengths[doc_id] = {}
            for field, content in fields.items():
                tokens = self.tokenize(content)
                filtered_tokens = [t for t in tokens if t not in self.stopwords]
                
                self.doc_lengths[doc_id][field] = len(filtered_tokens)
                
                for token in filtered_tokens:
                    if token not in self.inverted_index:
                        self.inverted_index[token] = {}
                    if doc_id not in self.inverted_index[token]:
                        self.inverted_index[token][doc_id] = {}
                    if field not in self.inverted_index[token][doc_id]:
                        self.inverted_index[token][doc_id][field] = 0
                    self.inverted_index[token][doc_id][field] += 1