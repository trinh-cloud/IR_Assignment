import urllib.request
import json
import os
import zipfile

class SciFactLoader:
    """
    Module xử lý tải corpus trực tiếp từ HuggingFace/BEIR bằng custom script
    tránh lỗi timeout của ir_datasets.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self._docs = None
        self._queries = None
        self._qrels = None

    def _download_and_load(self):
        from datasets import load_dataset
        
        # Load corpus, queries, and qrels test from Hugging Face
        # BEIR data on HF is usually Tevatron/beir or BeIR/scifact
        
        # We will use the direct BeIR datasets which are faster
        print("[INFO] Downloading from Hugging Face...")
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
        queries = load_dataset("BeIR/scifact", "queries", split="queries")
        qrels = load_dataset("BeIR/scifact-qrels", split="test")

        self._docs = {doc['_id']: {"title": doc['title'], "text": doc['text']} for doc in corpus}
        self._queries = {q['_id']: q['text'] for q in queries}
        
        self._qrels = {}
        for qrel in qrels:
            if qrel['score'] > 0:
                qid = str(qrel['query-id'])
                doc_id = str(qrel['corpus-id'])
                if qid not in self._qrels:
                    self._qrels[qid] = set()
                self._qrels[qid].add(doc_id)

    def parse_docs(self):
        if self._docs is None: self._download_and_load()
        return {doc_id: f"{data['title']} {data['text']}" for doc_id, data in self._docs.items()}

    def parse_docs_fields(self):
        if self._docs is None: self._download_and_load()
        return self._docs

    def parse_queries(self):
        if self._queries is None: self._download_and_load()
        return self._queries

    def parse_qrels(self):
        if self._qrels is None: self._download_and_load()
        return self._qrels
