import ir_datasets

class SciFactLoader:
    """
    Nạp và xử lý dữ liệu SciFact qua ir_datasets (beir/scifact/test).
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)

    def parse_docs(self):
        return {doc.doc_id: f"{doc.title} {doc.text}" for doc in self.dataset.docs_iter()}

    def parse_docs_fields(self):
        return {doc.doc_id: {'title': doc.title, 'text': doc.text} for doc in self.dataset.docs_iter()}

    def parse_queries(self):
        return {q.query_id: q.text for q in self.dataset.queries_iter()}

    def parse_qrels(self):
        qrels = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.relevance > 0:
                qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
        return qrels