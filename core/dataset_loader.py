import ir_datasets

class SciFactLoader:
    """
    Module xử lý trực tiếp bộ dữ liệu thông qua thư viện ir_datasets.
    Tập trung tải cấu trúc dữ liệu của beir/scifact/test.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            self._dataset = ir_datasets.load(self.dataset_name)
        return self._dataset

    def parse_docs(self):
        """
        Lấy tài liệu dạng văn bản liên tiếp (gộp Title và Text).
        Trả ra: {doc_id: "title text"}
        """
        dataset = self._get_dataset()
        docs = {}
        for doc in dataset.docs_iter():
            # Scifact document trong ir_datasets có trường doc_id, title, text
            docs[doc.doc_id] = f"{doc.title} {doc.text}"
        return docs

    def parse_docs_fields(self):
        """
        Lấy tài liệu nhưng chia thành các trường để dùng cho BM25F.
        Trả ra: {doc_id: {'title': str, 'text': str}}
        """
        dataset = self._get_dataset()
        docs = {}
        for doc in dataset.docs_iter():
            docs[doc.doc_id] = {
                'title': doc.title,
                'text': doc.text
            }
        return docs

    def parse_queries(self):
        """
        Lấy danh sách các câu truy vấn.
        Trả ra: {query_id: text}
        """
        dataset = self._get_dataset()
        queries = {}
        for query in dataset.queries_iter():
            queries[query.query_id] = query.text
        return queries

    def parse_qrels(self):
        """
        Lấy Ground-truth để đánh giá (chỉ lấy các tài liệu có relevance > 0).
        Trả ra: {query_id: set([doc1, doc2])}
        """
        dataset = self._get_dataset()
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = set()
                qrels[qrel.query_id].add(qrel.doc_id)
        return qrels