from .. import BaseSearch
import logging
import torch
import numpy as np
from typing import Dict
import heapq
from sklearn import svm
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Abstract class is BaseSearch
class SvmSearch(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.svm_setting = kwargs.get("svm_setting", True)
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
                    
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor).cpu()

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query

        corpus_embeddings = self.model.encode_corpus(
            corpus,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor
        ).cpu()

        for i, query_embedding in enumerate(tqdm(query_embeddings, desc='Running SVM for each query')):
            query_id = query_ids[i]

            x = np.vstack((query_embedding, corpus_embeddings))
            y = np.zeros(len(corpus) + 1)
            y[0] = 1
            clf = self._get_classifier()
            clf.fit(x, y)

            # Ranking by similarity score
            similarity_scores = clf.decision_function(x)[1:]
            sorted_indices_descending = np.argsort(-similarity_scores)
            sorted_similarity_scores = similarity_scores[sorted_indices_descending]
            for j, score in enumerate(sorted_similarity_scores[:top_k]):
                index = sorted_indices_descending[j]
                corpus_id = corpus_ids[index]
                heapq.heappush(result_heaps[query_id], (score, corpus_id))
        
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        
        return self.results
    
    def _get_classifier(self):
        svm_type = self.svm_setting['svm_type']
        if svm_type == 'LinearSVC':
            c = self.svm_setting['c'] if 'c' in self.svm_setting else 0.1
            clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=c, dual='auto')
        elif svm_type == 'SVC':
            kernel = self.svm_setting['kernel']
            gamma = self.svm_setting['gamma']
            clf = svm.SVC(kernel=kernel, gamma=gamma)
        return clf