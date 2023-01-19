from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
from XGB_letor import LetorBoost
import numpy as np
import re
# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

if __name__ == '__main__':
    for query in queries:
        
        print("Query  : ", query)
        print("Results:")
        print("=========================BM25==========================")
        doc_ids=[]
        for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
            doc_ids.append(doc)
            print(f"{doc:30} {score:>.3f}")
        
        model = Letor("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
        model_boost = LetorBoost("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
        X_unseen = []
        docs=[]
        for id in doc_ids:
            doc_text = ""
            with open(id) as file:
                for line in file:
                    doc_text+=line
                doc_text = re.sub(r"\s+", " ", doc_text)
            docs.append((id, doc_text))



        for doc_id, doc in docs:
            X_unseen.append(model.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        scores = model.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
        print("\n")
        print("query        :", query)
        print("SERP/Ranking :")
        print("=========================LETOR LGBM==========================")
        for (did, score) in sorted_did_scores:
            print(f"{did:30} {score:>.3f}")
        print("\n\n")

        model_boost = LetorBoost("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
        scores_boost = model_boost.predict(X_unseen)
        did_scores_boost = [x for x in zip([did for (did, _) in docs], scores_boost)]
        sorted_did_scores_boost = sorted(did_scores_boost, key = lambda tup: tup[1], reverse = True)
        print("\n")
        print("query        :", query)
        print("SERP/Ranking :")
        print("=========================LETOR XGBoost + LSI Model===========================")
        for (did, score) in sorted_did_scores_boost:
            print(f"{did:30} {score:>.3f}")
        print("\n\n")
        print("===================================================================")
        print("===================================================================")
        print("===================================================================")

        print()