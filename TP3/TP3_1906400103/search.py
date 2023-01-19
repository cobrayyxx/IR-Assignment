from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
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
for query in queries:
    
    print("Query  : ", query)
    print("Results:")
    print("===========TF-IDF ltn==============")
    doc_ids = []
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        doc_ids.append(doc)
        print(f"{doc:30} {score:>.3f}")


    model = Letor("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
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
    print("=========================LETOR==========================")
    for (did, score) in sorted_did_scores:
        print(f"{did:30} {score:>.3f}")
    print("\n\n")

    
    # print("===========TF-IDF atn==============")
    # for (score, doc) in BSBI_instance.retrieve_tfidf_var_atn(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")
    # print("===========TF-IDF ntn==============")
    # for (score, doc) in BSBI_instance.retrieve_tfidf_var_ntn(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")
    # print("===========TF-IDF btn==============")
    # for (score, doc) in BSBI_instance.retrieve_tfidf_var_btn(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")

print("===================================================================")
print("===================================================================")
print("===================================================================")

print()

for query in queries:
    
    print("Query  : ", query)
    print("Results:")
    print("=========================BM25==========================")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    
    model = Letor("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
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
    print("=========================LETOR==========================")
    for (did, score) in sorted_did_scores:
        print(f"{did:30} {score:>.3f}")
    print("\n\n")
