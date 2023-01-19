import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
import numpy as np
######## >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """ menghitung search effectiveness metric score dengan 
      Discounted Cumulative Gain

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score DCG
  """
  # TODO
  score = 0
  for i in range(1, len(ranking)+1):

    d = 1/math.log2(i+1)
    r = ranking[i-1]                                                                                                                                                                                                   
    score += d*r
  return score

def prec(ranking):
  result = 0
  for i in range(1, len(ranking)+1):
    r = ranking[i-1]
    result += r

  result = result / len(ranking)
  return result

def ap(ranking):
  """ menghitung search effectiveness metric score dengan 
      Average Precision

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score AP
  """
  # TODO
  score = 0

  for i in range(1, len(ranking)+1):
    precision = prec(ranking[0:i])
    rank = ranking[i-1]
    score += precision*rank
  result = score/ranking.count(1)
  return result
    

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
          did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

  print("Hasil evaluasi TF-IDF terhadap 30 queries")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("AP score  =", sum(ap_scores) / len(ap_scores))

def eval_bm25(qrels, k1=1.6,b=1, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve_bm25(query, k = k, k1=k1, b=b):
          did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

  print(f"Hasil evaluasi BM25 terhadap 30 queries dengan k1={k1} dan b={b}")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("AP score  =", sum(ap_scores) / len(ap_scores))


def eval_tfidf(qrels, type, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    doc_collection = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])
      ltn = BSBI_instance.retrieve_tfidf(query, k = k)
      atn = BSBI_instance.retrieve_tfidf_var_atn(query, k = k)
      ntn = BSBI_instance.retrieve_tfidf_var_ntn(query, k = k)
      btn = BSBI_instance.retrieve_tfidf_var_btn(query, k = k)
      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      
      var_eval = {"ltn":ltn,
                "atn":atn,
                "ntn":ntn,
                "btn": btn}
      ranking = []
      for (score, doc) in var_eval[type]:
          did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
          doc_collection.append(doc)
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

    print(f"Hasil evaluasi TF-IDF variasi {type} terhadap 30 queries")
    print("RBP score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score =", sum(dcg_scores) / len(dcg_scores))
    print("AP score  =", sum(ap_scores) / len(ap_scores))
    return doc_collection

def retrieve_docid_doccontents(doc_collection):
    docs=[]
    for doc_id in doc_collection:
      doc_str = ""
      with open(doc_id) as f:
          for line in f:
              doc_str += re.sub(r"\s+", " ", str(line))
      docs.append((doc_id, doc_str))
    return docs

def eval_letor(qrels, doc_collection, query_file = "queries.txt"):
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

    docs = retrieve_docid_doccontents(doc_collection)
    ranking_letor=[]
 
    model = Letor("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")

    with open(query_file) as file:
      rbp_scores_letor = []
      dcg_scores_letor = []
      ap_scores_letor = []

      for line in file:
        parts = line.strip().split()
        qid = parts[0]
        query = " ".join(parts[1:])
        X_unseen = []
        for doc_id, doc in docs:
          X_unseen.append(model.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)

        scores = model.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(
                  did_scores, key=lambda tup: tup[1], reverse=True)
        for (did, score) in sorted_did_scores:
            did_1 = int(re.search(r'.*\.*\\.*\\(.*)\.txt', did).group(1))
            # print(did_1)
            ranking_letor.append(qrels[qid][did_1])


      rbp_scores_letor.append(rbp(ranking_letor))
      dcg_scores_letor.append(dcg(ranking_letor))
      ap_scores_letor.append(ap(ranking_letor))
    print(f"Hasil evaluasi Letor terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_letor) / len(rbp_scores_letor))
    print("DCG score =", sum(dcg_scores_letor) / len(dcg_scores_letor))
    print("AP score  =", sum(ap_scores_letor) / len(ap_scores_letor))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # eval(qrels)
  print("=============TF-IDF======================")
  docs = eval_tfidf(qrels, type='ltn')
  eval_letor(qrels,docs)
  eval_tfidf(qrels, type='atn')
  eval_tfidf(qrels, type="ntn")
  eval_tfidf(qrels, type="btn")
  print("=============BM25======================")
  eval_bm25(qrels)  
  eval_bm25(qrels, k1=1.3, b=0.75)
  eval_bm25(qrels, k1=2, b=0.5)