import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
import numpy as np

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
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
    res = (1 - p) * score
    return res


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
    # DONE
    score = 0
    for i in range(1, len(ranking) + 1):
        score += float(ranking[i - 1] / math.log(i + 1, 2))
    return score


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
    # DONE
    score, R = 0, sum(ranking)
    for i in range(1, len(ranking) + 1):
        prec = sum(ranking[:i]) / len(ranking[:i])
        score += float((prec / R) * ranking[i - 1])
    return score

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """ memuat query relevance judgment (qrels)
        dalam format dictionary of dictionary
        qrels[query id][document id]

        dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
        relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
        Doc 10 tidak relevan dengan Q3.

    """
    qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels

# >>>>> EVALUASI !


def eval(qrels, query_file="queries.txt", k=1000):
    """
      loop ke semua 30 query, hitung score di setiap query,
      lalu hitung MEAN SCORE over those 30 queries.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    with open(query_file) as file:
        rbp_scores_tfidf, dcg_scores_tfidf, ap_scores_tfidf = [], [], []
        rbp_scores_bm25, dcg_scores_bm25, ap_scores_bm25 = [], [], []
        rbp_scores_boolean_tfidf, dcg_scores_boolean_tfidf, ap_scores_boolean_tfidf = [], [], []
        rbp_scores_tf_sublinear, dcg_scores_tf_sublinear, ap_scores_tf_sublinear = [], [], []
        rbp_scores_booleantf_probidf, dcg_scores_booleantf_probidf, ap_scores_booleantf_probidf = [], [], []
        rbp_scores_bm25_2, dcg_scores_bm25_2, ap_scores_bm25_2 = [], [], []
        rbp_scores_bm25_3, dcg_scores_bm25_3, ap_scores_bm25_3 = [], [], []
        rbp_scores_bm25_4, dcg_scores_bm25_4, ap_scores_bm25_4 = [], [], []
        rbp_scores_letor, dcg_scores_letor, ap_scores_letor = [], [], []
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
            # yang tertera di qrels
            model = Letor("nfcorpus/train.docs",
                          "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
            doc_collection = []
            X_unseen = []
            docs = []
            ranking_tfidf = []
            ranking_letor = []
            for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
                doc_collection.append(doc)
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                # print(did)
                ranking_tfidf.append(qrels[qid][did])

            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))
            for doc_id in doc_collection:
                doc_str = ""
                with open(doc_id) as f:
                    for line in f:
                        doc_str += re.sub(r"\s+", " ", str(line))
                docs.append((doc_id, doc_str))

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

            ranking_tf_sublinear = []
            for (score, doc) in BSBI_instance.retrieve_tfidf_var_atn(query, k=k):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_tf_sublinear.append(qrels[qid][did])
            rbp_scores_tf_sublinear.append(rbp(ranking_tf_sublinear))
            dcg_scores_tf_sublinear.append(dcg(ranking_tf_sublinear))
            ap_scores_tf_sublinear.append(ap(ranking_tf_sublinear))

            ranking_boolean_tfidf = []
            for (score, doc) in BSBI_instance.retrieve_tfidf_var_ntn(query, k=k):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_boolean_tfidf.append(qrels[qid][did])
            rbp_scores_boolean_tfidf.append(rbp(ranking_boolean_tfidf))
            dcg_scores_boolean_tfidf.append(dcg(ranking_boolean_tfidf))
            ap_scores_boolean_tfidf.append(ap(ranking_boolean_tfidf))

            ranking_booleantf_probidf = []
            for (score, doc) in BSBI_instance.retrieve_tfidf_var_btn(query, k=k):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_booleantf_probidf.append(qrels[qid][did])
            rbp_scores_booleantf_probidf.append(rbp(ranking_booleantf_probidf))
            dcg_scores_booleantf_probidf.append(dcg(ranking_booleantf_probidf))
            ap_scores_booleantf_probidf.append(ap(ranking_booleantf_probidf))

            ranking_bm25 = []
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_bm25.append(qrels[qid][did])
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            ranking_bm25_2 = []
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=1.3, b=0.7):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_bm25_2.append(qrels[qid][did])
            rbp_scores_bm25_2.append(rbp(ranking_bm25_2))
            dcg_scores_bm25_2.append(dcg(ranking_bm25_2))
            ap_scores_bm25_2.append(ap(ranking_bm25_2))

            ranking_bm25_3 = []
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=1.9, b=0.8):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_bm25_3.append(qrels[qid][did])
            rbp_scores_bm25_3.append(rbp(ranking_bm25_3))
            dcg_scores_bm25_3.append(dcg(ranking_bm25_3))
            ap_scores_bm25_3.append(ap(ranking_bm25_3))

            ranking_bm25_4 = []
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=1.4, b=0.72):
                did = int(re.search(r'.*\.*\\.*\\(.*)\.txt', doc).group(1))
                ranking_bm25_4.append(qrels[qid][did])
            rbp_scores_bm25_4.append(rbp(ranking_bm25_4))
            dcg_scores_bm25_4.append(dcg(ranking_bm25_4))
            ap_scores_bm25_4.append(ap(ranking_bm25_4))

    print("Hasil evaluasi TF-IDF terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
    print(f"{'-' * 65}")
    print("Hasil evaluasi LETOR terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_letor) / len(rbp_scores_letor))
    print("DCG score =", sum(dcg_scores_letor) / len(dcg_scores_letor))
    print("AP score  =", sum(ap_scores_letor) / len(ap_scores_letor))
    print(f"{'-' * 65}")
    print("Hasil evaluasi ATN TF-IDF terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_tf_sublinear) /
          len(rbp_scores_tf_sublinear))
    print("DCG score =", sum(dcg_scores_tf_sublinear) /
          len(dcg_scores_tf_sublinear))
    print("AP score  =", sum(ap_scores_tf_sublinear) /
          len(ap_scores_tf_sublinear))
    print(f"{'-' * 65}")
    print("Hasil evaluasi NTN TF-IDF terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_boolean_tfidf) /
          len(rbp_scores_boolean_tfidf))
    print("DCG score =", sum(dcg_scores_boolean_tfidf) /
          len(dcg_scores_boolean_tfidf))
    print("AP score  =", sum(ap_scores_boolean_tfidf) /
          len(ap_scores_boolean_tfidf))
    print(f"{'-' * 65}")
    print("Hasil evaluasi BTN TF Prob IDF terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_booleantf_probidf) /
          len(rbp_scores_booleantf_probidf))
    print("DCG score =", sum(dcg_scores_booleantf_probidf) /
          len(dcg_scores_booleantf_probidf))
    print("AP score  =", sum(ap_scores_booleantf_probidf) /
          len(ap_scores_booleantf_probidf))
    print(f"{'-' * 65}")
    print("Hasil evaluasi BM25 dengan k1=1.3 dan b=0.75 terhadap 30 queries")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))
    print(f"{'-' * 65}")


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    eval(qrels)
