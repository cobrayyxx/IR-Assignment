from bsbi import BSBIIndex
from compression import VBEPostings

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
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    # print("===========TF-IDF atn==============")
    # for (score, doc) in BSBI_instance.retrieve_tfidf_var_atn(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")
    # print("===========TF-IDF ntn==============")
    # for (score, doc) in BSBI_instance.retrieve_tfidf_var_ntn(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")
    # print("===========TF-IDF btn==============")
    # for (score, doc) in BSBI_instance.retrieve_tfidf_var_btn(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")
    # print("===========BM25==============")
    # for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
    #     print(f"{doc:30} {score:>.3f}")
    
    # print()