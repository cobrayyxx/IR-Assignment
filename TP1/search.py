from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["olahraga", "tumor", "hidup sehat"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    counter = 0
    for doc in BSBI_instance.retrieve(query):
        counter+=1
        print(doc)
    print(counter)
    print()