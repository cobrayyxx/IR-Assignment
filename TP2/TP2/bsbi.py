import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import heapq
nltk.download('punkt')
class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.td_tf = None
        self.doc_length = {}


        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        # tambahan untuk mengambil self.doc_length 
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as idx:
            self.doc_length = idx.doc_length
            self.average_doc_length =  sum(self.doc_length.values()) / len(self.doc_length)
            

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        porter_stemmer = PorterStemmer()
        file_dir = os.path.join(self.data_dir, block_dir_relative)
        stop_words = set(stopwords.words('english'))
        td_tf = []
        result_list = []
        for file in os.listdir(file_dir):
            doc = os.path.join(file_dir,file)
            doc_id = self.doc_id_map[doc]
            with open(doc, 'r') as f:
                sentence = f.read()
                token_query = word_tokenize(sentence)
                clear_punc_token = [re.sub(r'[^\w\s]', ' ', token) for token in token_query]
                stem_token = [porter_stemmer.stem(token) for token in clear_punc_token]
                for token in stem_token:
                    if token not in stop_words:
                        tf_per_file = stem_token.count(token)
                        term_id = self.term_id_map[token]

                        td_tf.append((term_id, doc_id, tf_per_file))      
                        result_list.append((term_id, doc_id))
        self.td_tf = td_tf # buat di invert_write
        return result_list

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id, tf in self.td_tf:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add((doc_id,tf))
        for term_id in sorted(term_dict.keys()):
            postings_list = []
            tf_list = []
            doc_id_sorted = sorted(term_dict[term_id], key=lambda tup: tup[0])
            for doc_id, tf in doc_id_sorted:
                postings_list.append(doc_id)
                tf_list.append(tf)

            index.append(term_id, postings_list, tf_list) #intinya sort berdasar tupple elmen pertama/ docID



    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()
        scores = []

        porter_stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        token_query = word_tokenize(query)
        clear_punc_token = [re.sub(r'[^\w\s]', ' ', token) for token in token_query]
        stem_token = [porter_stemmer.stem(token) for token in clear_punc_token]
        result_query = [token for token in stem_token if token not in stop_words]

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as idx:
            for chunk_query in result_query:
                try:
                    pl, tf = idx.get_postings_list(self.term_id_map[chunk_query])
                except:
                    continue
                n = len(self.doc_length)
                df = len(pl)
                weight_term_query = math.log10(n/df)
                score = 0

                pairs = []
                for i in range(df):
                    if tf[i] > 0:
                        weight_term_doc = 1 + math.log10(tf[i])
                    else:
                        weight_term_doc = 0
                    score = weight_term_doc * weight_term_query
                    pairs.append((pl[i],score))
                scores = sorted_merge_posts_and_tfs(scores, pairs)

            scores_result = sorted(scores, key=lambda tup:tup[1], reverse=True)
            if len(scores_result) > k:
                top_k_doc = scores_result[:k]
            else:
                top_k_doc = scores_result
            for i in range(len(top_k_doc)):
                top_k_doc[i] = (top_k_doc[i][1], self.doc_id_map[top_k_doc[i][0]])
            return top_k_doc

    def retrieve_tfidf_var_ntn(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = tf(t, D)      jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()
        scores = []

        porter_stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        token_query = word_tokenize(query)
        clear_punc_token = [re.sub(r'[^\w\s]', ' ', token) for token in token_query]
        stem_token = [porter_stemmer.stem(token) for token in clear_punc_token]
        result_query = [token for token in stem_token if token not in stop_words]

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as idx:
            for chunk_query in result_query:
                try:
                    pl, tf = idx.get_postings_list(self.term_id_map[chunk_query])
                except:
                    continue
                n = len(self.doc_length)
                df = len(pl)
                weight_term_query = math.log10(n/df) #IDF
                score = 0
                pairs = []
                for i in range(df):
                    if tf[i] > 0:
                        weight_term_doc = tf[i] #tf
                    else:
                        weight_term_doc = 0
                    score = weight_term_doc * weight_term_query
                    pairs.append((pl[i],score))
                scores = sorted_merge_posts_and_tfs(scores, pairs)

            scores_result = sorted(scores, key=lambda tup:tup[1], reverse=True)
            if len(scores_result) > k:
                top_k_doc = scores_result[:k]
            else:
                top_k_doc = scores_result
            for i in range(len(top_k_doc)):
                top_k_doc[i] = (top_k_doc[i][1], self.doc_id_map[top_k_doc[i][0]])
            return top_k_doc
            
    def retrieve_tfidf_var_atn(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = 0.5+(0.5*tf(t,D)/max_tf)       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()
        scores = []

        porter_stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        token_query = word_tokenize(query)
        clear_punc_token = [re.sub(r'[^\w\s]', ' ', token) for token in token_query]
        stem_token = [porter_stemmer.stem(token) for token in clear_punc_token]
        result_query = [token for token in stem_token if token not in stop_words]

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as idx:
            for chunk_query in result_query:
                try:
                    pl, tf = idx.get_postings_list(self.term_id_map[chunk_query])
                except:
                    continue
                n = len(self.doc_length)
                df = len(pl)
                weight_term_query = math.log10(n/df) #IDF
                score = 0
                max_tf = max(tf)
                pairs = []
                for i in range(df):
                    if tf[i] > 0:
                        enum = 0.5*tf[i]
                        weight_term_doc = 0.5+(enum/max_tf) #tf
                    else:
                        weight_term_doc = 0
                    score = weight_term_doc * weight_term_query
                    pairs.append((pl[i],score))
                scores = sorted_merge_posts_and_tfs(scores, pairs)

            scores_result = sorted(scores, key=lambda tup:tup[1], reverse=True)
            if len(scores_result) > k:
                top_k_doc = scores_result[:k]
            else:
                top_k_doc = scores_result
            for i in range(len(top_k_doc)):
                top_k_doc[i] = (top_k_doc[i][1], self.doc_id_map[top_k_doc[i][0]])
            return top_k_doc   

    def retrieve_tfidf_var_btn(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = 1       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()
        scores = []

        porter_stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        token_query = word_tokenize(query)
        clear_punc_token = [re.sub(r'[^\w\s]', ' ', token) for token in token_query]
        stem_token = [porter_stemmer.stem(token) for token in clear_punc_token]
        result_query = [token for token in stem_token if token not in stop_words]

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as idx:
            for chunk_query in result_query:
                try:
                    pl, tf = idx.get_postings_list(self.term_id_map[chunk_query])
                except:
                    continue
                n = len(self.doc_length)
                df = len(pl)
                weight_term_query = math.log10(n/df) #IDF
                score = 0

                pairs = []
                for i in range(df):
                    if tf[i] > 0:
                        weight_term_doc = 1 #tf
                    else:
                        weight_term_doc = 0
                    score = weight_term_doc * weight_term_query
                    pairs.append((pl[i],score))
                scores = sorted_merge_posts_and_tfs(scores, pairs)

            scores_result = sorted(scores, key=lambda tup:tup[1], reverse=True)
            if len(scores_result) > k:
                top_k_doc = scores_result[:k]
            else:
                top_k_doc = scores_result
            for i in range(len(top_k_doc)):
                top_k_doc[i] = (top_k_doc[i][1], self.doc_id_map[top_k_doc[i][0]])
            return top_k_doc   

    def retrieve_bm25(self, query, k=10, k1 = 1.6, b= 1):
        """
        w(t, Q) = IDF = log (N / df(t))
        
        w(t,D) = Advance TF = (k1+1) * tf(t) / k1((1-b)+b*doc_length/avg_doc_length) + tf(t)

        """
        self.load()
        scores = []

        porter_stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        token_query = word_tokenize(query)
        clear_punc_token = [re.sub(r'[^\w\s]', ' ', token) for token in token_query]
        stem_token = [porter_stemmer.stem(token) for token in clear_punc_token]
        result_query = [token for token in stem_token if token not in stop_words]

        pairs = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as idx:
            for chunk_query in result_query:
                try:
                    pl, tf = idx.get_postings_list(self.term_id_map[chunk_query])
                except:
                    continue

                n = len(self.doc_length)
                df = len(pl)
                weight_term_query = math.log10(n/df)
                score = 0
                pairs = []
                for i in range(df):
                    if tf[i] > 0:
                        numerator = (1 + k1) * tf[i]
                        denumerator = k1*((1-b)+b*self.doc_length[i]/self.average_doc_length)+tf[i]
                        weight_term_doc = numerator/ denumerator 
                    else:
                        weight_term_doc = 0
                    score = weight_term_doc * weight_term_query
                    pairs.append((pl[i],score))
                scores = sorted_merge_posts_and_tfs(scores, pairs)

            scores_result = sorted(scores, key=lambda tup:tup[1], reverse=True)
            if len(scores_result) > k:
                top_k_doc = scores_result[:k]
            else:
                top_k_doc = scores_result
            for i in range(len(top_k_doc)):
                top_k_doc[i] = (top_k_doc[i][1], self.doc_id_map[top_k_doc[i][0]])
            return top_k_doc

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
