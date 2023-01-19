  # melalui qrels, kita akan buat sebuah dataset untuk training
# LambdaMART model dengan format
#
# [(query_text, document_text, relevance), ...]
#
# relevance awalnya bernilai 1, 2, 3 --> tidak perlu dinormalisasi
# biarkan saja integer (syarat dari library LightGBM untuk
# LambdaRank)
#
# relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant)
import random
import lightgbm as lgb
import numpy as np

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import lightgbm

class Document:
    def __init__(self, path:str):
        self.path = path
        self.documents = {}
        with open(self.path, encoding='utf-8') as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()

class Queries:
    def __init__(self, path:str):
        self.path = path
        self.queries = {}
        with open(self.path, encoding='utf-8') as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()

class Letor(Document, Queries):


    def __init__(self, path_doc, path_queries, path_qrels):
        self.documents = Document(path_doc).documents
        self.queries = Queries(path_queries).queries
        self.NUM_NEGATIVES = 1
        self.NUM_LATENT_TOPICS = 200
        self.q_docs_rels = {}
        self.qrels(path_qrels)
        self.group_qid_count = []
        self.dataset = []
        self.create_dataset()
        self.dictionary = Dictionary()
        self.bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics
        

    def qrels(self, path_qrels):
        with open(path_qrels) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rels:
                        self.q_docs_rels[q_id] = []
                    self.q_docs_rels[q_id].append((doc_id, int(rel)))

    def create_dataset(self):
        for q_id in self.q_docs_rels:
            docs_rels = self.q_docs_rels[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
    
    
    # test melihat representasi vector dari sebuah dokumen & query
    def vector_rep(self,text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
    
    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def split_dataset(self):
        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        # print(X.shape)
        # print(Y.shape)
        return X, Y
    
    def train_letor(self):
        X, Y = self.split_dataset()
        ranker = lightgbm.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)

        # di contoh kali ini, kita tidak menggunakan validation set
        # jika ada yang ingin menggunakan validation set, silakan saja
        ranker.fit(X, Y,
                group = self.group_qid_count,
                verbose = 10)
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        # print(len(X))
        # print(len(self.group_qid_count))
        return ranker

    def predict(self, arr):
        ranker = self.train_letor()
        scores = ranker.predict(arr)
        return scores

if __name__=="__main__":
    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
        ("D2", "study hard as your blood boils"), 
        ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
        ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
        ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]

    # sekedar pembanding, ada bocoran: D3 & D5 relevant, D1 & D4 partially relevant, D2 tidak relevan

    # bentuk ke format numpy array
    X_unseen = []
    model = Letor("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
    
    for doc_id, doc in docs:
        X_unseen.append(model.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)

    scores = model.predict(X_unseen)

    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
        print(did, score)

    

    

