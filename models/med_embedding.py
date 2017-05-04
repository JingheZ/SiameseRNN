"""
Learn medical embedding
# extract the dxs, meds, procedures to learn a vector representation
# need tp think about how to structure it since it is two-level: pt level and visit level
# think about GloVe which uses co-occurrence counts rather than orders,
# There are orders between visits, and no orders within visits
# One solution: make each visit as a document to learn code vectors, do not consider patients
# The other solution is consider the visits by patients

"""

import pickle
import pandas as pd
import os
import time
from gensim.models import Word2Vec#, Doc2Vec
# from glove import Glove: using glove, there is a decay for word distances


def process_doc_data(filenames):
    # load data
    with open('./data/visit_ranks.pickle', 'rb') as f:
        visit_ranks_reverse, _ = pickle.load(f)
    f.close()

    with open(filenames[0], 'rb') as f:
        data_dx = pickle.load(f)
    f.close()

    with open(filenames[1], 'rb') as f:
        data_med = pickle.load(f)
    f.close()

    with open(filenames[2], 'rb') as f:
        data_proc = pickle.load(f)
    f.close()

    vdoc = {}
    vdoc = get_items_in_visit(data_dx, visit_ranks_reverse, vdoc)
    vdoc = get_items_in_visit(data_med, visit_ranks_reverse, vdoc)
    vdoc = get_items_in_visit(data_proc, visit_ranks_reverse, vdoc)
    return vdoc


def get_items_in_visit(data, doc, vdoc):
    for i in data.index:
        pid = data['ptid'].loc[i]
        vid = data['vid'].loc[i]
        item = data['itemid'].loc[i]
        rk = doc[pid][vid]
        k = pid + '#' + str(rk)
        if not vdoc.__contains__(k):
            vdoc[k] = []
        vdoc[k].append(str(item))
    return vdoc


def get_doc_lengths(docs):
    lengths = []
    for i in docs:
        lengths.append(len(i))
    lengths = pd.Series(lengths)
    return lengths


# def med_embedding_model(docs, size, window, min_count, workers, sg, iter):
#     model = Word2Vec(docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)
#     return model


if __name__ == '__main__':
    # ============== get visit doc data for embedding =============
    filenames = ['./data/dxs_data_v2.pickle', './data/med_orders_v2.pickle', './data/proc_orders_v2.pickle']
    visit_docs = process_doc_data(filenames)  # 923707 visits (docs)
    with open('./data/visit_docs.pickle', 'wb') as f:
        pickle.dump(visit_docs, f)
    f.close()

    # ============== get the proc id and names ====================
    proc_names = pd.read_csv('./data/proc_id_name.csv', dtype=object)
    proc_names.index = proc_names['PROC_ID']
    del proc_names['PROC_ID']

    # ============ learn embedding for med codes ==========================
    # analyze lengths of the docs
    with open('./data/visit_docs.pickle', 'rb') as f:
        visit_docs = pickle.load(f)

    docs = list(visit_docs.values())
    lengths = get_doc_lengths(docs)
    lengths.describe()
    lengths.quantile(0.995) # 903; there are 4618 docs have longer weights than this

    size = 100
    window = 903
    min_count = 100
    workers = 28
    iter = 10
    sg = 1 # skip-gram:1; cbow: 0
    model_path = './results/w2v_size' + str(size) + '_window' + str(window) + '_sg' + str(sg)
    # if os.path.exists(model_path):
    #     model = Word2Vec.load(model_path)
    # else:
    a = time.time()
    model = Word2Vec(docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)
    model.save(model_path)
    b = time.time()
    print('training time (mins): %.3f' % ((b - a) / 60)) # 191 mins
    vocab = list(model.wv.vocab.keys())
    c = vocab[1]
    sims = model.most_similar(c)
    print(c)
    print(sims)


