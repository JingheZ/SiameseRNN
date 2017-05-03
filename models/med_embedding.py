"""
Learn medical embedding
# extract the dxs, meds, procedures to learn a vector representation
# need tp think about how to structure it since it is two-level: pt level and visit level
# think about GloVe which uses co-occurrence counts rather than orders,
# There are orders between visits, and no orders within visits
# One solution: make each visit as a document to learn code vectors, do not consider patients
# The other solution is consider the visits by patients

"""

from gensim.models import Word2Vec, Doc2Vec
import pickle
import pandas as pd
import os


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




def med_embedding_model(docs, size, window, min_count, workers, sg, iter):
    docs = list(docs.values())
    model = Word2Vec(docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter)
    return model


if __name__ == '__main__':
    filenames = ['./data/dxs_data_v2.pickle', './data/med_orders_v2.pickle', './data/proc_orders_v2.pickle']
    visit_docs = process_doc_data(filenames)
    with open('./data/visit_docs.pickle', 'wb') as f:
        pickle.dump(visit_docs, f)
    f.close()

    visit_docs2 = {}
    for k, v in visit_docs.items():
        visit_docs2[k] = []
        for v0 in v:
            visit_docs2[k].append(str(v0))

    with open('./data/visit_docs.pickle', 'wb') as f:
        pickle.dump(visit_docs, f)
    f.close()

    # ============ learn embedding for dx ==========================
    with open('./data/visit_docs.pickle', 'rb') as f:
        doc = pickle.load(f)
    f.close()
    size = 150
    window = 5
    min_count = 50
    workers = 28
    iter = 20
    sg = 0 # skip-gram:1; cbow: 0
    model_path = 'w2v_dx_size' + str(size) + '_window' + str(window) + '_sg' + str(sg)
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = med_embedding_model(doc, size, window, min_count, workers, sg, iter)
        model.save(model_path)
    vocab = list(model.wv.vocab.keys())

