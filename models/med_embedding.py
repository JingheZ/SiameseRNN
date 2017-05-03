"""
Learn medical embedding
# extract the dxs, meds, procedures to learn a vector representation
# need tp think about how to structure it since it is two-level: pt level and visit level
# think about GloVe which uses co-occurrence counts rather than orders,
# There are orders between visits, and no orders within visits
# One solution: make each visit as a document to learn code vectors, do not consider patients
# The other solution is consider the visits by patients

"""