import numpy as np
from scipy.sparse import csr_matrix

def getLexClassifier(pos_words,neg_words,vocab):
    clf = csr_matrix((np.hstack([-1*np.ones(len(neg_words)),np.ones(len(pos_words))]),
                      (np.zeros(len(neg_words)+len(pos_words)),
                       np.hstack([neg_words,pos_words]))),shape=[1,len(vocab)])
    return clf


def pmiPredictor(x,pos_lex,neg_lex):
    '''
    based on Turney (2002) and Mohammed et al (2013)
    '''
    def getPMI(x,lex1,lex2,smoothing=1e-3):
        docs = (x[:,lex1].sum(axis=1)>x[:,lex2].sum(axis=1))
        pmi = np.log(x.T.dot(docs)+smoothing).T - np.log(x.sum(axis=0)+.001) - np.log(docs.sum())
        return np.array(pmi)[0], np.array(docs.T)[0]

    pmi_pos,pos_docs = getPMI(x,pos_lex,neg_lex)
    pmi_neg,neg_docs = getPMI(x,neg_lex,pos_lex)

    term_filter = np.array((1*(x[pos_docs,:].sum(axis=0)>=5) | (x[neg_docs,:].sum(axis=0)>5)))[0]

    so = term_filter * (pmi_pos - pmi_neg)
    return x.dot(so)

