import numpy as np
import sys
from scipy.special import gammaln

''' 
Methods for computing various parameters of the model, and statistics that relate to these parameters
'''

def estimateDCMFromMOM(x):
    '''
    A method-of-moments estimator for the concentration parameter of the DCM
    '''
    n_ii_diag = np.array(x.power(2).sum(axis=0))[0]
    Nsq_sum = (np.array(x.sum(axis=1).T)[0]**2).sum()
    n_i = np.array(x.sum(axis=0))[0]

    e_mu = getEMu(x)
    N_sum = n_i.sum()
    numerator = ((1 - e_mu) * e_mu * (e_mu * Nsq_sum - n_ii_diag)).sum()
    denominator = (((1-e_mu)*e_mu*(n_ii_diag - e_mu**2 * (Nsq_sum - N_sum) - e_mu * N_sum))).sum()
    e_c = numerator / denominator
    return e_mu, e_c

def getEMu(x):
    return np.array(x.sum(axis=0) / (0.+x.sum()))[0]    

def computeS(x,c = sys.float_info.max):
    '''
    s is a constant term in the expected co-occurrence counts for the Bayesian model.

    In the non-Bayesian model, $c \to \infty$
    '''
    N_t = np.array(x.sum(axis=1)).flatten()
    s = np.inner(N_t,N_t - (N_t + c)/(1+c))
    return s

def estimateK(x,pos_words,neg_words):
    '''
    This estimates a single predictiveness parameter
    '''
    cc_pn, e_cc_pn = getCoCountsTwoLex(x,pos_words,neg_words)
    return np.sqrt(e_cc_pn.sum() / cc_pn.sum() - 1)

def getCoCountsTwoLex(p_x,p_lex1,p_lex2,Bayesian=False):
    filter_to_x_shape = lambda lex : [i for i in lex if i < p_x.shape[1]]
    p_lex1 = filter_to_x_shape(p_lex1)
    p_lex2 = filter_to_x_shape(p_lex2)
    x1 = p_x[:,p_lex1]
    x2 = p_x[:,p_lex2]

    co_counts = (x1.T.dot(x2)).toarray()

    mu1 = x1.sum(axis=0) / p_x.sum()
    mu2 = x2.sum(axis=0) / p_x.sum()

    if Bayesian:
        _,c = estimateDCMFromMOM(p_x)
    else:
        c = sys.float_info.max

    s = computeS(p_x)
    e_co_counts = np.array(s * mu1.T.dot(mu2))

    return co_counts, e_co_counts

def computeR(x,lex,k_hat,c_hat):
    '''
    Compute the prediction rule for a given lexicon
    '''
    e_mu = getEMu(x)
    x_lex = x[:,lex].todense()
    
    numer = gammaln(1e-20+x_lex + (1+k_hat)*c_hat*e_mu[lex])\
        -gammaln(1e-20+x_lex + (1-k_hat)*c_hat*e_mu[lex])

    #print "trouble?",gammaln(x_lex + (1-k_hat)*c_hat*e_mu[lex]).min(),gammaln(x_lex + (1-k_hat)*c_hat*e_mu[lex]).max()

    denom = gammaln(1e-20+(1+k_hat)*c_hat*e_mu[lex])\
        -gammaln(1e-20+(1-k_hat)*c_hat*e_mu[lex])
    return np.array(numer.sum(axis=1) - denom.sum()).T[0]

def computeRNonBayes(x,lex,k_hat):
    weights = np.log(1+k_hat) - np.log(1-k_hat)
    return x[:,lex].dot(weights)


def e_co_diff(outer_mu,k1,k2):
    return (1. - np.outer(k1,k2)) * outer_mu

def makePredictionsKPerWord(x,pos_lex,neg_lex,k_hat_pos,k_hat_neg,c_hat, scale_preds=True, bayesian=True):
    '''
    separate \hat{k} for each lexicon.
    use this if seperate \hat{k} for each word too
    '''
    if bayesian:
        preds = computeR(x,pos_lex,k_hat_pos,c_hat) - computeR(x,neg_lex,k_hat_neg,c_hat)
    else:
        preds = computeRNonBayes(x,pos_lex,k_hat_pos) - computeRNonBayes(x,neg_lex,k_hat_neg)
    if scale:
        preds = scale(preds,x)
    return preds

scale = lambda preds, x : np.array(preds / (0.1 + x.sum(axis=1).T))[0]

