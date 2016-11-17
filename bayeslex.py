from scipy.sparse import csr_matrix
import numpy as np
import scipy as sp
import sys
import argparse

import admm
from bayeslex_data import getLex, loadData, loadExtraData
from bayeslex_baselines import pmiPredictor, getLexClassifier
from bayeslex_eval import threeClassAcc, resultString
from bayeslex_opt import BayesLexOptimizer
from bayeslex_stats import estimateDCMFromMOM,computeR,computeRNonBayes,scale,makePredictionsKPerWord

parser = argparse.ArgumentParser()
parser.add_argument('prefix')
parser.add_argument('poslex')
parser.add_argument('neglex')
parser.add_argument('--vocab_size',default=50000,type=int)
parser.add_argument('--epochs',default=200,type=int)
parser.add_argument('--iters_per_epoch',default=5,type=int)
parser.add_argument('--optimizer',default='admm')
parser.add_argument('--admm_rho',default=1.0,type=float)
parser.add_argument('--max_k',default=0.9,type=float)
parser.add_argument('--verbosity',default=0,type=int)
parser.add_argument('--extra',default=None,type=str)

prefilter_group = parser.add_mutually_exclusive_group(required=False)
prefilter_group.add_argument('--prefilter',dest='prefilter',action='store_true',
                             help="""Prefilter the vocabulary to only include items
                             whose observed cross-lexicon counts are lower than expected.
                             Does not seem to make things better.""")
prefilter_group.add_argument('--no-prefilter',dest='prefilter',action='store_false')
parser.set_defaults(prefilter=False)

grad_based_group = parser.add_mutually_exclusive_group(required=False)
grad_based_group.add_argument('--grad',dest='grad_based',action='store_true',
                              help="""Use gradient-based optimization inside ADMM inner loop""")
grad_based_group.add_argument('--quadratic',dest='grad_based',action='store_false',
                              help="""Use closed-form quadratic optimization inside ADMM inner loop""")
parser.set_defaults(grad_based=False)

global args

#noDiag = lambda mat : np.triu(mat,1) + np.tril(mat,-1)
        
def main():
    global args
    args = parser.parse_args()

    y,x,vocab = loadData(args.prefix,args.vocab_size)
    
    print args
    print "===================================="
    print "docs: %d\t vocabulary: %d\t tokens per doc: %.3f"%(x.shape[0],x.shape[1],x.sum(axis=1).mean())
    pos_lex = getLex(args.poslex,vocab)
    neg_lex = getLex(args.neglex,vocab)
    print "lexicon sizes: %d\t%d"%(len(pos_lex),len(neg_lex))
    
    clf = getLexClassifier(pos_lex,neg_lex,vocab)
    
    pred_baseline = np.array(clf.dot(x.T).todense())[0] 
    print resultString(scale(pred_baseline,x),y,"baseline")
    
    pred_presence = np.array(clf.dot((x>0).T).todense())[0] 
    print resultString(scale(pred_presence,x),y,"presence")

    pred_pmi = pmiPredictor(x,pos_lex,neg_lex)
    print resultString(scale(pred_pmi,x),y,"pmi")
    
    e_mu, c_hat = estimateDCMFromMOM(x)
    print 'c_hat=',c_hat
    sys.stdout.flush()

    # load extra data if provided
    if args.extra is not None:
        x_train = sp.sparse.vstack([x,loadExtraData(args.extra,vocab)])
    else:
        x_train = x

    opt = BayesLexOptimizer(x_train,pos_lex,neg_lex,
                                    prefilter=args.prefilter,
                                    max_k=args.max_k,
                                    verbosity=args.verbosity)
    if args.optimizer == 'admm':
        print 'ADMM optimization'
        opt.estimateADMM(max_iter=args.iters_per_epoch,
                         n_epochs=args.epochs,
                         rho=args.admm_rho,
                         grad_based=args.grad_based
        )
    elif args.optimizer == 'slsqp':
        print 'SLSQP optimization (warning, slow!)'
        opt.estimateSLSQP(max_iter=args.iters_per_epoch,
                          n_epochs=args.epochs)
    else:
        raise Exception('Valid optimizers are admm and slsqp only')
        
    pred_khat = makePredictionsKPerWord(x,opt.pos_lex,opt.neg_lex,opt.k_pos,opt.k_neg,c_hat,bayesian=False)
    print resultString(pred_khat,y,"LexiMom")

    pred_khat_bayes = makePredictionsKPerWord(x,opt.pos_lex,opt.neg_lex,opt.k_pos,opt.k_neg,c_hat,bayesian=True)
    print resultString(pred_khat_bayes,y,"LexiMom-Bayes")

    
if __name__ == "__main__":
    main()


