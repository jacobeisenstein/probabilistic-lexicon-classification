import numpy as np
import admm
from bayeslex_stats import getCoCountsTwoLex, getEMu, computeS, estimateK
# slsqp only
import scipy as sp
from bayeslex_stats import e_co_diff


class BayesLexOptimizer:
    def __init__(self,x,pos_lex,neg_lex,prefilter=False,max_k=0.9,verbosity=0):
        if prefilter: #try to eliminate words that don't cooccur more in-lexicon
            co_pn, e_co_pn = getCoCountsTwoLex(x,pos_lex,neg_lex)
            self.pos_lex = list(np.array(pos_lex)[co_pn.sum(axis=1) < e_co_pn.sum(axis=1)])
            self.neg_lex = list(np.array(neg_lex)[co_pn.sum(axis=0) < e_co_pn.sum(axis=0)])
            print "prefiltering from %d,%d to %d,%d"%(co_pn.shape[0],co_pn.shape[1],len(self.pos_lex),len(self.neg_lex))
        else:
            self.pos_lex = pos_lex
            self.neg_lex = neg_lex
        # now reload the counts using the lexicon
        co_pn, self.e_co_pn = getCoCountsTwoLex(x,self.pos_lex,self.neg_lex)

        e_mu = getEMu(x)
        self.N_pos = len(self.pos_lex)
        self.N_neg = len(self.neg_lex)
        self.mu_pos = e_mu[self.pos_lex]
        self.mu_neg = e_mu[self.neg_lex]
        self.s = computeS(x)

        self.co_pos = co_pn.sum(axis=1) #co-counts for each pos lex word
        self.co_neg = co_pn.T.sum(axis=1) #co-counts for each neg lex word

        rval = estimateK(x,self.pos_lex,self.neg_lex)
        ratio = np.sqrt(self.mu_neg.sum() / self.mu_pos.sum())
        #ratio = 1.
        self.k_pos = rval * ratio * np.ones(self.N_pos)
        self.k_neg = (rval / ratio) * np.ones(self.N_neg)
        self.x_sum = x.sum()  #TODO! count only lexicon words
        self.max_k = max_k
        self.verbosity = verbosity

    def estimateADMM(self,n_epochs=100,max_iter=100,rho=1.0,adaptive_rho=True,max_k=0.9,grad_based=False):
        ''' 
        This is the biconvex method in Sec 9 of Boyd et al.
        '''

        if grad_based:
            raise ValueError("Gradient-based optimization is not currently supported")
        
        u = 0.
        projector = lambda x : np.clip(x,0,max_k)
        for it in xrange(n_epochs):
            # solve k_pos = argmin(k_pos \in C) F(k_pos,k_neg) + penalty
            P_diag, P_low_rank, q, r = self.getLowRankQuadraticParams(self.co_pos,self.co_neg,self.mu_pos,self.mu_neg,self.k_neg,self.s,u,rho)
            self.k_pos = admm.admmQuadBounded(P_diag.copy(),P_low_rank,q.copy(), projector, max_iter=max_iter,rho=1.)
            # todo: offer L-BFGS based optimization here
            
            # solve k_neg = argmin(k_pos \in C) F(k_pos,k_neg) + penalty
            P_diag, P_low_rank, q, r = self.getLowRankQuadraticParams(self.co_neg,self.co_pos,self.mu_neg,self.mu_pos,self.k_pos,self.s,-u,rho)
            old_k_neg = self.k_neg
            self.k_neg = admm.admmQuadBounded(P_diag.copy(),P_low_rank,q.copy(), projector, max_iter=max_iter,rho=1.)

            # update dual parameter u
            violation = np.dot(self.mu_pos,self.k_pos) - np.dot(self.mu_neg,self.k_neg)
            u += violation
            
            # compute residuals (Boyd et al, page 18)
            #s^{k+1} in Boyd et al
            dual_residual = np.linalg.norm(rho * self.mu_pos * np.dot(self.mu_neg,self.k_neg - old_k_neg))**2
            #r^{k+1} in Boyd et al
            primal_residual = np.linalg.norm(violation)**2

            # first part is primal, second part is augmented lagrangian
            f_eval = admm.lowRankPlusDiagonalQuadProd(P_diag,P_low_rank,self.k_neg) + q.dot(self.k_neg) + r + rho * np.linalg.norm(violation)**2

            # update Rho
            rho,u = admm.updateRho(u,rho,primal_residual,dual_residual)
            
            # consider terminating
            # termination (page 19) p=1, n=size(pos_words), m=size(neg_words)

            if self.verbosity >= 2:
                print it, 'u =',u, f_eval, violation
            if self.verbosity >= 1:
                print "%d.\tDual residual=%.8f\tPrimal residual=%.8f\tRho=%.3f"%(it,dual_residual,primal_residual,rho)
            if self.verbosity >= 2:
                print ""

            eps_abs = 1e-3 / self.x_sum
            eps_rel = 1e-3 / self.x_sum
            
            eps_primal = 1. * eps_abs + eps_rel * np.max([np.dot(self.mu_pos,self.k_pos),
                                                          np.dot(self.mu_neg,self.k_neg)])
            eps_dual1 = np.sqrt(self.k_pos.shape[0]) * eps_abs + eps_rel * np.linalg.norm(u * self.mu_pos)**2
            eps_dual2 = np.sqrt(self.k_neg.shape[0]) * eps_abs + eps_rel * np.linalg.norm(u * self.mu_neg)**2
            if dual_residual < eps_dual1 + eps_dual2 and primal_residual < eps_primal:
                if self.verbosity >= 0:
                    print "done!\tit=%d\tdual=%.2e<min(%.2e,%.2e)\tprimal=%.2e<%.2e"%(it,dual_residual,eps_dual1,eps_dual2,primal_residual,eps_primal)
                break

        ## ADMM stuff
    def getLowRankQuadraticParams(self,counts_in,counts_out,mu_in,mu_out,k_out,s,u,rho1=1.):
        '''
        The augmented Lagrangian can be expressed as \frac{1}{2}x'Px + q'x + r
        with P = Diag(d) + UU'
        '''
        
        out_weight = mu_out.dot(k_out)
        counts_multiplier = self.s/self.x_sum
        
        # rho2 is due to the Lagrangian from the boundary constraint
        P_diag = ((s * out_weight * mu_in)**2)/self.x_sum #+rho2
        # there are two columns, corresponding to the primal and the Lagrangian arising from the equality constraint 
        P_low_rank = np.outer(np.array([s*np.sqrt((k_out**2).dot(mu_out**2)/self.x_sum),np.sqrt(rho1)]), mu_in).T

        # P = np.diag(P_diag) + P_low_rank.dot(P_low_rank.T)
        resid_in = counts_in - s * mu_out.sum() * mu_in
        resid_out = counts_out - s * mu_in.sum() * mu_out
        diff = u - out_weight
        q = counts_multiplier * out_weight * (resid_in * mu_in)\
            + counts_multiplier * (resid_out * mu_out * k_out).sum() * mu_in\
            + rho1 * diff * mu_in#\
            #+ rho2 * (v - a)
        
        r = .5 * ((resid_in**2).sum() + (resid_out**2).sum())/self.x_sum\
            + .5*rho1*diff**2#\
            #+ .5*rho2*np.linalg.norm(v-a)**2# + np.linalg.norm(w-b)**2)
        return P_diag,P_low_rank,q,r

            
        ## SQSLP Stuff
    
    def getDiffPos(self,p_k):
        return self.co_pos - e_co_diff(self.e_co_pn,p_k,self.k_neg).sum(axis=1)
    def getDiffNeg(self,p_k):
        return self.co_neg - e_co_diff(self.e_co_pn,self.k_pos,p_k).T.sum(axis=1)
    def sqErrPos(self,p_k):
        return (.5 * np.linalg.norm(self.getDiffPos(p_k))**2)/self.x_sum
    def sqErrNeg(self,p_k):
        return (.5 * np.linalg.norm(self.getDiffNeg(p_k))**2)/self.x_sum
    def jaccPos(self,p_k):
        return self.getDiffPos(p_k) * self.s * self.mu_pos * \
            (((1-self.k_neg)*self.mu_neg).sum()) / self.x_sum
    def jaccNeg(self,p_k):
        return self.getDiffNeg(p_k) * self.s * self.mu_neg * \
            (((1-self.k_pos)*self.mu_pos).sum()) / self.x_sum
    def con_fun_pos(self,p_k):
        return np.array(np.dot(self.mu_pos, p_k) - 
                        np.dot(self.mu_neg, self.k_neg))
    def con_jac_pos(self,p_k):
        return self.mu_pos
    def con_fun_neg(self,p_k):
        return np.array(np.dot(self.mu_pos, self.k_pos) - 
                        np.dot(self.mu_neg, p_k))
    def con_jac_neg(self,p_k):
        return self.mu_neg

    def estimateSLSQP(self,n_epochs=10,max_iter=5):
        for it in xrange(n_epochs):
            result_pos = sp.optimize.minimize\
                         (self.sqErrPos,
                          self.k_pos,
                          method='SLSQP',
                          bounds=[(0,0.9)]*self.N_pos,
                          jac=self.jaccPos,
                          constraints = ({'type':'eq',
                                          'fun':self.con_fun_pos,
                                          'jac':self.con_jac_pos}),
                          options={'maxiter':max_iter,'disp': False})
            self.k_pos = result_pos.x

            result_neg = sp.optimize.minimize\
                         (self.sqErrNeg,
                          self.k_neg,
                          method='SLSQP',
                          bounds=[(0,0.9)]*self.N_neg,
                          jac=self.jaccNeg,
                          constraints = ({'type':'eq',
                                          'fun':self.con_fun_neg,
                                          'jac':self.con_jac_neg}),
                          options={'maxiter':max_iter,'disp': False})
            self.k_neg = result_neg.x

            if self.verbosity > 0:
                print it, result_pos.fun + result_neg.fun

