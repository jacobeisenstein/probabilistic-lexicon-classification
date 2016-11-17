import numpy as np
import scipy as sp

def lowRankPlusDiagonalSolve(d,u,y,explicit_invert=True):
    # solve Ax = y for A = Diad(d) + uu'

    # dimensions 
    # d: Kx1
    # u: KxN
    # y: Kx1
    # assume K >> N
    #weirdly, explicit inversion seems to be faster
    
    K1 = d.shape[0]
    K,N = u.shape
    assert(K1 == K)
    d_inv = 1/d # K x 1
    part1 = d_inv * y #elementwise multiplication. K x 1
    d_inv_u = np.reshape(d_inv,[K,1]) * u #K x N
    
    denom = np.eye(N) + d_inv_u.T.dot(u) #N x N
    
    if explicit_invert:
        inv_denom = np.linalg.inv(denom) #N x N
        left_part = d_inv_u.dot(inv_denom) #K x N
    else:
        left_part = np.linalg.solve(denom,d_inv_u.T).T
    right_part = d_inv_u.T.dot(y) #N x 1
    part2 = left_part.dot(right_part) #K x 1
    return part1 - part2

def lowRankPlusDiagonalQuadProd(d,u,x):
    # A = diag(d) + U U'
    utx = np.array(u.T.dot(x))
    return x.dot(d*x) + utx.T.dot(utx)

def admmQuadBounded(P_diag,P_low_rank,q,projection,max_iter=100,rho=1.):
    a = np.zeros_like(q) #projected
    v = np.zeros_like(q) #dual variable

    P_diag += rho
    for _ in range(max_iter):
        x = lowRankPlusDiagonalSolve(P_diag,P_low_rank,-q)
        q -= rho * (v - a)
        a = projection(x+v)
        v += x - a
        q += rho * (v - a)
        #print np.linalg.norm(v)**2
    return projection(x)

def updateRho(u,rho,primal_residual,dual_residual,mu=5.,tau_incr=2.,tau_decr=2.):
    '''
    Adaptive computation of the ADMM penalty
    See page 20 of Boyd et al
    '''
    
    if primal_residual > mu * dual_residual:
        rho *= tau_incr
        u /= tau_incr
    elif dual_residual < mu * primal_residual:
        rho /= tau_decr
        u *= tau_decr
    return rho, u
        
