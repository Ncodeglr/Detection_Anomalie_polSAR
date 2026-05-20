import numpy as np
from cvxopt import matrix, solvers

def rbf_kernel_complex(X, Y=None, gamma=1.0):
    """
    RBF kernel for complex vectors:
    k(x,y) = exp(-gamma * ||x - y||^2), where ||x-y||^2 = sum_k |x_k - y_k|^2
    X: (n,d) complex
    """
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    Y = np.atleast_2d(Y)
    XX = np.sum(np.abs(X)**2, axis=1)[:, None]
    YY = np.sum(np.abs(Y)**2, axis=1)[None, :]
    XY_real = np.real(X @ Y.conj().T)   # Re(x^H y)
    d2 = XX + YY - 2 * XY_real
    return np.exp(-gamma * d2)

def rbf_kernel(X, Y=None, gamma=1.0):
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    Y = np.atleast_2d(Y)
    XX = np.sum(np.abs(X)**2, axis=1)[:, None]
    YY = np.sum(np.abs(Y)**2, axis=1)[None, :]
    XY_real = np.real(X @ Y.T)   # Re(x^H y)
    d2 = XX + YY - 2 * XY_real
    return np.exp(-gamma * d2)

def linear_kernel_complex(X, Y=None):
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    return X @ Y.conj().T  # returns complex matrix

def gauss_kernel(X, Y, gamma=1.0):
    K = np.zeros([len(X), len(X)])
    for i in range(len(X)-1):
        for j in range(len(X)-1):
            x, y = X[i,:], Y[i,:]
            real = np.sum((x.real - y.real)**2)
            im = np.sum((x.imag + y.imag)**2)
            ir = np.sum((x.imag + y.imag)*(x.real - y.real))
            K[i,j] = real + 2j*ir - im
    return K

def poly_complex_kernel(X, Y=None, c=1.0, d=2):
    if Y is None: Y = X
    K = (X.conj() @ Y.T + c)**d   # (x^H y + c)^d
    return K

from scipy.linalg import cholesky, solve_triangular

def cov_kernel(X, Y=None, M=None, gamma=0.01):
    if Y is None:
        Y = X
    L = cholesky(M, lower=True)
    X_t = X @ L.T.conj()
    Y_t = Y @ L.T.conj()
    XX = np.sum(np.abs(X_t)**2, axis=1)[:, None]
    YY = np.sum(np.abs(Y_t)**2, axis=1)[None, :]
    K = XX + YY - 2 * np.real(X_t @ Y_t.conj().T)
    return np.exp(-gamma * K)

def cov_kernel_inverse(X, Y=None, M=None, gamma=0.01):
    if Y is None:
        Y = X
    L = cholesky(M, lower=True)
    X1 = solve_triangular(L, X.T, lower=True)
    X_t = solve_triangular(L.conj().T, X1, lower=False).T
    Y1 = solve_triangular(L, Y.T, lower=True)
    Y_t = solve_triangular(L.conj().T, Y1, lower=False).T
    XX = np.sum(np.abs(X_t)**2, axis=1)[:, None]
    YY = np.sum(np.abs(Y_t)**2, axis=1)[None, :]
    K = XX + YY - 2*np.real(X_t @ Y_t.conj().T)
    return np.exp(-gamma*K)

def is_psd(A, tol=1e-10):
    A = np.array(A)
    A = (A + A.T.conj()) / 2
    eigvals = np.linalg.eigvalsh(A)
    return np.all(eigvals >= -tol), eigvals

def kernel_matrix(x, kernel='linear', gamma='auto'):
    X = np.atleast_2d(x)
    if kernel == 'rbf':
        return rbf_kernel(X, X, gamma=gamma)
    elif kernel == 'rbf_complex':
        return rbf_kernel_complex(X, X, gamma=gamma)
    elif kernel == 'linear':
        return (X @ X.conj().T)
    elif kernel == 'gauss':
        return  gauss_kernel(X, X, gamma=gamma)
    elif kernel == "poly":
        return poly_complex_kernel(X, X, c=1.0, d=gamma)
    elif kernel == "cov":
        return cov_kernel(X,  X, gamma)
    elif kernel == "inv_cov":
        return cov_kernel_inverse(X,  X, gamma)    

def train_svdd_complex_simple2(X, C=0.1, kernel='linear', gamma=1.0, verbose=False):
    """
    Entraîne un SVDD sur des données complexes en utilisant cvxopt.
    """
    n = X.shape[0]
    K = kernel_matrix(X, kernel=kernel, gamma=gamma)
    
    # QP : min 0.5 x^T P x + q^T x
    P = matrix(np.real((K + K.conj().T)/2).astype(np.float64)) # symétrisation
    q = matrix(-np.real(np.diag(K)).astype(np.float64))

    # Contraintes : Gx <= h  (0 <= alpha <= C)
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]).astype(np.float64))
    h = matrix(np.hstack([np.zeros(n), C*np.ones(n)]).astype(np.float64))

    # Contraintes d’égalité : Ax = b  (somme = 1)
    A = matrix(np.ones((1, n)).astype(np.float64))
    b = matrix(np.ones(1).astype(np.float64))

    # Résolution
    solvers.options['show_progress'] = verbose
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x']).flatten()

    return {
        'X_train': X,
        'alpha': alpha,
        'kernel': kernel,
        'gamma': gamma,
        'K': K
    }

def score_svdd_batched(model, X_test, chunk_size=10000):
    """
    Version vectorisée (rapide) pour évaluer un grand nombre d'images de Test d'un coup.
    Traite les données par lots (chunks) pour éviter de saturer la RAM.
    Retourne la distance^2 au centre (plus c'est grand, plus c'est une anomalie).
    """
    alpha = model['alpha']
    K_train = model['K']
    X_train = model['X_train']
    
    if model['kernel'] not in ['rbf', 'rbf_complex']:
        raise NotImplementedError("Pour l'instant batched scoring est opti pour RBF.")
        
    n_samples = X_test.shape[0]
    dist2 = np.zeros(n_samples)
    
    term3 = np.real(alpha @ (K_train @ alpha))
    YY = np.sum(np.abs(X_train)**2, axis=1)[None, :]
    
    for i in range(0, n_samples, chunk_size):
        X_chunk = X_test[i:i+chunk_size]
        XX = np.sum(np.abs(X_chunk)**2, axis=1)[:, None]
        XY = np.real(X_chunk @ X_train.conj().T)
        K_vec = np.exp(-model['gamma'] * (XX + YY - 2 * XY))
        
        term1 = 1.0 # Le RBF d'un point avec lui même fait 1
        term2 = -2.0 * np.real(K_vec @ alpha)
        
        dist2[i:i+chunk_size] = term1 + term2 + term3
        
    return dist2