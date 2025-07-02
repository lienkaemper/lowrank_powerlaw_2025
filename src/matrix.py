import numpy as np
from numpy.random import seed
from scipy.stats import linregress
import matplotlib.pyplot as plt

from numpy import random, linalg
'''Functions to generate (low rank) random matrices and to perform matrix manipulations'''


###Random model generators

def random_ball(num_points, dimension, radius=1):
    '''Generate "num_points" random points in "dimension" that have uniform probability over the unit ball
    scaled by "radius" (length of points are in range [0, "radius"]).
    Taken from: https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-spherez'''
    #Generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T

def random_dot(k,n,s,sphere=False):
    '''
    create dot product matrix of n random points sampled from $R^k$ with seed s
    if sphere==False, sample uniformly from [0,1]^k
    if sphere==True, sample uniformly from closed unit ball in $R^k$
    '''
    seed(s);
    if sphere==True:
        v=random_ball(num_points=n, dimension=k, radius=1)
    if sphere==False:
        v = random.random((n,k))
    A = v@v.T
    return A

def random_euclid_squared(k,n,s,sphere=False):
    #TODO, check if k-1 is correct
    '''
    Create distance matrix of n random points sampled from $R^k-2$
    if sphere==False, sample uniformly from [0,1]^k-2
    if sphere==True, sample uniformly from closed unit ball in $R^k-2$
    '''
    from scipy.spatial import distance_matrix
    seed(s);
    if sphere==True:
        P=random_ball(num_points=n, dimension=k-2, radius=1)
    if sphere==False:
        P = random.random((n,k-2))
    f = lambda x: x ** 2
    d = distance_matrix(P,P)
    mat=f(d)
    return mat



def random_symm(n,e,s,diagonal_zero=False):
    '''
    Create random nxn symmetric matrix
    iid uniform sampling from [0,1)*e with seed s
    '''
    seed(s)
    ut = n*(n-1)//2 #number of entries in upper triangular matrix
    P = random.rand(ut)*e
    A = np.zeros((n,n))
    A[np.triu_indices(n,1)]=P
    A = A+A.T
    if diagonal_zero==False:
        np.fill_diagonal(A,random.rand(n)*e)
    return A

def random_triu(k,n,s):
    '''
    Create random nxn symmetric matrix with singular values replaced by
    random vector with k non-zero entries sampled uniformly at random from [0,1]
    '''
    seed(s);
    A=random_symm(n,1,s,diagonal_zero=False)
    U,l,V= linalg.svd(A)
    l=random.random(l.size)
    l[random.choice(l.size,n-k,replace=False)]=0
    B = U@np.diag(l)@V
    return B

def random_truncate(k,n,s):
    '''
    Create random nxn symmetric matrix of rank k - with spectrum truncated
    to subset of k non zero entries of the original
    '''
    seed(s);
    A=random_symm(n,1,s,diagonal_zero=False)
    U,l,V= np.linalg.svd(A)
    l[random.choice(l.size,n-k,replace=False)]=0
    B = U@np.diag(l)@V
    return B

###Matrix manipulations

def shuffle_symm_matrix(A):
    ''' Shuffles de diagonal entries and shuffles the upper triangular and lower triangular entries in the same way.
    '''
    n=A.shape[0]
    up_triu=A[np.triu_indices(n,k=1)]
    np.random.shuffle(up_triu)
    diag=np.diagonal(A).copy()
    np.random.shuffle(diag)
    B=np.zeros(A.shape)
    B[np.triu_indices(n,k=1)]=up_triu
    B=B+B.T
    np.fill_diagonal(B,diag)
    return B

def f_A(A,beta,normalize_input=True, normalize_output=False, plot_hist=False):
    '''
    apply monotonic function $f=1-e^{-beta x}$ entry-wise to every element of A
    if normalize_input == True normalize entries to be between 0-1 before applying the function f
    if normalize_output == True normalize entries to be between 0-1 after applying the function f
    '''
    f = lambda x: (1-np.exp(-beta*x))
    B = np.zeros_like(A)
    if normalize_input==True:
        A=normalize_0_1(A)
    if plot_hist==True:
        print(np.min(A), np.max(A))
        plt.hist(A.flatten(), bins=100, density=True, alpha=0.5, label='Input Distribution')
        plt.show()
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = f(A[i,j]);

    if normalize_output==False:
        return B
    elif normalize_output==True:
        return normalize_0_1(B)

def normalize_0_1(A):
    '''Normalize entries of matrix to be between 0-1'''
    return (A-np.min(A))/(np.max(A)-np.min(A))

def random_symm_noise(A,epsilon=0,diagonal_zero=False, noise_type='uniform',seed=None):
    '''
    Adds symmetric additive noise to matrix A
    If type == uniform adds uniform[0,1] noise * epsilon (default=0 i.e. no noise)
    If type == normal adds normal(0,1) noise * epsilon (default=0 i.e. no noise)
    '''
    np.random.seed(seed=seed)
    if noise_type=='uniform':
        rd_matrix=np.random.uniform(size=A.shape)
    elif noise_type=='normal':
        rd_matrix=np.random.normal(size=A.shape)
    rd_matrix_sym=(rd_matrix+rd_matrix.T)/2 #Symmetrize
    if diagonal_zero==True:
        np.fill_diagonal(rd_matrix_sym, 0)
    return A+epsilon*rd_matrix_sym

### Matrix statistics
def compute_hist_mat_bounded(mat,bins=100,min_val=0,max_val=1,tol = 10 ** -8):#Todo move me to matrix
    '''compute histogram for 100 bins of equal width from 0 to 1 with extra bins for:
    numerical error given by tol and error in method i.e. more than tol value'''
    mat = mat.flatten()
    bins = np.concatenate(([-np.inf, min_val-tol], np.arange(min_val, max_val, (max_val-min_val) / bins),
                           [max_val, max_val + tol, np.inf]))  # bins for bugs, for numerical error and for values between 0-noise and 1+noise
    hist, b = np.histogram(mat, bins)
    return (bins, hist)

def compute_hist_mat(mat,bounded=True,bins=100,min_val=0,max_val=1,tol = 10 ** -8): #FIX ME and move me to matrix
    '''compute histogram of mat'''
    if bounded==True:
        return compute_hist_mat_bounded(mat,bins=100,min_val=0,max_val=1,tol = 10 ** -8)
    elif bounded==False:
        hist, bins=np.histogram(mat.flatten(), bins=100)
        return (bins, hist)