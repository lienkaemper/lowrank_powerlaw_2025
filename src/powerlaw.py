import os
from scipy.io import loadmat
import numpy as np
import math
import pandas
from scipy.linalg import svdvals
from sklearn.linear_model import LinearRegression
from numpy.random import normal

'''Functions compute spectra and their power laws'''

###Utilities
def issorted(arr):
    for i in range(len(arr)-1):
        if arr[i+1] > arr[i]:
            return False
    return True

### Power law estimates
def compute_spectrum(A,mean_centered=True):
    '''takes a matrix A and returns the sorted list of normalized eigenvalues in decreasing order'''
    if mean_centered==True:
        A=A-np.mean(A)
    ss, V = np.linalg.eig(A)
    return ss


#TODO check functions below

def w_cov(x,y,w):
    "Compute weighted covariance"
    return np.sum((x-np.average(x,weights=w))*(y-np.average(y,weights=w)))/np.sum(w)

def w_corrcoef(x,y,w):
    "Compute weighted correlation coefficient"
    return w_cov(x,y,w)/(np.sqrt(w_cov(x,x,w)*w_cov(y,y,w)))
    

def get_powerlaw(ss, trange, check_sort = True, weighted = True):
    '''Function as in KH code that fit exponent to variance curve doing weighted linear regression
    ss: array of spectrum (y values to fit)
    trange: array of indices of eigenvalues for fit (x values to fit)
    check_sort: check if ss is sorted'''

    abss = np.abs(ss)
    if check_sort:
        if not issorted(abss):
            abss = np.sort(abss)[::-1] 
    logss = np.log(abss)

    #If we check already that this is positive, then there is not need to take abs here.
    # In principle it won't matter, since we are adding abs for possible numerical error, but it would make the code easier to read to an outsider.
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    if weighted == True:
        w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
        # this does linear regression w/ some weights on the square error--we're finding the value of b minimizing (Wxb -Wy)^2,
        # where W is a matrix w/ w on the diagonal. maybe they got the idea here?
        # https://math.stackexchange.com/questions/1981157/regression-for-power-law
        # this gets a better fit.
        # which makes sense! it's weighting by the non-logged values of trange
        b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
        error_trange=np.sum(w*np.square(y-(x*b).sum(axis=1))) #This is what WLSQ is minimizing when restricted to trange
        r2_trange=w_corrcoef((x*b).sum(axis=1)[:,np.newaxis],y,w) #adding [:,np.newaxis] got rid of an error on this line, but I think there is still a bug--I get some warning about " Degrees of freedom <= 0 for slice"
    else:
        b = np.linalg.solve(x.T @ (x ), (x).T @ y).flatten()
        error_trange=np.sum(np.square(y-(x*b).sum(axis=1))) #This is what LSQ is minimizing when restricted to trange
        r2_trange=np.corrcoef((x*b).sum(axis=1)[:,np.newaxis],y)
        # if i use this second version, i get the same result as my function. this just does vanilla least squares linear regression
    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    #We are still missing computation of R^2 for the whole data set and to also return that and to have different options for w
    alpha = b[0]
    return alpha,ypred,r2_trange, error_trange


def get_powerlaw_matrix(A, window_min = None, window_max = None, weighted = True):
    ''' fit exponent to variance curve'''
    ss = np.linalg.eig(A)[0] ##np.linalg.svd(A)[1]
    ss = np.abs(ss)
    ss = np.sort(ss)[::-1]
    if window_min == None:
        window_min = 0
    if window_max == None:
        window_max = len(ss)
    trange = np.arange(window_min, window_max)
    return get_powerlaw(ss, trange, check_sort = False, weighted = weighted)


def force_spectrum(A,s):
    '''
    substitute the singular values of A with the array s
    '''
    U,S,V = np.linalg.svd(A, full_matrices=False)
    if type(s)==float:
        s = np.arange(1,S.size+1)**(-s/2)#Why are we using this specific power?
    D = np.diag(s)
    A_pl = U @ D @ V
    return A_pl