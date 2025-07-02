#!/usr/bin/env python3

import numpy as np
import pandas as pd
import itertools
import pickle
from tqdm import tqdm
import time
import argparse

#TODO FIX SO THAT WE CAN IMPORT FROM FOLDER ABOVE
import src.powerlaw as powerlaw
import src.matrix as matrix

#modified by caitlin July 2 2025

parser = argparse.ArgumentParser(description="Run spectrum and noise shuffle experiments.")
parser.add_argument('--test', action='store_true', help='Run with simplified test parameters')
args = parser.parse_args()

if args.test:
    ns = [100]
    ranks = [5, 10, 20]
    noise_levels = [0, 0.01, 0.025, 0.05, 0.1]
    betas = [3, 5]
    seeds = [1, 2]
    plot_hist = True
else:
    ns = [1000]
    ranks = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    seeds = [1, 2, 3]#, 4, 5, 6, 7, 8, 9, 10]
    noise_levels = [0, 0.01, 0.025, 0.05]
    betas = [5, 7.5, 10, 12.5, 15]
    plot_hist = False


def compute_ss_noise_and_shuffle(n, ranks, betas, seeds, noise_levels, ModelToRun):
    '''
    n: integer, matrix size
    ranks: list of integers, to be used as ranks
    seeds: list of integers, to be used as random seeds
    noise_levels: list of floats that determine size of uniform noise added
    ModelToRun: model to generate random matrix - ...
    '''
    ss = dict.fromkeys(ranks)  # makes a dictionary where the keys are the ranks, values are none
    ss_shuffle = dict.fromkeys(ranks)  # makes a dictionary where the keys are the ranks, values are none
    hist = dict.fromkeys(ranks)  # makes a dictionary where the keys are the ranks, "bins" is an extra key, values are none
    stats=dict.fromkeys(ranks)
    #Creating dictionary structure for storage
    for k in ranks:
        ss[k] = dict.fromkeys(noise_levels)
        ss_shuffle[k] = dict.fromkeys(noise_levels)
        hist[k] = dict.fromkeys(noise_levels)
        stats[k]=dict.fromkeys(noise_levels)
    for k, noise_level, in itertools.product(ranks, noise_levels):
        ss[k][noise_level]  = dict.fromkeys(betas + ["ori"])
        ss_shuffle[k][noise_level]  = dict.fromkeys(betas + ["ori"])
        hist[k][noise_level] = dict.fromkeys(betas + ["ori"])
        stats[k][noise_level] = dict.fromkeys(betas + ["ori"])
    for k, noise_level, beta in itertools.product(ranks, noise_levels, betas + ["ori"]):
        ss[k][noise_level][beta] = dict.fromkeys(seeds)
        ss_shuffle[k][noise_level][beta] = dict.fromkeys(seeds)
        hist[k][noise_level][beta] = dict.fromkeys(seeds)
    for k, noise_level, beta in itertools.product(ranks, noise_levels, betas + ["ori"]):
        stats[k][noise_level][beta]= dict.fromkeys(seeds)
    for k, noise_level, beta,seed in itertools.product(ranks, noise_levels, betas + ["ori"],seeds):
        stats[k][noise_level][beta][seed]=  dict.fromkeys(['min','max','mean','std'])

    #Generating data
    for k, noise_level, seed in tqdm(itertools.product(ranks, noise_levels, seeds),desc="Rank,noise and seed, fixed n",total=len(ranks) *len(noise_levels)* len(seeds)):
        diagonal_zero=False #most models are non-zero in the diagonal
        if ModelToRun==matrix.random_euclid_squared:
            diagonal_zero=True #only distance models are zero in the diagonal
        A = ModelToRun(k, n, seed)
        noise_scaled = (np.max(A)-np.min(np.min(A),0))*noise_level
        A = matrix.random_symm_noise(A,noise_scaled,diagonal_zero=diagonal_zero)
        A_shuffle = matrix.shuffle_symm_matrix(A)
        spec = powerlaw.compute_spectrum(A,mean_centered=False)
        spec_shuffle = powerlaw.compute_spectrum(A_shuffle,mean_centered=False)
        bins, hist_bin = matrix.compute_hist_mat(A,bounded=False)
        ss[k][noise_level]['ori'][seed] = spec
        ss_shuffle[k][noise_level]['ori'][seed] = spec_shuffle
        hist[k][noise_level]['ori'][seed] = (bins, hist_bin)

        stats[k][noise_level]['ori'][seed]["min"]=np.min(A)
        stats[k][noise_level]['ori'][seed]["max"]=np.max(A)
        stats[k][noise_level]['ori'][seed]["mean"]=np.mean(A)
        stats[k][noise_level]['ori'][seed]["std"]=np.std(A)

        for beta in betas:
            fA = matrix.f_A(A, beta,normalize_input=True,normalize_output=False, plot_hist=plot_hist)
            fA_shuffle = matrix.shuffle_symm_matrix(fA)
            spec = powerlaw.compute_spectrum(fA,mean_centered=False)
            spec_shuffle=powerlaw.compute_spectrum(fA_shuffle,mean_centered=False)
            bins, hist_bin = matrix.compute_hist_mat(fA,bounded=True)
            ss[k][noise_level][beta][seed]= spec
            ss_shuffle[k][noise_level][beta][seed]= spec_shuffle
            hist[k][noise_level][beta][seed] = (bins, hist_bin)

            stats[k][noise_level][beta][seed]["min"]=np.min(fA)
            stats[k][noise_level][beta][seed]["max"]=np.max(fA)
            stats[k][noise_level][beta][seed]["mean"]=np.mean(fA)
            stats[k][noise_level][beta][seed]["std"]=np.std(fA)
            stats[k][noise_level][beta][seed]["std"]=np.std(fA)
    return ss, ss_shuffle, hist,stats


#models to run
models=[matrix.random_truncate, matrix.random_euclid_squared,matrix.random_dot,matrix.random_triu]
#models=[matrix.random_euclid_squared]

for ModelToRun in tqdm(models,
                       desc="Loop on model", total=4):
    start = time.time()
    for n in tqdm(ns, desc="Loop on matrix size", total=len(ns)):
        root="../data/ss_noise_shuffle/"
        path_out_ss = root+"spectrum_model_{}_n_{}.pickle".format(ModelToRun.__name__, n)
        print(path_out_ss)
        path_out_ss_shuffle = root+"spectrum_shuffle_control_model_{}_n_{}.pickle".format(ModelToRun.__name__, n)
        print(path_out_ss_shuffle)
        path_out_hist = root+"histograms_entries_model_{}_n_{}.pickle".format(ModelToRun.__name__, n)
        print(path_out_hist)
        path_out_stats = root+"stats_entries_model_{}_n_{}.pickle".format(ModelToRun.__name__, n)
        print(path_out_stats)
        ss, ss_shuffle, hist,stats = compute_ss_noise_and_shuffle(n, ranks, betas, seeds, noise_levels, ModelToRun)
        print("writing spectra")
        with open(path_out_ss, 'wb') as fp:
            pickle.dump(ss, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("done")
        print("writing spectra shuffled")
        with open(path_out_ss_shuffle, 'wb') as fp:
            pickle.dump(ss_shuffle, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("done")
        print("writing histograms")
        with open(path_out_hist, 'wb') as fp:
            pickle.dump(hist, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("done")
        print("writing stats")
        with open(path_out_stats, 'wb') as fp:
            pickle.dump(stats, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("done")
    print(f'Done in {(time.time()-start)/60:.2f} minutes')

