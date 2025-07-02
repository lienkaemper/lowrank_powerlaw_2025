#!/bin/bash -l
conda activate lowrank_powerlaw_2025
python run_ss_noise_shuffle.py --test
python plot_ss_and_alpha_fits.py --test