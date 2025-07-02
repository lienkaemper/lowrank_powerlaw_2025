import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import argparse

#modified by caitlin July 2 2025

# THIS SCRIPT will plot the cumulative spectrums next to the sorted spectrum
# in addition, it will have an option to not crop the y-axis of the sorted spectrum
# main script copied from 'powerlaw_script_SEPT_28.py' in nikki's personal directory
# many functions copied function from powerlaw.py to edit for own use

# this is a copy of 'lowrank_script_OCT_21.py' from nikki's personal directory
# and saves figures in '../results/sorted_spectrum_figs'
# it's been modified to take updated data structure from Nov 2022
#
# 12/24/2022: Nikki modified to take data updated by Daniela on 12/24/22

parser = argparse.ArgumentParser(description="Run spectrum and noise shuffle experiments.")
parser.add_argument('--test', action='store_true', help='Run with simplified test parameters')
args = parser.parse_args()

if args.test:
    ns = [100]
    ranks = [5, 10, 20]
    noise_levels = [0, 0.01, 0.025, 0.05, 0.1]
    betas = [3, 5]
    seeds = [1, 2]
    tstart = 5
    tstop = 50
else:
    ns = [1000]
    ranks = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    seeds = [1, 2, 3] #, 4, 5, 6, 7, 8, 9, 10]
    noise_levels = [0, 0.01, 0.025, 0.05]
    betas = [5, 7.5, 10, 12.5, 15]
    tstart  = 5
    tstop = 500


def issorted(arr):
    for i in range(len(arr)-1):
        if arr[i+1] > arr[i]:
            return False
    return True

def w_cov(x,y,w):
    "Compute weighted covariance"
    print('shape x,y,w:', np.shape(x),np.shape(y),np.shape(w))
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
    logss = np.log10(abss)

    #If we check already that this is positive, then there is not need to take abs here.
    # In principle it won't matter, since we are adding abs for possible numerical error, but it would make the code easier to read to an outsider.
    y = logss[trange][:,np.newaxis]
    y = np.reshape(y,(np.shape(y)[0],))
    #trange += 1
    nt = len(trange)#trange.size
    #x = np.asarray([np.log(j) for j in trange]) #np.concatenate((np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)#np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1) # WHAT IS THIS ?!? 
    x = np.asarray([np.log10(j) for j in trange])
    if weighted == True:
        w = np.asarray([1.0/j for j in trange]) #1.0 / trange#.astype(np.float32)[:,np.newaxis
        b = np.polyfit(x, y, 1,w=w,full =True)
        print('b', b) # Polynomial coefficients, highest power first.
        # residuals, rank, singular_values, rcond
        alpha = b[0][0]
        yshift = b[0][1]
        print('alpha',alpha)
        residuals = b[1][0]

        b0 = np.polyfit(x, y, 0,w=w,full =True)
        residuals0 = b0[1][0]

        R2 = 1 - residuals/residuals0
        print('R2', R2)
        corrcoeff = np.sqrt(R2)
        print('corrcoeff',corrcoeff)
        
        #error_trange= np.sum((w*np.square(y-(x*b)))) #np.sum(w*np.square(y-(x*b).sum(axis=1))) #This is what WLSQ is minimizing when restricted to trange
        #r2_trange= w_corrcoef((x*b),y,w) #adding [:,np.newaxis] got rid of an error on this line, but I think there is still a bug--I get some warning about " Degrees of freedom <= 0 for slice"
    else:
        b = np.polyfit(x, y, 1,full =True)
        print('b', np.shape(b), b) # Polynomial coefficients, highest power first.
        # residuals, rank, singular_values, rcond
        alpha = b[0][0]
        yshift = b[0][1]
        print('alpha',alpha)
        residuals = b[1][0]

        b0 = np.polyfit(x, y, 0,full =True)
        residuals0 = b0[1][0]

        R2 = 1 - residuals/residuals0
        print('R2', R2)
        corrcoeff = np.sqrt(R2)
        print('corrcoeff',corrcoeff)
     
    ypred = [(alpha*j+yshift) for j in x] #np.exp((x * b))
    #We are still missing computation of R^2 for the whole data set and to also return that and to have different options for w
    return alpha,yshift,ypred,R2,corrcoeff # r2_trange, error_trange

# For a given: 
#
# model ('dot', 'euc','triu','trunc'), 
# rank (5,10,15,20,30,40,50), 
# beta (none, 3,5,7,10), 
# size (1000), 
# noise (10)
#
# THIS FUNCTION:
# load spectrum of all seeds 
# make sure spectrum is sorted
# average aross all seeds to get an average sorted spectrum
# fits a line to the log-log sorted abs spectrum
# reports abs of slope of line (alpha) and goodness of fit (R^2)
# and returns the plot in a single panel to be used in a grid-like figure w various model parameters

# ss_mean0_synthetic_noise_shuffle/

def make_avg_ss_plot(ax, model, size, beta, rank, noise, seed_list, crop_y=True):

    # Nikki commented out on 12/24/22 because have new data frame uploaded by Daniela
    # df = pd.read_pickle('../data/archived/ss_synthetic_noise_shuffle/ss_noise_and_shuffle_model_random_%s_n_%s.pickle' % (model,size))
    df = pd.read_pickle('../data/ss_noise_shuffle/spectrum_model_random_%s_n_%d.pickle' % (model,size))

    SS = np.zeros((len(seed_list),size))
    for seed in seed_list:
        uss = df[rank][noise][beta][seed] # unsorted, un-absolute value spectrum
        ass = [np.abs(j) for j in uss]
        rss = np.sort(ass)
        ss = [rss[len(rss)-j-1] for j in range(len(rss))]
        ss = [(1/np.sum(ss))*j for j in ss]
        SS[seed-1,:] = ss
    avg_ss = np.mean(SS,axis=0)
    std_ss = np.sqrt(np.var(SS,axis=0))


    # note: not plotting first (largest) eigenvalue
    ax.plot([np.log10(s) for s in range(1,len(avg_ss))],[np.log10(s) for s in avg_ss[1::]],'blue',linewidth=1)
    ax.plot([np.log10(s) for s in range(1,len(avg_ss))],[np.log10(avg_ss[s] + std_ss[s]) for s in range(1,len(std_ss))],'cyan',linewidth=0.5)
    ax.plot([np.log10(s) for s in range(1,len(avg_ss))],[np.log10(avg_ss[s] - std_ss[s]) for s in range(1,len(std_ss))],'cyan',linewidth=0.5)

    #plt.xlim([0,3]) 
    ax.set_xticks([0,1,2,3],labels=[r'$10^0$',r'$10^1$',r'$10^2$',r'$10^3$'])
    ax.set_xlim([0,3])

    if crop_y == True:
        ax.set_ylim([-5,-1])
        ax.set_yticks([-5,-3,-1],labels=[r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$'])

    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))

    return ax

def make_avg_ss_grid_of_plots(model, size, beta_list, rank_list, noise, seed_list, crop_y=True):
    
    fig, axes = plt.subplots(nrows = len(beta_list), ncols = len(rank_list), figsize = (12,7))

    for i in range(len(beta_list)):
        beta = beta_list[i]
        for j in range(len(rank_list)):
            rank = rank_list[j]
            axes[i,j] = make_avg_ss_plot(axes[i,j], model, size, beta, rank, noise, seed_list, crop_y)
            
            if j == 0:
                axes[i,j].set_ylabel('Beta %s' % beta)
            if i == 0:
                axes[i,j].set_title('Rank %s' % rank)

    plt.suptitle('%s, noise %.3f' % (model, noise))
    plt.tight_layout()
    noise_name = str(int(noise*100))
    plt.savefig('../results/sorted_spectrum_figs/%s_%spercentnoise_avg_sorted_spectrum_grid' % (model,noise_name))
    #plt.show()


def make_avg_ss_fit_plot(ax, tstart, tstop, model, size, beta, rank, noise, seed_list, weight_flag=True, crop_y=True):
    
    # Nikki commented out on 12/24/22 because have new data frame uploaded by Daniela
    #df = pd.read_pickle('../data/archived/ss_synthetic_noise_shuffle/ss_noise_and_shuffle_model_random_%s_n_%s.pickle' % (model,size))
    df = pd.read_pickle('../data/ss_noise_shuffle/spectrum_model_random_%s_n_%s.pickle' % (model,size))

    SS = np.zeros((len(seed_list),size))
    for seed in seed_list:
        uss = df[rank][noise][beta][seed] # unsorted, un-absolute value spectrum
        ass = [np.abs(j) for j in uss]
        rss = np.sort(ass)
        ss = [rss[len(rss)-j-1] for j in range(len(rss))]
        ss = [(1/np.sum(ss))*j for j in ss]
        SS[seed-1,:] = ss
    avg_ss = np.mean(SS,axis=0)
    std_ss = np.sqrt(np.var(SS,axis=0))

    trange = np.asarray([j for j in range(tstart,tstop+1)])
    
    alpha,yshift,ypred,R2,corrcoeff = get_powerlaw(avg_ss, trange, check_sort = True, weighted = weight_flag)

    #ax.axline((np.log10(tstart),np.log10(ss[tstart])),(np.log10(tstop),np.log10(ss[tstop+1])),color='green')
    
    #ax.plot([np.log10(s) for s in range(tstart,tstop+1)], [s for s in ypred],'orange',linewidth=2)
    
    ax.axline((0,yshift),slope=alpha,color='red',linewidth=1)
    
    ax.axline((np.log10(tstart),0),(np.log10(tstart),1),color='green',linewidth=0.5)
    ax.axline((np.log10(tstop),0),(np.log10(tstop),1),color='green',linewidth=0.5)

    # note: not plotting first (largest) eigenvalue
    ax.plot([np.log10(s) for s in range(1,len(avg_ss))],[np.log10(s) for s in avg_ss[1::]],'blue',linewidth=1)
    ax.plot([np.log10(s) for s in range(1,len(avg_ss))],[np.log10(avg_ss[s] + std_ss[s]) for s in range(1,len(std_ss))],'cyan',linewidth=0.5)
    ax.plot([np.log10(s) for s in range(1,len(avg_ss))],[np.log10(avg_ss[s] - std_ss[s]) for s in range(1,len(std_ss))],'cyan',linewidth=0.5)

    #plt.xlim([0,3]) 
    ax.set_xticks([0,1,2,3],labels=[r'$10^0$',r'$10^1$',r'$10^2$',r'$10^3$'])
    ax.set_xlim([0,3])

    if crop_y==True:
        ax.set_ylim([-5,-1])
        ax.set_yticks([-5,-3,-1],labels=[r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$'])


    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))

    return ax, alpha, R2


def make_avg_ss_fit_grid_of_plots(tstart, tstop, model, size, beta_list, rank_list, noise, seed_list, weight_flag = True, crop_y=True):
    
    fig, axes = plt.subplots(nrows = len(beta_list), ncols = len(rank_list), figsize = (12,7))

    for i in range(len(beta_list)):
        beta = beta_list[i]
        for j in range(len(rank_list)):
            rank = rank_list[j]
            axes[i,j], alpha, R2 = make_avg_ss_fit_plot(axes[i,j], tstart, tstop, model, size, beta, rank, noise, seed_list, weight_flag, crop_y)
            
            if j == 0:
                axes[i,j].set_ylabel('Beta %s' % beta)
            if i == 0:
                axes[i,j].set_title('Rank %s' % rank)
            axes[i,j].annotate('%.2f \n%.2f' % (np.abs(alpha), R2), xy = (0.6,0.6), xycoords='axes fraction')
    plt.suptitle('%s, noise %.3f (weighted fit = %s): trange [%d,%d]' % (model, noise, weight_flag, tstart, tstop))
    plt.tight_layout()
    noise_name = str(int(noise*100))
    plt.savefig('../results/sorted_spectrum_figs/%s_%spercentnoise_avg_sorted_spectrum_grid_trange_%d_%d_weight%s' % (model,noise_name,tstart,tstop,weight_flag))
    #plt.show()


def make_avg_cumulative_ss_plot(ax, model, size, beta, rank, noise, seed_list, crop_y=True):

    #
    #df = pd.read_pickle('../data/archived/ss_synthetic_noise_shuffle/ss_noise_and_shuffle_model_random_%s_n_%s.pickle' % (model,size))
    df = pd.read_pickle('../data/ss_noise_shuffle/spectrum_model_random_%s_n_%s.pickle' % (model,size))

    SS = np.zeros((len(seed_list),size))
    for seed in seed_list:
        uss = df[rank][noise][beta][seed] # unsorted, un-absolute value spectrum
        ass = [np.abs(j) for j in uss]
        rss = np.sort(ass)
        ss = [rss[len(rss)-j-1] for j in range(len(rss))]
        ss = [(1/np.sum(ss))*j for j in ss]
        SS[seed-1,:] = ss
    avg_ss = np.mean(SS,axis=0)
    std_ss = np.sqrt(np.var(SS,axis=0))

    # 
    cum_ss = [np.sum(avg_ss[0:j]) for j in range(0,len(avg_ss))]
    #print('cum_ss', cum_ss)

    # note: not plotting first (largest) eigenvalue
    ax.plot([np.log10(s) for s in range(0,len(cum_ss))],[j for j in cum_ss],'blue',linewidth=1)
    
    #
    ax.set_xticks([0,1,2,3],labels=[r'$10^0$',r'$10^1$',r'$10^2$',r'$10^3$'])
    ax.set_xlim([0,3])

    if crop_y == True:
        ax.set_ylim([-5,-1])
        ax.set_yticks([-5,-3,-1],labels=[r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$'])


    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))

    return ax

def make_avg_cumulative_ss_grid_of_plots(model, size, beta_list, rank_list, noise, seed_list, crop_y=True):
    fig, axes = plt.subplots(nrows = len(beta_list), ncols = len(rank_list), figsize = (12,7))

    for i in range(len(beta_list)):
        beta = beta_list[i]
        for j in range(len(rank_list)):
            rank = rank_list[j]
            axes[i,j] = make_avg_cumulative_ss_plot(axes[i,j], model, size, beta, rank, noise, seed_list, crop_y)
            
            if j == 0:
                axes[i,j].set_ylabel('Beta %s' % beta)
            if i == 0:
                axes[i,j].set_title('Rank %s' % rank)
                

    plt.suptitle('%s, noise %.3f' % (model, noise))
    plt.tight_layout()
    noise_name = str(int(noise*100))
    plt.savefig('../results/sorted_spectrum_figs/%s_%spercentnoise_avg_cumulative_ss_grid' % (model,noise_name))


def make_avg_ss_and_cum_ss_grid_of_plots(model,size,beta_list,rank_list,noise,seed_list,crop_y=True):
    fig, axes = plt.subplots(nrows = len(beta_list), ncols = len(2*rank_list), figsize = (12,7))

    for i in range(len(beta_list)):
        beta = beta_list[i]
        for j in range(0,len(rank_list)):
            rank = rank_list[j]
            axes[i,2*j] = make_avg_ss_plot(axes[i,(2*j)], model, size, beta, rank, noise, seed_list, crop_y)
            axes[i,(2*j+1)] = make_avg_cumulative_ss_plot(axes[i,(2*j+1)], model, size, beta, rank, noise, seed_list, crop_y)
            
            axes[i,2*j].set_xticks([])
            axes[i,2*j].set_yticks([])
            axes[i,(2*j+1)].set_xticks([])
            axes[i,(2*j+1)].set_yticks([])

            if j == 0:
                axes[i,2*j].set_ylabel('Beta %s' % beta)
            if i == 0:
                axes[i,2*j].set_title('Rank %s' % rank)
                axes[i,(2*j+1)].set_title('Rank %s' % rank)
            
            axes[i,j].set_xticks([])
            axes[i,2*j].set_xticks([])
            axes[i,j].set_yticks([])
            axes[i,2*j].set_yticks([])

    plt.suptitle('%s, noise %.3f' % (model, noise))
    plt.tight_layout()
    noise_name = str(int(noise*100))
    plt.savefig('../results/sorted_spectrum_figs/%s_%spercentnoise_avg_N_cum_ss_grid' % (model,noise_name))
    #plt.show()

#model = 'truncate' # ['euclid_squared', 'dot', 'truncate', 'triu']
for model in ['triu', 'truncate', 'euclid_squared', 'dot']:
    size = ns[0] # 1000, 2000, 5000 - I think only have data for size 1000? :/
    seed_list = seeds

    for noise in noise_levels:
        # 12/24/22: Nikki commented out below and added above since Daniela made new data
        #noise = 0.025 #[0, 0.025, 0.05, 0.1]

        rank_list = ranks
        # 12/24/22: Nikki commented out below and added above since we decided to run new betas/ranks and Daniela made the new data
        #rank_list = [5, 10, 15, 20, 30, 40, 50]

        beta_list = betas
        # 12/24/22: Nikki commented out below and added above since we decided to run new betas/ranks and Daniela made the new data
        #beta_list =  ['ori', 3, 5, 7.5, 10]

        #tstart = 2
        tstart = tstart
        tstop = tstop
            # for triu I like [2,300], truncate seems like a lower upper bound
            # [10,300] seems to fit nicely too, [2,200] seems alright too, [2,20] is interesting

        weight_flag = True

        crop_y = False

        # THIS PLOTS WITH THE BEST FIT LINE:
        make_avg_ss_fit_grid_of_plots(tstart, tstop, model, size, beta_list, rank_list, noise, seed_list, weight_flag, crop_y)
        
        # THIS PLOTS WITHOUT THE BEST FIT LINE:
        make_avg_ss_grid_of_plots(model, size, beta_list, rank_list, noise, seed_list, crop_y)

        # ATTEMPT TO PLOT CUMULATIVE SS: it works! :) 
        make_avg_cumulative_ss_grid_of_plots(model, size, beta_list, rank_list, noise, seed_list, crop_y)

        # ATTEMPT TO PLOT BOTH AVG SS AND CUMULATIVE SS:
        make_avg_ss_and_cum_ss_grid_of_plots(model, size, beta_list, rank_list, noise, seed_list, crop_y)
    
        # WE DECIDED: [5,500] for both triu and truncate