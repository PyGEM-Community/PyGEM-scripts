""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
from collections import OrderedDict
import datetime
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
import scipy
import xarray as xr
# Local libraries
#import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
#import pygemfxns_gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup

# Script options
option_compare_mcmc_emulator_fullsim = True


#%% ===== COMPARE MCMC FROM EMULATOR AND FULL SIMULATIONS =====
if option_compare_mcmc_emulator_fullsim:
    
    overwrite = False
    cal_fp = '/Users/drounce/Documents/HiMAT/Output/calibration-fullsim/'
    plot_count_as_percent = False
    glac_nos_pkl_fn = cal_fp + 'cal_fullsim_glac_nos.pkl'
    
    if not os.path.exists(glac_nos_pkl_fn) or overwrite:
    
        glac_nos = []
        for fp in os.listdir(cal_fp):
            if not fp.startswith('.'):
                for i in os.listdir(cal_fp + fp):
                    if i.endswith('-modelprms_dict.pkl'):
                        glac_nos.append(i.split('-')[0])
        glac_nos = sorted(glac_nos)
        
        # Limit to 2500
        nglaciers = 2500
        if len(glac_nos) > nglaciers:
            
            # Random indices
            rand_idxs_all = np.arange(len(glac_nos))
            np.random.shuffle(rand_idxs_all)
            rand_idxs = rand_idxs_all[:nglaciers]
            glac_nos = [glac_nos[x] for x in rand_idxs]
            glac_nos = sorted(glac_nos)
        
        # Save for reproducability
        with open(glac_nos_pkl_fn, 'wb') as f:
            pickle.dump(glac_nos, f)        
        
    else:
        with open(glac_nos_pkl_fn, 'rb') as f:
            glac_nos = pickle.load(f)
            
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_nos)
    
    mb_obs_err = []
    emulator_mb_med = []
    emulator_mb_mad = []
    fullsim_mb_med = []
    fullsim_mb_mad = []
    emulator_tbias_med = []
    emulator_tbias_mad = []
    fullsim_tbias_med = []
    fullsim_tbias_mad = []
    emulator_kp_med = []
    emulator_kp_mad = []
    fullsim_kp_med = []
    fullsim_kp_mad = []
    emulator_ddfsnow_med = []
    emulator_ddfsnow_mad = []
    fullsim_ddfsnow_med = []
    fullsim_ddfsnow_mad = []
    
    for glac_no in glac_nos:
        
        modelprms_fn = glac_no + '-modelprms_dict.pkl'
        modelprms_fp = cal_fp + glac_no.split('.')[0].zfill(2) + '/'
        with open(modelprms_fp + modelprms_fn, 'rb') as f:
            modelprms_dict = pickle.load(f)
            
        mb_obs_err.append(modelprms_dict['emulator']['mb_obs_mwea_err'][0])
        emulator_mb_med.append(np.median(modelprms_dict['MCMC']['mb_mwea']['chain_0']))
        emulator_mb_mad.append(median_abs_deviation(modelprms_dict['MCMC']['mb_mwea']['chain_0']))
        fullsim_mb_med.append(np.median(modelprms_dict['MCMC_fullsim']['mb_mwea']['chain_0']))
        fullsim_mb_mad.append(median_abs_deviation(modelprms_dict['MCMC_fullsim']['mb_mwea']['chain_0']))
        emulator_tbias_med.append(np.median(modelprms_dict['MCMC']['tbias']['chain_0']))
        emulator_tbias_mad.append(median_abs_deviation(modelprms_dict['MCMC']['tbias']['chain_0']))
        fullsim_tbias_med.append(np.median(modelprms_dict['MCMC_fullsim']['tbias']['chain_0']))
        fullsim_tbias_mad.append(median_abs_deviation(modelprms_dict['MCMC_fullsim']['tbias']['chain_0']))
        emulator_kp_med.append(np.median(modelprms_dict['MCMC']['kp']['chain_0']))
        emulator_kp_mad.append(median_abs_deviation(modelprms_dict['MCMC']['kp']['chain_0']))
        fullsim_kp_med.append(np.median(modelprms_dict['MCMC_fullsim']['kp']['chain_0']))
        fullsim_kp_mad.append(median_abs_deviation(modelprms_dict['MCMC_fullsim']['kp']['chain_0']))
        emulator_ddfsnow_med.append(np.median(modelprms_dict['MCMC']['ddfsnow']['chain_0']))
        emulator_ddfsnow_mad.append(median_abs_deviation(modelprms_dict['MCMC']['ddfsnow']['chain_0']))
        fullsim_ddfsnow_med.append(np.median(modelprms_dict['MCMC_fullsim']['ddfsnow']['chain_0']))
        fullsim_ddfsnow_mad.append(median_abs_deviation(modelprms_dict['MCMC_fullsim']['ddfsnow']['chain_0']))
    
    mb_obs_err = np.array(mb_obs_err)
    emulator_mb_med = np.array(emulator_mb_med)
    emulator_mb_mad = np.array(emulator_mb_mad)
    fullsim_mb_med = np.array(fullsim_mb_med)
    fullsim_mb_mad = np.array(fullsim_mb_mad)
    emulator_tbias_med = np.array(emulator_tbias_med)
    emulator_tbias_mad = np.array(emulator_tbias_mad)
    fullsim_tbias_med = np.array(fullsim_tbias_med)
    fullsim_tbias_mad = np.array(fullsim_tbias_mad)
    emulator_kp_med = np.array(emulator_kp_med)
    emulator_kp_mad = np.array(emulator_kp_mad)
    fullsim_kp_med = np.array(fullsim_kp_med)
    fullsim_kp_mad = np.array(fullsim_kp_mad)
    emulator_ddfsnow_med = np.array(emulator_ddfsnow_med)
    emulator_ddfsnow_mad = np.array(emulator_ddfsnow_mad)
    fullsim_ddfsnow_med = np.array(fullsim_ddfsnow_med)
    fullsim_ddfsnow_mad = np.array(fullsim_ddfsnow_mad)
    
    dif_mb_med = emulator_mb_med - fullsim_mb_med
    dif_mb_mad = emulator_mb_mad - fullsim_mb_mad
    dif_mb_med_norm = (emulator_mb_med - fullsim_mb_med) / mb_obs_err
    dif_tbias_med = emulator_tbias_med - fullsim_tbias_med
    dif_tbias_mad = emulator_tbias_mad - fullsim_tbias_mad
    dif_kp_med = emulator_kp_med - fullsim_kp_med
    dif_kp_mad = emulator_kp_mad - fullsim_kp_mad
    dif_ddfsnow_med = emulator_ddfsnow_med - fullsim_ddfsnow_med
    dif_ddfsnow_mad = emulator_ddfsnow_mad - fullsim_ddfsnow_mad

    #%%
    # ===== PRIOR VS POSTERIOR FOR EACH GLACIER =====    
    print('abs max dif mb_med:', np.max(np.absolute(dif_mb_med)))
    print('abs max dif mb_mad:', np.max(np.absolute(dif_mb_mad)))
    print('abs max dif mb_med_norm:', np.max(np.absolute(dif_mb_med_norm)))
    print('abs max dif tbias_med:', np.max(np.absolute(dif_tbias_med)))
    print('abs max dif tbias_mad:', np.max(np.absolute(dif_tbias_mad)))
    print('abs max dif kp_med:', np.max(np.absolute(dif_kp_med)))
    print('abs max dif kp_mad:', np.max(np.absolute(dif_kp_mad)))
    print('abs max dif ddfsnow_med:', np.max(np.absolute(dif_ddfsnow_med)))
    print('abs max dif ddfsnow_mad:', np.max(np.absolute(dif_ddfsnow_mad)))
    
    # Bin spacing (note: offset them, so centered on 0)
    bdict = {}
    bdict['mb_mwea-Median'] = np.arange(-0.25, 0.26, 0.01) - 0.005
    bdict['tbias-Median'] = np.arange(-1, 1.025, 0.05) - 0.025
    bdict['kp-Median'] = np.arange(-1, 1.025, 0.05) - 0.025
    bdict['ddfsnow-Median'] = np.arange(-1, 1.025, 0.05) - 0.025
    bdict['mb_mwea-Median Absolute Deviation'] = np.arange(-0.25, 0.26, 0.01) - 0.005
    bdict['tbias-Median Absolute Deviation'] = np.arange(-1, 1.025, 0.05) - 0.025
    bdict['kp-Median Absolute Deviation'] = np.arange(-1, 1.025, 0.05) - 0.025
    bdict['ddfsnow-Median Absolute Deviation'] = np.arange(-1, 1.025, 0.05) - 0.025
    
    vn_label_dict = {'mb_mwea':'Mass Balance (m w.e. $\mathregular{yr^{-1}}$)',                                                                      
                     'kp':'Precipitation Factor (-)',                                                              
                     'tbias':'Temperature Bias ($\mathregular{^{\circ}C}$)',                                                               
                     'ddfsnow':'f$_{snow}$ (mm w.e. $\mathregular{d^{-1}}$ $\mathregular{^{\circ}C^{-1}}$)'}
    
    variables = ['mb_mwea', 'tbias', 'kp', 'ddfsnow']
    estimators = ['Median', 'Median Absolute Deviation']
    
    fig, ax = plt.subplots(len(variables), len(estimators), squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.1, 'hspace':0.4})    
    
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    ncount = 0
    for nvar, vn in enumerate(variables):
        print(nvar, vn)

        if vn == 'mb_mwea':
            em_med = emulator_mb_med
            em_mad = emulator_mb_mad
            fullsim_med = fullsim_mb_med
            fullsim_mad = fullsim_mb_mad
        elif vn == 'tbias':
            em_med = emulator_tbias_med
            em_mad = emulator_tbias_mad
            fullsim_med = fullsim_tbias_med
            fullsim_mad = fullsim_tbias_mad
        elif vn == 'kp':
            em_med = emulator_kp_med
            em_mad = emulator_kp_mad
            fullsim_med = fullsim_kp_med
            fullsim_mad = fullsim_kp_mad
        elif vn == 'ddfsnow':
            em_med = emulator_ddfsnow_med * 1e3
            em_mad = emulator_ddfsnow_mad * 1e3
            fullsim_med = fullsim_ddfsnow_med * 1e3
            fullsim_mad = fullsim_ddfsnow_mad * 1e3
        
        dif_med = em_med - fullsim_med
        dif_mad = em_mad - fullsim_mad
        
        print('  dif_mean (min/max):', np.round(dif_med.min(),2), np.round(dif_med.max(),2))
        print('  dif_std (min/max):', np.round(dif_mad.min(),2), np.round(dif_mad.max(),2))

        for nest, estimator in enumerate(estimators):
            if estimator == 'Median':
                dif = dif_med
                bcolor = 'lightgrey'
            elif estimator == 'Median Absolute Deviation':
                dif = dif_mad
                bcolor = 'lightgrey'
        
            # ===== Plot =====
#            hist, bins = np.histogram(dif)
            hist, bins = np.histogram(dif, bins=bdict[vn + '-' + estimator])
            if plot_count_as_percent:
                hist = hist * 100.0 / hist.sum()
                y_label = 'Count (%)'
                tdict = {}
                glac_ylim = 40
                tdict['Median'] = np.arange(0, glac_ylim + 1, 10)
                tdict['Median Absolute Deviation'] = np.arange(0, glac_ylim + 1, 10)
                
            else:
                y_label = 'Count'
                tdict = {}
                glac_ylim = 750
                tdict['Median'] = np.arange(0, glac_ylim + 1, 100)
                tdict['Median Absolute Deviation'] = np.arange(0, glac_ylim + 1, 100)
                
            bins_centered = bins[1:] + (bins[0] - bins[1]) / 2
            # plot histogram
            ax[nvar,nest].bar(x=bins_centered, height=hist, width=(bins[1]-bins[0]), align='center',
                              edgecolor='black', color=bcolor, alpha=0.5)
            ax[nvar,nest].set_yticks(tdict[estimator])
            ax[nvar,nest].set_ylim(0,glac_ylim)                
            
            # axis labels
            ax[nvar,nest].set_xlabel(vn_label_dict[vn], fontsize=10, labelpad=1)
            if nvar == 0:
                ax[nvar,nest].set_title('$\Delta$ ' + estimator, fontsize=12)
            if nest == 1:
                ax[nvar,nest].set_yticks([])
                
            print('  ', estimator, '% near 0:', np.round(hist[np.where(bins > 0)[0][0] - 1]))
    
            letter = letters[ncount]
            ncount += 1
            ax[nvar,nest].text(0.98, 0.98, letter, size=10, horizontalalignment='right', 
                               verticalalignment='top', transform=ax[nvar,nest].transAxes, weight='bold')
                
    # Save figure
    fig.set_size_inches(6.5,8)
    figure_fn = 'mcmc_emulator_vs_fullsim_hist.png'
    figure_fp = cal_fp + '../figures/'
    if not os.path.exists(figure_fp):
        os.makedirs(figure_fp)
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    #%%
    fig, ax = plt.subplots(len(variables), len(estimators), squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.3, 'hspace':0.4})  
    
    vn_label_dict = {'mb_mwea':'Mass Balance\n(m w.e. $\mathregular{yr^{-1}}$)',                                                                      
                     'kp':'Precipitation Factor\n(-)',                                                              
                     'tbias':'Temperature Bias\n($\mathregular{^{\circ}C}$)',                                                               
                     'ddfsnow':'f$_{snow}$\n(mm w.e. $\mathregular{d^{-1}}$ $\mathregular{^{\circ}C^{-1}}$)'}
    
    ncount = 0
    for nvar, vn in enumerate(variables):
        print(nvar, vn)

        if vn == 'mb_mwea':
            em_med = emulator_mb_med
            em_mad = emulator_mb_mad
            fullsim_med = fullsim_mb_med
            fullsim_mad = fullsim_mb_mad
            ymin = -0.8
            ymax = 0.8
            ytick_major = 0.2
            ytick_minor = 0.1
        elif vn == 'tbias':
            em_med = emulator_tbias_med
            em_mad = emulator_tbias_mad
            fullsim_med = fullsim_tbias_med
            fullsim_mad = fullsim_tbias_mad
            ymin = -6
            ymax = 6
            ytick_major = 2
            ytick_minor = 1
        elif vn == 'kp':
            em_med = emulator_kp_med
            em_mad = emulator_kp_mad
            fullsim_med = fullsim_kp_med
            fullsim_mad = fullsim_kp_mad
            ymin = -2.5
            ymax = 2.5
            ytick_major = 1
            ytick_minor = 0.5
        elif vn == 'ddfsnow':
            em_med = emulator_ddfsnow_med * 1e3
            em_mad = emulator_ddfsnow_mad * 1e3
            fullsim_med = fullsim_ddfsnow_med * 1e3
            fullsim_mad = fullsim_ddfsnow_mad * 1e3
            ymin = -2.5
            ymax = 2.5
            ytick_major = 1
            ytick_minor = 0.5
        
        dif_med = em_med - fullsim_med
        dif_mad = em_mad - fullsim_mad
        
        print('  dif_mean (min/max):', np.round(dif_med.min(),2), np.round(dif_med.max(),2))
        print('  dif_std (min/max):', np.round(dif_mad.min(),2), np.round(dif_mad.max(),2))

        for nest, estimator in enumerate(estimators):
            if estimator == 'Median':
                dif = dif_med
            elif estimator == 'Median Absolute Deviation':
                dif = dif_mad
        
            # ===== Plot =====         
            max_size = 300
            # plot histogram
            ax[nvar,nest].scatter(main_glac_rgi.Area.values, dif, s=2, color='k', marker='o')
            ax[nvar,nest].hlines(0, 0, main_glac_rgi.Area.max(), color='k', lw=0.5)
            ax[nvar,nest].set_xlim(0,max_size)
            ax[nvar,nest].set_ylim(ymin, ymax)
            
            ax[nvar,nest].yaxis.set_major_locator(MultipleLocator(ytick_major))
            ax[nvar,nest].yaxis.set_minor_locator(MultipleLocator(ytick_minor))
            ax[nvar,nest].tick_params(direction='inout', right=False)
            
            ax[nvar,nest].xaxis.set_major_locator(MultipleLocator(100))
            ax[nvar,nest].xaxis.set_minor_locator(MultipleLocator(20))
            ax[nvar,nest].tick_params(direction='inout', right=False)
            
            
            # axis labels
            if nest == 0:
                ax[nvar,nest].set_ylabel(vn_label_dict[vn], fontsize=10, labelpad=1)
            if nvar == 3:
                ax[nvar,nest].set_xlabel('Area (km$^{2}$)')
            if nvar == 0:
                ax[nvar,nest].set_title('$\Delta$ ' + estimator, fontsize=12)
            
            letter = letters[ncount]
            ncount += 1
            ax[nvar,nest].text(0.98, 0.98, letter, size=10, horizontalalignment='right', 
                               verticalalignment='top', transform=ax[nvar,nest].transAxes, weight='bold')
                
    # Save figure
    fig.set_size_inches(6.5,8)
    figure_fn = 'mcmc_emulator_vs_fullsim_scatter_vs_area_' + str(max_size) + 'km2.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    
    #%%
    print('95% mb_mwea_norm:', np.round(np.percentile(dif_mb_med_norm,2.5),2), np.round(np.percentile(dif_mb_med_norm,97.5),2))
    print('95% mb_mwea:', np.round(np.percentile(dif_mb_med,2.5),3), np.round(np.percentile(dif_mb_med,97.5),3))
    print('95% tbias:', np.round(np.percentile(dif_tbias_med,2.5),3), np.round(np.percentile(dif_tbias_med,97.5),3))
    print('95% kp:', np.round(np.percentile(dif_kp_med,2.5),3), np.round(np.percentile(dif_kp_med,97.5),3))
    print('95% ddfsnow:', np.round(np.percentile(dif_ddfsnow_med,2.5),5), np.round(np.percentile(dif_ddfsnow_med,97.5),5))
    
    print('normalized dif mb_med by mb_obs_err:')
    print('min/max:', np.round(np.min(dif_mb_med_norm),3), np.round(np.max(dif_mb_med_norm),3))

#%%
#import pickle
##fn = '/Users/drounce/Documents/HiMAT/PyGEM/Cal_fullsim_batch_0.pkl'
#fn = '/Users/drounce/Documents/HiMAT/Output/calibration-fullsim/01/1.00058-modelprms_dict.pkl'
#with open(fn, 'rb') as f:
#    A = pickle.load(f)