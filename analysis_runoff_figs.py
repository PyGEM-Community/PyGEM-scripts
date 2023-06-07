#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses for Science manuscript

@author: drounce
"""
# Built-in libraries
import argparse
#from collections import OrderedDict
from collections import Counter
#import datetime
#import glob
import os
import pickle
import shutil
import time
#import zipfile
# External libraries
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
#from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.basemap import Basemap
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.stats import linregress
from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter
#import scipy
import xarray as xr
# Local libraries
#import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
#import pygemfxns_gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup

#from oggm import utils
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.shop import debris 
from oggm import tasks

time_start = time.time()

#%% ===== Input data =====
# Script options
option_runoff_figs = True                  # General runoff figures


rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
#rcps = ['ssp126']


deg_groups = [1.5,2,3,4]
deg_groups_bnds = [0.25, 0.5, 0.5, 0.5]
deg_group_colors = ['#4575b4', '#fee090', '#fdae61', '#f46d43', '#d73027']
temp_colordict = {}
for ngroup, deg_group in enumerate(deg_groups):
    temp_colordict[deg_group] = deg_group_colors[ngroup]

gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']

normyear = 2015

watershed_dict_fn = '/Users/drounce/Documents/HiMAT/qgis_datasets/rgi60_watershed_dict.csv'
watershed_csv = pd.read_csv(watershed_dict_fn)
watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))

temp_dev_fn = 'Global_mean_temp_deviation_2081_2100_rel_1850_1900.csv'

rgi_shp_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_all_simplified2_robinson.shp'
rgi_regions_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_regions_robinson-v2.shp'

degree_size = 0.1

rgi_reg_dict = {'all':'Global',
                'global':'Global',
                1:'Alaska',
                2:'W Canada & US',
                3:'Arctic Canada North',
                4:'Arctic Canada South',
                5:'Greenland Periphery',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus & Middle East',
                13:'Central Asia',
                14:'South Asia West',
                15:'South Asia East',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctic & Subantarctic'
                }
# Colors list
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}

rcp_namedict = {'ssp126':'SSP1-2.6', 'ssp245':'SSP2-4.5', 'ssp370':'SSP3-7.0', 'ssp585':'SSP5-8.5'}


title_dict = {'Amu_Darya': 'Amu Darya',
              'Brahmaputra': 'Brahmaputra',
              'Ganges': 'Ganges',
              'Ili': 'Ili',
              'Indus': 'Indus',
              'Inner_Tibetan_Plateau': 'Inner TP',
              'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
              'Irrawaddy': 'Irrawaddy',
              'Mekong': 'Mekong',
              'Salween': 'Salween',
              'Syr_Darya': 'Syr Darya',
              'Tarim': 'Tarim',
              'Yangtze': 'Yangtze',
              'inner_TP': 'Inner TP',
              'Karakoram': 'Karakoram',
              'Yigong': 'Yigong',
              'Yellow': 'Yellow',
              'Bhutan': 'Bhutan',
              'Everest': 'Everest',
              'West Nepal': 'West Nepal',
              'Spiti Lahaul': 'Spiti Lahaul',
              'tien_shan': 'Tien Shan',
              'Pamir': 'Pamir',
              'pamir_alai': 'Pamir Alai',
              'Kunlun': 'Kunlun',
              'Hindu Kush': 'Hindu Kush',
              13: 'Central Asia',
              14: 'South Asia West',
              15: 'South Asia East',
              'all': 'HMA',
              'Altun Shan':'Altun Shan',
              'Central Himalaya':'C Himalaya',
              'Central Tien Shan':'C Tien Shan',
              'Dzhungarsky Alatau':'Dzhungarsky Alatau',
              'Eastern Himalaya':'E Himalaya',
              'Eastern Hindu Kush':'E Hindu Kush',
              'Eastern Kunlun Shan':'E Kunlun Shan',
              'Eastern Pamir':'E Pamir',
              'Eastern Tibetan Mountains':'E Tibetan Mtns',
              'Eastern Tien Shan':'E Tien Shan',
              'Gangdise Mountains':'Gangdise Mtns',
              'Hengduan Shan':'Hengduan Shan',
              'Karakoram':'Karakoram',
              'Northern/Western Tien Shan':'N/W Tien Shan',
              'Nyainqentanglha':'Nyainqentanglha',
              'Pamir Alay':'Pamir Alay',
              'Qilian Shan':'Qilian Shan',
              'Tanggula Shan':'Tanggula Shan',
              'Tibetan Interior Mountains':'Tibetan Int Mtns',
              'Western Himalaya':'W Himalaya',
              'Western Kunlun Shan':'W Kunlun Shan',
              'Western Pamir':'W Pamir'
              }

def peakwater(runoff, time_values, nyears):
    """Compute peak water based on the running mean of N years
    
    Parameters
    ----------
    runoff : np.array
        one-dimensional array of runoff for each timestep
    time_values : np.array
        time associated with each timestep
    nyears : int
        number of years to compute running mean used to smooth peakwater variations
        
    Output
    ------
    peakwater_yr : int
        peakwater year
    peakwater_chg : float
        percent change of peak water compared to first timestep (running means used)
    runoff_chg : float
        percent change in runoff at the last timestep compared to the first timestep (running means used)
    """
    runningmean = uniform_filter(runoff, size=(nyears))
    peakwater_idx = np.where(runningmean == runningmean.max())[-1][0]
    peakwater_yr = time_values[peakwater_idx]
    peakwater_chg = (runningmean[peakwater_idx] - runningmean[0]) / runningmean[0] * 100
    runoff_chg = (runningmean[-1] - runningmean[0]) / runningmean[0] * 100
    return peakwater_yr, peakwater_chg, runoff_chg


#%%
if option_runoff_figs:
    
    regions = [11]
    
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']

    watershed_min_glaciers = 50

    year_values = np.arange(2000,2102)
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0, option_wateryear='calendar')
    time_values = dates_table.loc[:,'date'].tolist()
    
    sims_fp = '/Users/drounce/Documents/HiMAT/spc_ultee/'
    aggregated_fp = sims_fp + '/sims_aggregated/'
    fig_fp = aggregated_fp + 'figures/watersheds/'
    
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp)
    
    runoff_watershed_glac_rcps_dict = {}
    for nreg, reg in enumerate(regions):
                
        #%%
        runoff_fp = aggregated_fp + 'runoff_monthly/' + str(reg).zfill(2) + '/'
        mass_fp = aggregated_fp + 'mass_annual/' + str(reg).zfill(2) + '/'
        
#        mass_annual_rcps_dict = {}
        for rcp in rcps:
            runoff_fns = []
            runoff_fns_int = []
            for i in os.listdir(runoff_fp):
                if rcp in i:
                    runoff_fns.append(i)
                    runoff_fns_int.append(int(i.split('-')[-2]))
            runoff_fns = [x for _,x in sorted(zip(runoff_fns_int, runoff_fns))]
            
#            mass_fns = []
#            mass_fns_int = []
#            for i in os.listdir(mass_fp):
#                if rcp in i:
#                    mass_fns.append(i)
#                    mass_fns_int.append(int(i.split('-')[-2]))
#            mass_fns = [x for _,x in sorted(zip(mass_fns_int, mass_fns))]
        
            reg_glac_runoff_monthly = None
            for ds_fn in runoff_fns:
                print(ds_fn)
                ds_batch = xr.open_dataset(runoff_fp + ds_fn)
                
                if reg_glac_runoff_monthly is None:
                    reg_glac_runoff_monthly = ds_batch.glac_runoff_monthly.values
                    rgiids = list(ds_batch.RGIId.values)
                else:
                    reg_glac_runoff_monthly = np.concatenate((reg_glac_runoff_monthly, 
                                                              ds_batch.glac_runoff_monthly.values), axis=1)
                    rgiids = rgiids + list(ds_batch.RGIId.values)
#            rgiids = sorted(rgiids)
                    
            # Watersheds
            glacno_list = [x.split('-')[1] for x in rgiids]
            main_glac_rgi_reg = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
            main_glac_rgi_reg['watershed'] = main_glac_rgi_reg.RGIId.map(watershed_dict)
            
            # unique watersheds
            watersheds_unique = list(main_glac_rgi_reg.watershed.unique())
            watershed_count = {}
            watershed_idxs = {}
            for watershed in watersheds_unique:
                main_glac_rgi_watershed = main_glac_rgi_reg.loc[main_glac_rgi_reg['watershed'] == watershed]
                watershed_count[watershed] = main_glac_rgi_watershed.shape[0]
                watershed_idxs[watershed] = list(main_glac_rgi_watershed.index.values)

            # runoff data by watershed
            watershed_glac_runoff_monthly = {}
            for watershed in watersheds_unique:
                if watershed_count[watershed] > watershed_min_glaciers:
                    watershed_glac_runoff_monthly[watershed] = None
            for watershed in watersheds_unique:
                if watershed_count[watershed] > watershed_min_glaciers:
                    if watershed_glac_runoff_monthly[watershed] is None:
                        watershed_glac_runoff_monthly[watershed] = reg_glac_runoff_monthly[:,watershed_idxs[watershed],:]
                    else:
                        watershed_glac_runoff_monthly[watershed] = np.concatenate((watershed_glac_runoff_monthly[watershed], 
                                                                                   reg_glac_runoff_monthly[:,watershed_idxs[watershed],:]), axis=1)
            
            #%%
            # Dict for watersheds and scenarios
            for watershed in watersheds_unique:
                if watershed_count[watershed] > watershed_min_glaciers:
                    if not watershed in list(runoff_watershed_glac_rcps_dict.keys()):
                        runoff_watershed_glac_rcps_dict[watershed] = {}
                        for i in rcps:
                            runoff_watershed_glac_rcps_dict[watershed][i] = None
            
            # Record RCP/watershed data
            for watershed in watersheds_unique:
                if watershed_count[watershed] > watershed_min_glaciers:
                    if runoff_watershed_glac_rcps_dict[watershed][rcp] is None:
                        runoff_watershed_glac_rcps_dict[watershed][rcp] = watershed_glac_runoff_monthly[watershed]
                    else:
                        runoff_watershed_glac_rcps_dict[watershed][rcp] = np.concatenate((runoff_watershed_glac_rcps_dict[watershed][rcp],
                                                                                          watershed_glac_runoff_monthly[watershed]), axis=1)
    #%% ===== WATERSHED PLOTS =====
    watersheds = sorted(list(runoff_watershed_glac_rcps_dict.keys()))
    if 'Irrawaddy' in watersheds:
        watersheds.remove('Irrawaddy')
    if 'Yellow' in watersheds:
        watersheds.remove('Yellow')
    
    years_raw = [int(np.datetime_as_string(x).split('-')[0]) for x in ds_batch.time.values]
    years = np.unique(years_raw)
    
    #%%
    # ----- ANNUAL RUNOFF -----    
    ncols = 4
    if len(watersheds) > ncols:
        nrows = int(np.ceil(len(watersheds)/ncols))
        ncols = int(np.ceil(len(watersheds)/nrows))
    else:
        nrows = 1
        ncols = len(watersheds)
    print(nrows, ncols)

    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False, sharey=False, 
                               gridspec_kw = {'wspace':0.3, 'hspace':0.4})
    
    nrow = 0
    ncol = 0
    ymax = None
    for nwatershed, watershed in enumerate(watersheds):
        
        
        print(nrow, ncol, watershed)
        
        runoff_text = None
        for rcp in rcps:
            # Load monthly data
            ws_rcp_runoff_glac_monthly = runoff_watershed_glac_rcps_dict[watershed][rcp]
            # Aggregate glaciers to watserhed
            ws_rcp_runoff_monthly = ws_rcp_runoff_glac_monthly.sum(1)
            # Aggregate monthly to annual
            ws_rcp_runoff_annual = ws_rcp_runoff_monthly.reshape(ws_rcp_runoff_monthly.shape[0], 
                                                                 int(ws_rcp_runoff_monthly.shape[1]/12),12).sum(2)
            # Take mean of GCMs for RCP scenario
            ws_rcp_runoff_annual_mean = np.mean(ws_rcp_runoff_annual,axis=0)
            # Smooth with running mean
            ws_rcp_runoff_annual_mean_smoothed = uniform_filter(ws_rcp_runoff_annual_mean, size=(11))
            # Avg 2000-2020 for normalizing
            ws_rcp_runoff_2000_2020_mean = np.mean(ws_rcp_runoff_annual_mean_smoothed[0:20])
            
            # Label
            if nrow == 0 and ncol+1 == ncols:
                rcp_label = rcp_namedict[rcp]
            else:
                rcp_label = None
            
            # Plot each RCP scenario
            ax[nrow,ncol].plot(years, ws_rcp_runoff_annual_mean_smoothed / ws_rcp_runoff_2000_2020_mean, 
                               color=rcp_colordict[rcp], linestyle='-', 
                               linewidth=1, zorder=4, label=rcp_label)
            
            # Plot peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_rcp_runoff_annual_mean, years, 11)
            peakwater_yr_idx = np.where(years==peakwater_yr)[0][0]
            ax[nrow,ncol].plot((peakwater_yr, peakwater_yr), 
                               (0, ws_rcp_runoff_annual_mean_smoothed[peakwater_yr_idx] / ws_rcp_runoff_2000_2020_mean), 
                               color=rcp_colordict[rcp], linewidth=1, linestyle='--', zorder=5)
            
            if runoff_text is None:
                # Runoff per year
                runoff_text = str(np.round(ws_rcp_runoff_2000_2020_mean/1e9,1)) + ' km$^{3}$ yr$^{-1}$'
                ax[nrow,ncol].text(0.98, 0.98, runoff_text, size=10, horizontalalignment='right', 
                                   verticalalignment='top', transform=ax[nrow,ncol].transAxes)
                # Watershed label
                if watershed in title_dict.keys():
                    watershed_label = title_dict[watershed]
                else:
                    watershed_label = watershed
                ax[nrow,ncol].text(1, 1.01, watershed_label, size=12, horizontalalignment='right', 
                                   verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)
                
            
            if ymax is None:
                ymax = ws_rcp_runoff_annual_mean_smoothed[peakwater_yr_idx] / ws_rcp_runoff_2000_2020_mean
            elif ws_rcp_runoff_annual_mean_smoothed[peakwater_yr_idx] / ws_rcp_runoff_2000_2020_mean > ymax:
                ymax = ws_rcp_runoff_annual_mean_smoothed[peakwater_yr_idx] / ws_rcp_runoff_2000_2020_mean
            
        if nrow == 0 and ncol+1 == ncols:
            ax[nrow,ncol].legend(loc=(1.05,0.5), fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                      )
        
        # Update rows and cols
        ncol += 1
        if ncol == ncols:
            ncol = 0
            nrow += 1
    
    # Set axes
    nrow = 0
    ncol = 0
    for nwatershed, watershed in enumerate(watersheds):
        if ncol == 0:
            ax[nrow,ncol].set_ylabel('Runoff (-)')
        
        ymax = int(np.ceil(ymax/0.25))*0.25
        ax[nrow,ncol].set_ylim(0,ymax)
        ax[nrow,ncol].set_xlim(2000,2100)
        
        ax[nrow,ncol].yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax[nrow,ncol].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax[nrow,ncol].xaxis.set_major_locator(plt.MultipleLocator(40))
        ax[nrow,ncol].xaxis.set_minor_locator(plt.MultipleLocator(20))
        
        ax[nrow,ncol].yaxis.set_ticks_position('both')
        ax[nrow,ncol].tick_params(axis='both', which='major', labelsize=10, direction='inout')
        ax[nrow,ncol].tick_params(axis='both', which='minor', labelsize=10, direction='inout')  
        
#        if nrow < nrows-1:
#            ax[nrow,ncol].axes.xaxis.set_ticklabels([])

        # Update rows and cols
        ncol += 1
        if ncol == ncols:
            ncol = 0
            nrow += 1
            
    fig_fn = ('watersheds_' + str(years[0]) + '-' + str(years[-1]) + '_multigcm_runoff_annual.png')
    fig.set_size_inches(8,2*nrows)
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
        
    #%%
        
    # Plots for watersheds:
    assert 1==0, 'export data for each GCM/scenario'
    assert 1==0, 'plot for each watershed'
#        
#            reg_runoff_monthly = reg_glac_runoff_monthly.sum(1)
#            reg_runoff_annual = reg_runoff_monthly.reshape(reg_runoff_monthly.shape[0],int(reg_runoff_monthly.shape[1]/12),12).sum(2)
#
##            reg_glac_mass = None
##            for ds_fn in mass_fns:
##                print(ds_fn)
##                ds_batch = xr.open_dataset(mass_fp + ds_fn)
##                
##                if reg_glac_mass is None:
##                    reg_glac_mass = ds_batch.glac_mass_annual.values
##                else:
##                    reg_glac_mass = np.concatenate((reg_glac_mass, 
##                                                              ds_batch.glac_mass_annual.values), axis=1)
##            reg_mass_annual = reg_glac_mass.sum(1)
#
#            runoff_annual_rcps_dict[rcp] = reg_runoff_annual
##            mass_annual_rcps_dict[rcp] = reg_mass_annual
#            
#            
            
#    
#            assert 1==0, 'here'
