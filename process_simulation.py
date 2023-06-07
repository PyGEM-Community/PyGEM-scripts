""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
import argparse
import collections
#import datetime
#import glob
import os
import pickle
import shutil
import time
import zipfile
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
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.basemap import Basemap
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
#from scipy.stats import linregress
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


#%% ===== Input data =====
# Script options
option_find_missing = False             # Checks file transfers and finds missing glaciers
option_move_files = False               # Moves files from one directory to another
option_zip_sims = False                 # Zips binned and stats output for each region (gcm/scenario)
option_process_data = False             # Processes data for regional statistics
option_process_data_nodebris = False    # Processes data for regional statistics
option_process_data_wcalving = False    # Processes data for regional statistics replacing tidewater glaciers w calving included
option_process_fa_err = False           # Processes frontal ablation error for regional statistics
option_calving_mbclim_era5 = False      # mbclim of two lowest elevation bins for Will 
option_multigcm_plots_reg = False        # Multi-GCM plots of various parameters for RGI regions
option_multigcm_plots_ws = False        # Multi-GCM plots of various parameters for watersheds
option_glacier_cs_plots = False         # Individual glacier cross section plots
option_glacier_cs_plots_NSFANS = False  # Plots for NSF ANS proposal
option_debris_comparison = False        # Mutli-GCM comparison of including debris or not
option_calving_comparison = False       # Multi-GCM comparison of including frontal ablation or not
option_policy_temp_figs = False         # Policy figures based on temperature deviations 
option_sensitivity_figs = False         # Mass balance sensitivity figures based on temp/prec deviations
option_swap_calving_sims = False        # Unzip, swap in tidewater glacier runs with calving on, zip back and save
option_extract_sims = False             # Extract sims to process subregions
option_export_timeseries = False        # Export csv data for review of paper
option_extract_area = False              # Export csv of area


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


def excess_meltwater_m3(glac_vol, option_lastloss=1):
    """ Excess meltwater based on running minimum glacier volume 
    
    Note: when analyzing excess meltwater for a region, if there are glaciers that gain mass, the excess meltwater will
    be zero. Consequently, the total excess meltwater will actually be more than the total mass loss because these
    positive mass balances do not "remove" total excess meltwater.
    
    Parameters
    ----------
    glac_vol : np.array
        glacier volume [km3]
    option_lastloss : int
        1 - excess meltwater based on last time glacier volume is lost for good
        0 - excess meltwater based on first time glacier volume is lost (poorly accounts for gains)
    option_lastloss = 1 calculates excess meltwater from the last time the glacier volume is lost for good
    option_lastloss = 0 calculates excess meltwater from the first time the glacier volume is lost, but does
      not recognize when the glacier volume returns
    """
    glac_vol_m3 = glac_vol * pygem_prms.density_ice / pygem_prms.density_water * 1000**3
    if option_lastloss == 1:
        glac_vol_runningmin = np.maximum.accumulate(glac_vol_m3[:,::-1],axis=1)[:,::-1]
        # initial volume sets limit of loss (gaining and then losing ice does not contribute to excess melt)
        for ncol in range(0,glac_vol_m3.shape[1]):
            mask = glac_vol_runningmin[:,ncol] > glac_vol_m3[:,0]
            glac_vol_runningmin[mask,ncol] = glac_vol_m3[mask,0]
    else:
        # Running minimum volume up until that time period (so not beyond it!)
        glac_vol_runningmin = np.minimum.accumulate(glac_vol_m3, axis=1)
    glac_excess = glac_vol_runningmin[:,:-1] - glac_vol_runningmin[:,1:] 
    return glac_excess
        

def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'rgi_region':
        groups = main_glac_rgi_all.O1Region.unique().tolist()
        group_cn = 'O1Region'
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    else:
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn



#%%
time_start = time.time()
if option_find_missing:
    option_best_calving = False  # Option to look at only tidewater glaciers
    
    for reg in regions:
        
        # All glaciers for fraction and missing
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        
        if option_best_calving:
            # Best
            rcp = 'ssp126'
            gcm_name = 'BCC-CSM2-MR'
            # Filepath where glaciers are stored
            netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
            netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
            
            # Load the glaciers
            glacno_list_gcmrcp = []
            for i in os.listdir(netcdf_fp):
                if i.endswith('.nc'):
                    glacno_list_gcmrcp.append(i.split('_')[0])
            glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
            main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glacno_list_gcmrcp)
            print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
        
        # Load glaciers
        glacno_list = []
        glacno_list_gcmrcp_missing = {}
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Check other file too
                glacno_binned_count = 0
                for i in os.listdir(netcdf_fp_binned):
                    if i.endswith('.nc'):
                        glacno_binned_count += 1
                print('  count of stats  files:', len(glacno_list_gcmrcp))
                print('  count of binned files:', glacno_binned_count)
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
                
                # Missing glaciers by gcm/rcp
                glacno_list_gcmrcp_missing[gcm_name + '-' + rcp] = (
                        sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list_gcmrcp).tolist()))
                
#                #%%
#                # Hack to find missing compared to other runs
##                glacno_list_best = glacno_list_gcmrcp['CESM2-ssp245']
#                fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
#                with open(pickle_fp + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'rb') as f:
#                    glacno_list_best = pickle.load(f)
#                
#                glacno_list_find = glacno_list_gcmrcp
#                
#                A = sorted(np.setdiff1d(glacno_list_best, glacno_list_find).tolist())
#                if len(A) > 0:
#                    print('  missing:', A)
#                #%%
                
                
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        
        #%%




#%%
if option_zip_sims:
    """ Zip simulations """
    for reg in regions:
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Calving glaciers
        termtype_list = [1,5]
        main_glac_rgi_calving = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi_calving.reset_index(inplace=True, drop=True)
        glacno_list_calving = list(main_glac_rgi_calving.glacno.values)
        
        for gcm_name in gcm_names:
            for rcp in rcps:
                print('zipping', reg, gcm_name, rcp)

                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- Zip directories -----
                zipped_fp_binned = netcdf_fp_cmip5 + '_zipped/' + str(reg).zfill(2) + '/binned/'
                zipped_fp_stats = netcdf_fp_cmip5 + '_zipped/' + str(reg).zfill(2) + '/stats/'
                zipped_fn_binned = gcm_name + '_' + rcp + '_binned'
                zipped_fn_stats = gcm_name + '_' + rcp + '_stats'
                
                if not os.path.exists(zipped_fp_binned):
                    os.makedirs(zipped_fp_binned, exist_ok=True)
                if not os.path.exists(zipped_fp_stats):
                    os.makedirs(zipped_fp_stats, exist_ok=True)
                    
                shutil.make_archive(zipped_fp_binned + zipped_fn_binned, 'zip', netcdf_fp_binned)
                shutil.make_archive(zipped_fp_stats + zipped_fn_stats, 'zip', netcdf_fp_stats)
                
                if not 'nodebris' in netcdf_fp_cmip5:
                    # ----- Copy calving glaciers for comparison -----
                    if len(glacno_list_calving) > 0:
                        calving_fp_binned = netcdf_fp_cmip5 + '_calving/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                        calving_fp_stats = netcdf_fp_cmip5 + '_calving/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                        
                        if not os.path.exists(calving_fp_binned):
                            os.makedirs(calving_fp_binned, exist_ok=True)
                        if not os.path.exists(calving_fp_stats):
                            os.makedirs(calving_fp_stats, exist_ok=True)
                        
                        # Copy calving glaciers for comparison
                        for glacno in glacno_list_calving:
                            binned_fn = glacno + '_' + gcm_name + '_' + rcp + '_MCMC_ba1_50sets_2000_2100_binned.nc'
                            if os.path.exists(netcdf_fp_binned + binned_fn):
                                shutil.copyfile(netcdf_fp_binned + binned_fn, calving_fp_binned + binned_fn)
                            stats_fn = glacno + '_' + gcm_name + '_' + rcp + '_MCMC_ba1_50sets_2000_2100_all.nc'
                            if os.path.exists(netcdf_fp_stats + stats_fn):
                                shutil.copyfile(netcdf_fp_stats + stats_fn, calving_fp_stats + stats_fn)
    
#                # ----- Missing glaciers -----
#                # Filepath where glaciers are stored
#                # Load the glaciers
#                glacno_list_stats = []
#                for i in os.listdir(netcdf_fp_stats):
#                    if i.endswith('.nc'):
#                        glacno_list_stats.append(i.split('_')[0])
#                glacno_list_stats = sorted(glacno_list_stats)
#                
#                glacno_list_binned = []
#                for i in os.listdir(netcdf_fp_binned):
#                    if i.endswith('.nc'):
#                        glacno_list_binned.append(i.split('_')[0])
#                glacno_list_binned = sorted(glacno_list_binned)
#                
#                glacno_list_all = list(main_glac_rgi_all.glacno.values)
#                
#                A = np.setdiff1d(glacno_list_stats, glacno_list_binned).tolist()
#                B = np.setdiff1d(glacno_list_all, glacno_list_stats).tolist()
#                
#                print(len(B), B)
#                
#                if rcp in ['rcp26']:
#                    C = glacno_list_stats.copy()
#                elif rcp in ['rcp45']:
#                    D = glacno_list_stats.copy()
##                C_dif = np.setdiff1d(D, C).tolist()
                    
                
                #%%

if option_process_data:

    overwrite_pickle = False
    
    grouping = 'all'

    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
        
#    def mwea_to_gta(mwea, area):
#        return mwea * pygem_prms.density_water * area / 1e12
    
    #%%
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')

        # ===== EXPORT RESULTS =====
        success_fullfn = csv_fp + 'CMIP5_success.csv'
        success_cns = ['O1Region', 'count_success', 'count', 'count_%', 'reg_area_km2_success', 'reg_area_km2', 'reg_area_%']
        success_df_single = pd.DataFrame(np.zeros((1,len(success_cns))), columns=success_cns)
        success_df_single.loc[0,:] = [reg, main_glac_rgi.shape[0], main_glac_rgi_all.shape[0],
                                      np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,2),
                                      np.round(main_glac_rgi.Area.sum(),2), np.round(main_glac_rgi_all.Area.sum(),2),
                                      np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,2)]
        if os.path.exists(success_fullfn):
            success_df = pd.read_csv(success_fullfn)
            
            # Add or overwrite existing file
            success_idx = np.where((success_df.O1Region == reg))[0]
            if len(success_idx) > 0:
                success_df.loc[success_idx,:] = success_df_single.values
            else:
                success_df = pd.concat([success_df, success_df_single], axis=0)
                
        else:
            success_df = success_df_single
            
        success_df = success_df.sort_values('O1Region', ascending=True)
        success_df.reset_index(inplace=True, drop=True)
        success_df.to_csv(success_fullfn, index=False)                
        
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size) * degree_size
        main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size) * degree_size
        deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi))]
        main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
        
        # River Basin
        watershed_dict_fn = pygem_prms.main_directory + '/../qgis_datasets/rgi60_watershed_dict.csv'
        watershed_csv = pd.read_csv(watershed_dict_fn)
        watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
        main_glac_rgi['watershed'] = main_glac_rgi.RGIId.map(watershed_dict)
        if len(np.where(main_glac_rgi.watershed.isnull())[0]) > 0:
            main_glac_rgi.loc[np.where(main_glac_rgi.watershed.isnull())[0],'watershed'] = 'nan'
        
        #%%
        # Unique Groups
        # O2 Regions
        unique_regO2s = np.unique(main_glac_rgi['O2Region'])
        
        # Degrees
        if main_glac_rgi['deg_id'].isnull().all():
            unique_degids = None
        else:
            unique_degids = np.unique(main_glac_rgi['deg_id'])
        
        # Watersheds
        if main_glac_rgi['watershed'].isnull().all():
            unique_watersheds = None
        else:
            unique_watersheds = np.unique(main_glac_rgi['watershed'])

        # Elevation bins
        elev_bin_size = 10
        zmax = int(np.ceil(main_glac_rgi.Zmax.max() / elev_bin_size) * elev_bin_size) + 500
        elev_bins = np.arange(0,zmax,elev_bin_size)
        elev_bins = np.insert(elev_bins, 0, -1000)
        
        
        # Pickle datasets
        # Glacier list
        fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
        if not os.path.exists(pickle_fp + str(reg).zfill(2) + '/'):
            os.makedirs(pickle_fp + str(reg).zfill(2) + '/')
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'wb') as f:
            pickle.dump(glacno_list, f)
        
        # O2Region dict
        fn_unique_regO2s = 'R' + str(reg) + '_unique_regO2s.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_regO2s, 'wb') as f:
            pickle.dump(unique_regO2s, f)      
        # Watershed dict
        fn_unique_watersheds = 'R' + str(reg) + '_unique_watersheds.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_watersheds, 'wb') as f:
            pickle.dump(unique_watersheds, f) 
        # Degree ID dict
        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_degids, 'wb') as f:
            pickle.dump(unique_degids, f)
        
        fn_elev_bins = 'R' + str(reg) + '_elev_bins.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_elev_bins, 'wb') as f:
            pickle.dump(elev_bins, f)
        
        #%%
        years = None        
        for gcm_name in gcm_names:
            for rcp in rcps:

                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_reg):
                    os.makedirs(pickle_fp_reg)
                pickle_fp_regO2 =  pickle_fp + str(reg).zfill(2) + '/O2Regions/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_regO2):
                    os.makedirs(pickle_fp_regO2)
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_watershed):
                    os.makedirs(pickle_fp_watershed)
                pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_degid):
                    os.makedirs(pickle_fp_degid)
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                regO2_rcp_gcm_str = 'R' + str(reg) + '_O2Regions_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name
                
                # Volume
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                fn_regO2_vol_annual = regO2_rcp_gcm_str + '_vol_annual.pkl'
                fn_watershed_vol_annual = watershed_rcp_gcm_str + '_vol_annual.pkl'
                fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                # Volume below sea level 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_regO2_vol_annual_bwl = regO2_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_watershed_vol_annual_bwl = watershed_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_degid_vol_annual_bwl = degid_rcp_gcm_str + '_vol_annual_bwl.pkl'
                # Volume below debris
                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_regO2_vol_annual_bd = regO2_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_watershed_vol_annual_bd = watershed_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_degid_vol_annual_bd = degid_rcp_gcm_str + '_vol_annual_bd.pkl'
                # Area 
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_regO2_area_annual = regO2_rcp_gcm_str + '_area_annual.pkl'
                fn_watershed_area_annual = watershed_rcp_gcm_str + '_area_annual.pkl'
                fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                # Area below debris
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_regO2_area_annual_bd = regO2_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_watershed_area_annual_bd = watershed_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_degid_area_annual_bd = degid_rcp_gcm_str + '_area_annual_bd.pkl'
                # Binned Volume
                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_regO2_vol_annual_binned = regO2_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_watershed_vol_annual_binned = watershed_rcp_gcm_str + '_vol_annual_binned.pkl'
                # Binned Volume below debris
                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_regO2_vol_annual_binned_bd = regO2_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_watershed_vol_annual_binned_bd = watershed_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                # Binned Area
                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_regO2_area_annual_binned = regO2_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_watershed_area_annual_binned = watershed_rcp_gcm_str + '_area_annual_binned.pkl'
                # Binned Area below debris
                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_regO2_area_annual_binned_bd = regO2_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_watershed_area_annual_binned_bd = watershed_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                # Mass balance: accumulation
                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                fn_regO2_acc_monthly = regO2_rcp_gcm_str + '_acc_monthly.pkl'
                fn_watershed_acc_monthly = watershed_rcp_gcm_str + '_acc_monthly.pkl'  
                # Mass balance: refreeze
                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_regO2_refreeze_monthly = regO2_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_watershed_refreeze_monthly = watershed_rcp_gcm_str + '_refreeze_monthly.pkl'
                # Mass balance: melt
                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                fn_regO2_melt_monthly = regO2_rcp_gcm_str + '_melt_monthly.pkl'
                fn_watershed_melt_monthly = watershed_rcp_gcm_str + '_melt_monthly.pkl'
                # Mass balance: frontal ablation
                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                fn_regO2_frontalablation_monthly = regO2_rcp_gcm_str + '_frontalablation_monthly.pkl'
                fn_watershed_frontalablation_monthly = watershed_rcp_gcm_str + '_frontalablation_monthly.pkl'
                # Mass balance: total mass balance
                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_regO2_massbaltotal_monthly = regO2_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_watershed_massbaltotal_monthly = watershed_rcp_gcm_str + '_massbaltotal_monthly.pkl' 
                fn_degid_massbaltotal_monthly = degid_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                # Binned Climatic Mass Balance
                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_regO2_mbclim_annual_binned = regO2_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                # Runoff: moving-gauged
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_regO2_runoff_monthly_moving = regO2_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_degid_runoff_monthly_moving = degid_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                # Runoff: fixed-gauged
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_regO2_runoff_monthly_fixed = regO2_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_degid_runoff_monthly_fixed = degid_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                # Runoff: precipitation
                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
                fn_regO2_prec_monthly = regO2_rcp_gcm_str + '_prec_monthly.pkl'
                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
                # Runoff: off-glacier precipitation
                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
                fn_regO2_offglac_prec_monthly = regO2_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                # Runoff: off-glacier melt
                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_regO2_offglac_melt_monthly = regO2_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                # Runoff: off-glacier refreeze
                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_regO2_offglac_refreeze_monthly = regO2_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                # ELA
                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                fn_regO2_ela_annual = regO2_rcp_gcm_str + '_ela_annual.pkl'
                fn_watershed_ela_annual = watershed_rcp_gcm_str + '_ela_annual.pkl'
                # AAR
                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                fn_regO2_aar_annual = regO2_rcp_gcm_str + '_aar_annual.pkl'
                fn_watershed_aar_annual = watershed_rcp_gcm_str + '_aar_annual.pkl'
                    
                if not os.path.exists(pickle_fp_reg + fn_reg_vol_annual) or overwrite_pickle:

                    # Entire region
                    years = None
                    reg_vol_annual = None
                    reg_vol_annual_bwl = None
                    reg_vol_annual_bd = None
                    reg_area_annual = None
                    reg_area_annual_bd = None
                    reg_vol_annual_binned = None
                    reg_vol_annual_binned_bd = None
                    reg_area_annual_binned = None
                    reg_area_annual_binned_bd = None
                    reg_mbclim_annual_binned = None
                    reg_acc_monthly = None
                    reg_refreeze_monthly = None
                    reg_melt_monthly = None
                    reg_frontalablation_monthly = None
                    reg_massbaltotal_monthly = None
                    reg_runoff_monthly_fixed = None
                    reg_runoff_monthly_moving = None
                    reg_prec_monthly = None
                    reg_offglac_prec_monthly = None
                    reg_offglac_melt_monthly = None
                    reg_offglac_refreeze_monthly = None
                    reg_ela_annual = None
                    reg_ela_annual_area = None # used for weighted area calculations
                    reg_area_annual_acc = None
                    reg_area_annual_frombins = None
                    
                    # Subregion groups
                    regO2_vol_annual = None
                    regO2_vol_annual_bwl = None
                    regO2_vol_annual_bd = None
                    regO2_area_annual = None
                    regO2_area_annual_bd = None
                    regO2_vol_annual_binned = None
                    regO2_vol_annual_binned_bd = None
                    regO2_area_annual_binned = None
                    regO2_area_annual_binned_bd = None
                    regO2_mbclim_annual_binned = None
                    regO2_acc_monthly = None
                    regO2_refreeze_monthly = None
                    regO2_melt_monthly = None
                    regO2_frontalablation_monthly = None
                    regO2_massbaltotal_monthly = None
                    regO2_runoff_monthly_fixed = None
                    regO2_runoff_monthly_moving = None
                    regO2_prec_monthly = None
                    regO2_offglac_prec_monthly = None
                    regO2_offglac_melt_monthly = None
                    regO2_offglac_refreeze_monthly = None
                    regO2_ela_annual = None
                    regO2_ela_annual_area = None # used for weighted area calculations
                    regO2_area_annual_acc = None
                    regO2_area_annual_frombins = None
                    
                    # Watershed groups
                    watershed_vol_annual = None
                    watershed_vol_annual_bwl = None
                    watershed_vol_annual_bd = None
                    watershed_area_annual = None
                    watershed_area_annual_bd = None
                    watershed_vol_annual_binned = None
                    watershed_vol_annual_binned_bd = None
                    watershed_area_annual_binned = None
                    watershed_area_annual_binned_bd = None
                    watershed_mbclim_annual_binned = None
                    watershed_acc_monthly = None
                    watershed_refreeze_monthly = None
                    watershed_melt_monthly = None
                    watershed_frontalablation_monthly = None
                    watershed_massbaltotal_monthly = None
                    watershed_runoff_monthly_fixed = None
                    watershed_runoff_monthly_moving = None
                    watershed_prec_monthly = None
                    watershed_offglac_prec_monthly = None
                    watershed_offglac_melt_monthly = None
                    watershed_offglac_refreeze_monthly = None
                    watershed_ela_annual = None
                    watershed_ela_annual_area = None # used for weighted area calculations
                    watershed_area_annual_acc = None
                    watershed_area_annual_frombins = None
    
                    # Degree groups                
                    degid_vol_annual = None
                    degid_vol_annual_bwl = None
                    degid_vol_annual_bd = None
                    degid_area_annual = None
                    degid_area_annual_bd = None
                    degid_massbaltotal_monthly = None
                    degid_runoff_monthly_fixed = None
                    degid_runoff_monthly_moving = None
    
    
                    for nglac, glacno in enumerate(glacno_list):
                        if nglac%10 == 0:
                            print(gcm_name, rcp, glacno)
                        
                        # Group indices
                        glac_idx = np.where(main_glac_rgi['glacno'] == glacno)[0][0]
                        regO2 = main_glac_rgi.loc[glac_idx, 'O2Region']
                        regO2_idx = np.where(regO2 == unique_regO2s)[0][0]
                        watershed = main_glac_rgi.loc[glac_idx,'watershed']
                        watershed_idx = np.where(watershed == unique_watersheds)
                        degid = main_glac_rgi.loc[glac_idx, 'deg_id']
                        degid_idx = np.where(degid == unique_degids)[0][0]
                        
                        # Filenames
                        nsim_strs = ['50', '1', '100', '150', '200', '250']
                        ds_binned = None
                        nset = -1
                        while ds_binned is None and nset <= len(nsim_strs):
                            nset += 1
                            nsim_str = nsim_strs[nset]
                            
                            try:
                                netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                netcdf_fn_binned = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
        
                                netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                
                                # Open files
                                ds_binned = xr.open_dataset(netcdf_fp_binned + '/' + netcdf_fn_binned)
                                ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                            except:
                                ds_binned = None
                            
                        # Years
                        if years is None:
                            years = ds_stats.year.values
                            
                                
                        # ----- 1. Volume (m3) vs. Year ----- 
                        glac_vol_annual = ds_stats.glac_volume_annual.values[0,:]
                        # All
                        if reg_vol_annual is None:
                            reg_vol_annual = glac_vol_annual
                        else:
                            reg_vol_annual += glac_vol_annual
                        # O2Region
                        if regO2_vol_annual is None:
                            regO2_vol_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_vol_annual[regO2_idx,:] = glac_vol_annual
                        else:
                            regO2_vol_annual[regO2_idx,:] += glac_vol_annual
                        # Watershed
                        if watershed_vol_annual is None:
                            watershed_vol_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_vol_annual[watershed_idx,:] = glac_vol_annual
                        else:
                            watershed_vol_annual[watershed_idx,:] += glac_vol_annual
                        # DegId
                        if degid_vol_annual is None:
                            degid_vol_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_vol_annual[degid_idx,:] = glac_vol_annual
                        else:
                            degid_vol_annual[degid_idx,:] += glac_vol_annual
                            
                            
                        # ----- 2. Volume below-sea-level (m3) vs. Year ----- 
                        #  - initial elevation is stored
                        #  - bed elevation is constant in time
                        #  - assume sea level is at 0 m a.s.l.
                        z_sealevel = 0
                        bin_z_init = ds_binned.bin_surface_h_initial.values[0,:]
                        bin_thick_annual = ds_binned.bin_thick_annual.values[0,:,:]
                        bin_z_bed = bin_z_init - bin_thick_annual[:,0]
                        # Annual surface height
                        bin_z_surf_annual = bin_z_bed[:,np.newaxis] + bin_thick_annual
                        
                        # Annual volume (m3)
                        bin_vol_annual = ds_binned.bin_volume_annual.values[0,:,:]
                        # Annual area (m2)
                        bin_area_annual = np.zeros(bin_vol_annual.shape)
                        bin_area_annual[bin_vol_annual > 0] = (
                                bin_vol_annual[bin_vol_annual > 0] / bin_thick_annual[bin_vol_annual > 0])
                        
                        # Processed based on OGGM's _vol_below_level function
                        bwl = (bin_z_bed[:,np.newaxis] < 0) & (bin_thick_annual > 0)
                        if bwl.any():
                            # Annual surface height (max of sea level for calcs)
                            bin_z_surf_annual_bwl = bin_z_surf_annual.copy()
                            bin_z_surf_annual_bwl[bin_z_surf_annual_bwl > z_sealevel] = z_sealevel
                            # Annual thickness below sea level (m)
                            bin_thick_annual_bwl = bin_thick_annual.copy()
                            bin_thick_annual_bwl = bin_z_surf_annual_bwl - bin_z_bed[:,np.newaxis]
                            bin_thick_annual_bwl[~bwl] = 0
                            # Annual volume below sea level (m3)
                            bin_vol_annual_bwl = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bwl[bwl] = bin_thick_annual_bwl[bwl] * bin_area_annual[bwl]
                            glac_vol_annual_bwl = bin_vol_annual_bwl.sum(0)
                            
                            # All
                            if reg_vol_annual_bwl is None:
                                reg_vol_annual_bwl = glac_vol_annual_bwl
                            else:
                                reg_vol_annual_bwl += glac_vol_annual_bwl
                            # O2Region
                            if regO2_vol_annual_bwl is None:
                                regO2_vol_annual_bwl = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bwl[regO2_idx,:] = glac_vol_annual_bwl
                            else:
                                regO2_vol_annual_bwl[regO2_idx,:] += glac_vol_annual_bwl
                            # Watershed
                            if watershed_vol_annual_bwl is None:
                                watershed_vol_annual_bwl = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bwl[watershed_idx,:] = glac_vol_annual_bwl
                            else:
                                watershed_vol_annual_bwl[watershed_idx,:] += glac_vol_annual_bwl
                            # DegId
                            if degid_vol_annual_bwl is None:
                                degid_vol_annual_bwl = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bwl[degid_idx,:] = glac_vol_annual_bwl
                            else:
                                degid_vol_annual_bwl[degid_idx,:] += glac_vol_annual_bwl
                        
    
                        # ----- 3. Volume below-debris vs. Time ----- 
                        gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
                        fls = gdir.read_pickle('inversion_flowlines')
                        bin_debris_hd = np.zeros(bin_z_init.shape)
                        bin_debris_ed = np.zeros(bin_z_init.shape) + 1
                        if 'debris_hd' in dir(fls[0]):
                            bin_debris_hd[0:fls[0].debris_hd.shape[0]] = fls[0].debris_hd
                            bin_debris_ed[0:fls[0].debris_hd.shape[0]] = fls[0].debris_ed
                        if bin_debris_hd.sum() > 0:
                            bin_vol_annual_bd = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bd[bin_debris_hd > 0, :] = bin_vol_annual[bin_debris_hd > 0, :]
                            glac_vol_annual_bd = bin_vol_annual_bd.sum(0)
                            
                            # All
                            if reg_vol_annual_bd is None:
                                reg_vol_annual_bd = glac_vol_annual_bd
                            else:
                                reg_vol_annual_bd += glac_vol_annual_bd
                            # O2Region
                            if regO2_vol_annual_bd is None:
                                regO2_vol_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bd[regO2_idx,:] = glac_vol_annual_bd
                            else:
                                regO2_vol_annual_bd[regO2_idx,:] += glac_vol_annual_bd
                            # Watershed
                            if watershed_vol_annual_bd is None:
                                watershed_vol_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bd[watershed_idx,:] = glac_vol_annual_bd
                            else:
                                watershed_vol_annual_bd[watershed_idx,:] += glac_vol_annual_bd
                            # DegId
                            if degid_vol_annual_bd is None:
                                degid_vol_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bd[degid_idx,:] = glac_vol_annual_bd
                            else:
                                degid_vol_annual_bd[degid_idx,:] += glac_vol_annual_bd
                        
                        
                        # ----- 4. Area vs. Time ----- 
                        glac_area_annual = ds_stats.glac_area_annual.values[0,:]
                        # All
                        if reg_area_annual is None:
                            reg_area_annual = glac_area_annual
                        else:
                            reg_area_annual += glac_area_annual
                        # O2Region
                        if regO2_area_annual is None:
                            regO2_area_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual[regO2_idx,:] = glac_area_annual
                        else:
                            regO2_area_annual[regO2_idx,:] += glac_area_annual
                        # Watershed
                        if watershed_area_annual is None:
                            watershed_area_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual[watershed_idx,:] = glac_area_annual
                        else:
                            watershed_area_annual[watershed_idx,:] += glac_area_annual
                        # DegId
                        if degid_area_annual is None:
                            degid_area_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_area_annual[degid_idx,:] = glac_area_annual
                        else:
                            degid_area_annual[degid_idx,:] += glac_area_annual
                        
                        
                        # ----- 5. Area below-debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            bin_area_annual_bd = np.zeros(bin_area_annual.shape)
                            bin_area_annual_bd[bin_debris_hd > 0, :] = bin_area_annual[bin_debris_hd > 0, :]
                            glac_area_annual_bd = bin_area_annual_bd.sum(0)
                            
                            # All
                            if reg_area_annual_bd is None:
                                reg_area_annual_bd = glac_area_annual_bd
                            else:
                                reg_area_annual_bd += glac_area_annual_bd
                            # O2Region
                            if regO2_area_annual_bd is None:
                                regO2_area_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_area_annual_bd[regO2_idx,:] = glac_area_annual_bd
                            else:
                                regO2_area_annual_bd[regO2_idx,:] += glac_area_annual_bd
                            # Watershed
                            if watershed_area_annual_bd is None:
                                watershed_area_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_area_annual_bd[watershed_idx,:] = glac_area_annual_bd
                            else:
                                watershed_area_annual_bd[watershed_idx,:] += glac_area_annual_bd
                            # DegId
                            if degid_area_annual_bd is None:
                                degid_area_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_area_annual_bd[degid_idx,:] = glac_area_annual_bd
                            else:
                                degid_area_annual_bd[degid_idx,:] += glac_area_annual_bd
                        
                        
                        # ----- 6. Binned glacier volume vs. Time ----- 
                        bin_vol_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_vol_annual[:,ncol])
                            bin_vol_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_vol_annual_binned is None:
                            reg_vol_annual_binned = bin_vol_annual_10m
                        else:
                            reg_vol_annual_binned += bin_vol_annual_10m
                        # O2Region
                        if regO2_vol_annual_binned is None:
                            regO2_vol_annual_binned = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                            regO2_vol_annual_binned[regO2_idx,:,:] = bin_vol_annual_10m
                        else:
                            regO2_vol_annual_binned[regO2_idx,:,:] += bin_vol_annual_10m
                        # Watershed
                        if watershed_vol_annual_binned is None:
                            watershed_vol_annual_binned = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                            watershed_vol_annual_binned[watershed_idx,:,:] = bin_vol_annual_10m
                        else:
                            watershed_vol_annual_binned[watershed_idx,:,:] += bin_vol_annual_10m
                        
    
                        # ----- 7. Binned glacier volume below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_vol_annual_10m_bd = bin_vol_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_vol_annual_binned_bd is None:
                                reg_vol_annual_binned_bd = bin_vol_annual_10m_bd
                            else:
                                reg_vol_annual_binned_bd += bin_vol_annual_10m_bd
                            # O2Region
                            if regO2_vol_annual_binned_bd is None:
                                regO2_vol_annual_binned_bd = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] += bin_vol_annual_10m_bd
                            # Watershed
                            if watershed_vol_annual_binned_bd is None:
                                watershed_vol_annual_binned_bd = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] += bin_vol_annual_10m_bd
    
    
                        # ----- 8. Binned glacier area vs. Time ----- 
                        bin_area_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_area_annual[:,ncol])
                            bin_area_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_area_annual_binned is None:
                            reg_area_annual_binned = bin_area_annual_10m
                        else:
                            reg_area_annual_binned += bin_area_annual_10m
                        # O2Region
                        if regO2_area_annual_binned is None:
                            regO2_area_annual_binned = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                            regO2_area_annual_binned[regO2_idx,:,:] = bin_area_annual_10m
                        else:
                            regO2_area_annual_binned[regO2_idx,:,:] += bin_area_annual_10m
                        # Watershed
                        if watershed_area_annual_binned is None:
                            watershed_area_annual_binned = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                            watershed_area_annual_binned[watershed_idx,:,:] = bin_area_annual_10m
                        else:
                            watershed_area_annual_binned[watershed_idx,:,:] += bin_area_annual_10m
    
    
                        
                        # ----- 9. Binned glacier area below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_area_annual_10m_bd = bin_area_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_area_annual_binned_bd is None:
                                reg_area_annual_binned_bd = bin_area_annual_10m_bd
                            else:
                                reg_area_annual_binned_bd += bin_area_annual_10m_bd
                            # O2Region
                            if regO2_area_annual_binned_bd is None:
                                regO2_area_annual_binned_bd = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                                regO2_area_annual_binned_bd[regO2_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                regO2_area_annual_binned_bd[regO2_idx,:,:] += bin_area_annual_10m_bd
                            # Watershed
                            if watershed_area_annual_binned_bd is None:
                                watershed_area_annual_binned_bd = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                                watershed_area_annual_binned_bd[watershed_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                watershed_area_annual_binned_bd[watershed_idx,:,:] += bin_area_annual_10m_bd
                                
    
                        # ----- 10. Mass Balance Components vs. Time -----
                        # - these are only meant for monthly and/or relative purposes 
                        #   mass balance from volume change should be used for annual changes
                        # Accumulation
                        glac_acc_monthly = ds_stats.glac_acc_monthly.values[0,:]
                        # All
                        if reg_acc_monthly is None:
                            reg_acc_monthly = glac_acc_monthly
                        else:
                            reg_acc_monthly += glac_acc_monthly
                        # O2Region
                        if regO2_acc_monthly is None:
                            regO2_acc_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_acc_monthly[regO2_idx,:] = glac_acc_monthly
                        else:
                            regO2_acc_monthly[regO2_idx,:] += glac_acc_monthly
                        # Watershed
                        if watershed_acc_monthly is None:
                            watershed_acc_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_acc_monthly[watershed_idx,:] = glac_acc_monthly
                        else:
                            watershed_acc_monthly[watershed_idx,:] += glac_acc_monthly
                        
                        
                        # Refreeze
                        glac_refreeze_monthly = ds_stats.glac_refreeze_monthly.values[0,:]
                        # All
                        if reg_refreeze_monthly is None:
                            reg_refreeze_monthly = glac_refreeze_monthly
                        else:
                            reg_refreeze_monthly += glac_refreeze_monthly
                        # O2Region
                        if regO2_refreeze_monthly is None:
                            regO2_refreeze_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_refreeze_monthly[regO2_idx,:] = glac_refreeze_monthly
                        else:
                            regO2_refreeze_monthly[regO2_idx,:] += glac_refreeze_monthly
                        # Watershed
                        if watershed_refreeze_monthly is None:
                            watershed_refreeze_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_refreeze_monthly[watershed_idx,:] = glac_refreeze_monthly
                        else:
                            watershed_refreeze_monthly[watershed_idx,:] += glac_refreeze_monthly
                            
                        # Melt
                        glac_melt_monthly = ds_stats.glac_melt_monthly.values[0,:]
                        # All
                        if reg_melt_monthly is None:
                            reg_melt_monthly = glac_melt_monthly
                        else:
                            reg_melt_monthly += glac_melt_monthly
                        # O2Region
                        if regO2_melt_monthly is None:
                            regO2_melt_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_melt_monthly[regO2_idx,:] = glac_melt_monthly
                        else:
                            regO2_melt_monthly[regO2_idx,:] += glac_melt_monthly
                        # Watershed
                        if watershed_melt_monthly is None:
                            watershed_melt_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_melt_monthly[watershed_idx,:] = glac_melt_monthly
                        else:
                            watershed_melt_monthly[watershed_idx,:] += glac_melt_monthly
                            
                        # Frontal Ablation
                        glac_frontalablation_monthly = ds_stats.glac_frontalablation_monthly.values[0,:]
                        # All
                        if reg_frontalablation_monthly is None:
                            reg_frontalablation_monthly = glac_frontalablation_monthly
                        else:
                            reg_frontalablation_monthly += glac_frontalablation_monthly
                        # O2Region
                        if regO2_frontalablation_monthly is None:
                            regO2_frontalablation_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_frontalablation_monthly[regO2_idx,:] = glac_frontalablation_monthly
                        else:
                            regO2_frontalablation_monthly[regO2_idx,:] += glac_frontalablation_monthly
                        # Watershed
                        if watershed_frontalablation_monthly is None:
                            watershed_frontalablation_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_frontalablation_monthly[watershed_idx,:] = glac_frontalablation_monthly
                        else:
                            watershed_frontalablation_monthly[watershed_idx,:] += glac_frontalablation_monthly
                            
                        # Total Mass Balance
                        glac_massbaltotal_monthly = ds_stats.glac_massbaltotal_monthly.values[0,:]
                        # All
                        if reg_massbaltotal_monthly is None:
                            reg_massbaltotal_monthly = glac_massbaltotal_monthly
                        else:
                            reg_massbaltotal_monthly += glac_massbaltotal_monthly
                        # O2Region
                        if regO2_massbaltotal_monthly is None:
                            regO2_massbaltotal_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_massbaltotal_monthly[regO2_idx,:] = glac_massbaltotal_monthly
                        else:
                            regO2_massbaltotal_monthly[regO2_idx,:] += glac_massbaltotal_monthly
                        # Watershed
                        if watershed_massbaltotal_monthly is None:
                            watershed_massbaltotal_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_massbaltotal_monthly[watershed_idx,:] = glac_massbaltotal_monthly
                        else:
                            watershed_massbaltotal_monthly[watershed_idx,:] += glac_massbaltotal_monthly
                        # DegId
                        if degid_massbaltotal_monthly is None:
                            degid_massbaltotal_monthly = np.zeros((len(unique_degids),glac_acc_monthly.shape[0]))
                            degid_massbaltotal_monthly[degid_idx,:] = glac_massbaltotal_monthly
                        else:
                            degid_massbaltotal_monthly[degid_idx,:] += glac_massbaltotal_monthly
                        
                        
                        
                        # ----- 11. Binned Climatic Mass Balance vs. Time -----
                        # - Various mass balance datasets may have slight mismatch due to averaging
                        #   ex. mbclim_annual was reported in mwe, so the area average will cause difference
                        #   ex. mbtotal_monthly was averaged on a monthly basis, so the temporal average will cause difference
                        bin_mbclim_annual = ds_binned.bin_massbalclim_annual.values[0,:,:]
                        bin_mbclim_annual_m3we = bin_mbclim_annual * bin_area_annual
    
    #                    glac_massbaltotal_annual_0 = bin_mbclim_annual_m3we.sum(0)
    #                    glac_massbaltotal_annual_1 = glac_massbaltotal_monthly.reshape(-1,12).sum(1)
    #                    glac_massbaltotal_annual_2 = ((glac_vol_annual[1:] - glac_vol_annual[0:-1]) * 
    #                                                  pygem_prms.density_ice / pygem_prms.density_water)
                        
                        bin_mbclim_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_mbclim_annual_m3we[:,ncol])
                            bin_mbclim_annual_10m[:,ncol] = bin_counts
                        # All
                        if reg_mbclim_annual_binned is None:
                            reg_mbclim_annual_binned = bin_mbclim_annual_10m
                        else:
                            reg_mbclim_annual_binned += bin_mbclim_annual_10m
                        # O2Region
                        if regO2_mbclim_annual_binned is None:
                            regO2_mbclim_annual_binned = np.zeros((len(unique_regO2s), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            regO2_mbclim_annual_binned[regO2_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            regO2_mbclim_annual_binned[regO2_idx,:,:] += bin_mbclim_annual_10m
                        # Watershed
                        if watershed_mbclim_annual_binned is None:
                            watershed_mbclim_annual_binned = np.zeros((len(unique_watersheds), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            watershed_mbclim_annual_binned[watershed_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            watershed_mbclim_annual_binned[watershed_idx,:,:] += bin_mbclim_annual_10m
    
                        
                        # ----- 12. Runoff vs. Time -----
                        glac_runoff_monthly = ds_stats.glac_runoff_monthly.values[0,:]
                        # Moving-gauge Runoff vs. Time
                        # All
                        if reg_runoff_monthly_moving is None:
                            reg_runoff_monthly_moving = glac_runoff_monthly
                        else:
                            reg_runoff_monthly_moving += glac_runoff_monthly
                        # O2Region
                        if regO2_runoff_monthly_moving is None:
                            regO2_runoff_monthly_moving = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_runoff_monthly_moving[regO2_idx,:] = glac_runoff_monthly
                        else:
                            regO2_runoff_monthly_moving[regO2_idx,:] += glac_runoff_monthly
                        # watershed
                        if watershed_runoff_monthly_moving is None:
                            watershed_runoff_monthly_moving = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_runoff_monthly_moving[watershed_idx,:] = glac_runoff_monthly
                        else:
                            watershed_runoff_monthly_moving[watershed_idx,:] += glac_runoff_monthly
                        # DegId
                        if degid_runoff_monthly_moving is None:
                            degid_runoff_monthly_moving = np.zeros((len(unique_degids),glac_acc_monthly.shape[0]))
                            degid_runoff_monthly_moving[degid_idx,:] = glac_runoff_monthly
                        else:
                            degid_runoff_monthly_moving[degid_idx,:] += glac_runoff_monthly
                            
                        # Fixed-gauge Runoff vs. Time
                        offglac_runoff_monthly = ds_stats.offglac_runoff_monthly.values[0,:]
                        glac_runoff_monthly_fixed = glac_runoff_monthly + offglac_runoff_monthly
                        # All
                        if reg_runoff_monthly_fixed is None:
                            reg_runoff_monthly_fixed = glac_runoff_monthly_fixed
                        else:
                            reg_runoff_monthly_fixed += glac_runoff_monthly_fixed
                        # O2Region
                        if regO2_runoff_monthly_fixed is None:
                            regO2_runoff_monthly_fixed = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_runoff_monthly_fixed[regO2_idx,:] = glac_runoff_monthly_fixed
                        else:
                            regO2_runoff_monthly_fixed[regO2_idx,:] += glac_runoff_monthly_fixed
                        # Watershed
                        if watershed_runoff_monthly_fixed is None:
                            watershed_runoff_monthly_fixed = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_runoff_monthly_fixed[watershed_idx,:] = glac_runoff_monthly_fixed
                        else:
                            watershed_runoff_monthly_fixed[watershed_idx,:] += glac_runoff_monthly_fixed
                        # DegId
                        if degid_runoff_monthly_fixed is None:
                            degid_runoff_monthly_fixed = np.zeros((len(unique_degids),glac_acc_monthly.shape[0]))
                            degid_runoff_monthly_fixed[degid_idx,:] = glac_runoff_monthly_fixed
                        else:
                            degid_runoff_monthly_fixed[degid_idx,:] += glac_runoff_monthly_fixed
                        
                        
                        # Runoff Components
                        # Precipitation
                        glac_prec_monthly = ds_stats.glac_prec_monthly.values[0,:]
                        # All
                        if reg_prec_monthly is None:
                            reg_prec_monthly = glac_prec_monthly
                        else:
                            reg_prec_monthly += glac_prec_monthly
                        # O2Region
                        if regO2_prec_monthly is None:
                            regO2_prec_monthly = np.zeros((len(unique_regO2s),glac_prec_monthly.shape[0]))
                            regO2_prec_monthly[regO2_idx,:] = glac_prec_monthly
                        else:
                            regO2_prec_monthly[regO2_idx,:] += glac_prec_monthly
                        # Watershed
                        if watershed_prec_monthly is None:
                            watershed_prec_monthly = np.zeros((len(unique_watersheds),glac_prec_monthly.shape[0]))
                            watershed_prec_monthly[watershed_idx,:] = glac_prec_monthly
                        else:
                            watershed_prec_monthly[watershed_idx,:] += glac_prec_monthly
                            
                        # Off-glacier Precipitation
                        offglac_prec_monthly = ds_stats.offglac_prec_monthly.values[0,:]
                        # All
                        if reg_offglac_prec_monthly is None:
                            reg_offglac_prec_monthly = offglac_prec_monthly
                        else:
                            reg_offglac_prec_monthly += offglac_prec_monthly
                        # O2Region
                        if regO2_offglac_prec_monthly is None:
                            regO2_offglac_prec_monthly = np.zeros((len(unique_regO2s),glac_prec_monthly.shape[0]))
                            regO2_offglac_prec_monthly[regO2_idx,:] = offglac_prec_monthly
                        else:
                            regO2_offglac_prec_monthly[regO2_idx,:] += offglac_prec_monthly
                        # Watershed
                        if watershed_offglac_prec_monthly is None:
                            watershed_offglac_prec_monthly = np.zeros((len(unique_watersheds),glac_prec_monthly.shape[0]))
                            watershed_offglac_prec_monthly[watershed_idx,:] = offglac_prec_monthly
                        else:
                            watershed_offglac_prec_monthly[watershed_idx,:] += offglac_prec_monthly
                            
                        # Off-glacier Melt
                        offglac_melt_monthly = ds_stats.offglac_melt_monthly.values[0,:]
                        # All
                        if reg_offglac_melt_monthly is None:
                            reg_offglac_melt_monthly = offglac_melt_monthly
                        else:
                            reg_offglac_melt_monthly += offglac_melt_monthly
                        # O2Region
                        if regO2_offglac_melt_monthly is None:
                            regO2_offglac_melt_monthly = np.zeros((len(unique_regO2s),glac_melt_monthly.shape[0]))
                            regO2_offglac_melt_monthly[regO2_idx,:] = offglac_melt_monthly
                        else:
                            regO2_offglac_melt_monthly[regO2_idx,:] += offglac_melt_monthly
                        # Watershed
                        if watershed_offglac_melt_monthly is None:
                            watershed_offglac_melt_monthly = np.zeros((len(unique_watersheds),glac_melt_monthly.shape[0]))
                            watershed_offglac_melt_monthly[watershed_idx,:] = offglac_melt_monthly
                        else:
                            watershed_offglac_melt_monthly[watershed_idx,:] += offglac_melt_monthly
                            
                        # Off-glacier Refreeze
                        # All
                        offglac_refreeze_monthly = ds_stats.offglac_refreeze_monthly.values[0,:]
                        if reg_offglac_refreeze_monthly is None:
                            reg_offglac_refreeze_monthly = offglac_refreeze_monthly
                        else:
                            reg_offglac_refreeze_monthly += offglac_refreeze_monthly
                        # O2Region
                        if regO2_offglac_refreeze_monthly is None:
                            regO2_offglac_refreeze_monthly = np.zeros((len(unique_regO2s),glac_refreeze_monthly.shape[0]))
                            regO2_offglac_refreeze_monthly[regO2_idx,:] = offglac_refreeze_monthly
                        else:
                            regO2_offglac_refreeze_monthly[regO2_idx,:] += offglac_refreeze_monthly
                        # Watershed
                        if watershed_offglac_refreeze_monthly is None:
                            watershed_offglac_refreeze_monthly = np.zeros((len(unique_watersheds),glac_refreeze_monthly.shape[0]))
                            watershed_offglac_refreeze_monthly[watershed_idx,:] = offglac_refreeze_monthly
                        else:
                            watershed_offglac_refreeze_monthly[watershed_idx,:] += offglac_refreeze_monthly
    
                        # ----- 13. ELA vs. Time -----
                        glac_ela_annual = ds_stats.glac_ELA_annual.values[0,:]
                        if np.isnan(glac_ela_annual).any():
                            # Quality control nan values 
                            #  - replace with max elev because occur when entire glacier has neg mb
                            bin_z_surf_annual_glaconly = bin_z_surf_annual.copy()
                            bin_z_surf_annual_glaconly[bin_thick_annual == 0] = np.nan
                            zmax_annual = np.nanmax(bin_z_surf_annual_glaconly, axis=0)
                            glac_ela_annual[np.isnan(glac_ela_annual)] = zmax_annual[np.isnan(glac_ela_annual)]
    
                        # Area-weighted ELA
                        # All
                        if reg_ela_annual is None:
                            reg_ela_annual = glac_ela_annual
                            reg_ela_annual_area = glac_area_annual.copy()
                        else:
                            # Use index to avoid dividing by 0 when glacier completely melts                            
                            ela_idx = np.where(reg_ela_annual_area + glac_area_annual > 0)[0]
                            reg_ela_annual[ela_idx] = (
                                    (reg_ela_annual[ela_idx] * reg_ela_annual_area[ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                    (reg_ela_annual_area[ela_idx] + glac_area_annual[ela_idx]))
                            reg_ela_annual_area += glac_area_annual
                        
                        # O2Region
                        if regO2_ela_annual is None:
                            regO2_ela_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual[regO2_idx,:] = glac_ela_annual
                            regO2_ela_annual_area = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual_area[regO2_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(regO2_ela_annual_area[regO2_idx,:] + glac_area_annual > 0)[0]
                            regO2_ela_annual[regO2_idx,ela_idx] = (
                                    (regO2_ela_annual[regO2_idx,ela_idx] * regO2_ela_annual_area[regO2_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (regO2_ela_annual_area[regO2_idx,ela_idx] + glac_area_annual[ela_idx]))
                            regO2_ela_annual_area[regO2_idx,:] += glac_area_annual
                        
                        # Watershed
                        if watershed_ela_annual is None:
                            watershed_ela_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual[watershed_idx,:] = glac_ela_annual
                            watershed_ela_annual_area = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual_area[watershed_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(watershed_ela_annual_area[watershed_idx,:] + glac_area_annual > 0)[0]
                            watershed_ela_annual[watershed_idx,ela_idx] = (
                                    (watershed_ela_annual[watershed_idx,ela_idx] * watershed_ela_annual_area[watershed_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (watershed_ela_annual_area[watershed_idx,ela_idx] + glac_area_annual[ela_idx]))
                            watershed_ela_annual_area[watershed_idx,:] += glac_area_annual
                        

                        # ----- 14. AAR vs. Time -----
                        #  - averaging issue with bin_area_annual.sum(0) != glac_area_annual
                        #  - hence only use these 
                        bin_area_annual_acc = bin_area_annual.copy()
                        bin_area_annual_acc[bin_mbclim_annual <= 0] = 0
                        glac_area_annual_acc = bin_area_annual_acc.sum(0)
                        glac_area_annual_frombins = bin_area_annual.sum(0)
                        
                        # All
                        if reg_area_annual_acc is None:
                            reg_area_annual_acc = glac_area_annual_acc.copy()
                            reg_area_annual_frombins = glac_area_annual_frombins.copy()
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        else:
                            reg_area_annual_acc += glac_area_annual_acc
                            reg_area_annual_frombins += glac_area_annual_frombins
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        # O2Regions
                        if regO2_area_annual_acc is None:
                            regO2_area_annual_acc = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_acc[regO2_idx,:] = glac_area_annual_acc.copy()
                            regO2_area_annual_frombins = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_frombins[regO2_idx,:] = glac_area_annual_frombins.copy()
                            regO2_aar_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        else:
                            regO2_area_annual_acc[regO2_idx,:] += glac_area_annual_acc
                            regO2_area_annual_frombins[regO2_idx,:] += glac_area_annual_frombins
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        # Watersheds
                        if watershed_area_annual_acc is None:
                            watershed_area_annual_acc = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_acc[watershed_idx,:] = glac_area_annual_acc.copy()
                            watershed_area_annual_frombins = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_frombins[watershed_idx,:] = glac_area_annual_frombins.copy()
                            watershed_aar_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        else:
                            watershed_area_annual_acc[watershed_idx,:] += glac_area_annual_acc
                            watershed_area_annual_frombins[watershed_idx,:] += glac_area_annual_frombins
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        
                    # ===== PICKLE DATASETS =====
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'wb') as f:
                        pickle.dump(reg_vol_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'wb') as f:
                        pickle.dump(regO2_vol_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'wb') as f:
                        pickle.dump(watershed_vol_annual, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'wb') as f:
                        pickle.dump(degid_vol_annual, f)
                    # Volume below sea level 
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'wb') as f:
                        pickle.dump(reg_vol_annual_bwl, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bwl, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bwl, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'wb') as f:
                        pickle.dump(degid_vol_annual_bwl, f) 
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'wb') as f:
                        pickle.dump(degid_vol_annual_bd, f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'wb') as f:
                        pickle.dump(reg_area_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'wb') as f:
                        pickle.dump(regO2_area_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'wb') as f:
                        pickle.dump(watershed_area_annual, f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'wb') as f:
                        pickle.dump(degid_area_annual, f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'wb') as f:
                        pickle.dump(degid_area_annual_bd, f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned, f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned_bd, f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'wb') as f:
                        pickle.dump(reg_area_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned, f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned_bd, f)
                    # Mass balance: accumulation
                    with open(pickle_fp_reg + fn_reg_acc_monthly, 'wb') as f:
                        pickle.dump(reg_acc_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_acc_monthly, 'wb') as f:
                        pickle.dump(regO2_acc_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_acc_monthly, 'wb') as f:
                        pickle.dump(watershed_acc_monthly, f)
                    # Mass balance: refreeze
                    with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'wb') as f:
                        pickle.dump(reg_refreeze_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_refreeze_monthly, 'wb') as f:
                        pickle.dump(regO2_refreeze_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_refreeze_monthly, 'wb') as f:
                        pickle.dump(watershed_refreeze_monthly, f)
                    # Mass balance: melt
                    with open(pickle_fp_reg + fn_reg_melt_monthly, 'wb') as f:
                        pickle.dump(reg_melt_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_melt_monthly, 'wb') as f:
                        pickle.dump(regO2_melt_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_melt_monthly, 'wb') as f:
                        pickle.dump(watershed_melt_monthly, f)
                    # Mass balance: frontal ablation
                    with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'wb') as f:
                        pickle.dump(reg_frontalablation_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_frontalablation_monthly, 'wb') as f:
                        pickle.dump(regO2_frontalablation_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_frontalablation_monthly, 'wb') as f:
                        pickle.dump(watershed_frontalablation_monthly, f)
                    # Mass balance: total mass balance
                    with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(reg_massbaltotal_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(regO2_massbaltotal_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(watershed_massbaltotal_monthly, f)
                    with open(pickle_fp_degid + fn_degid_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(degid_massbaltotal_monthly, f)  
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(reg_mbclim_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(regO2_mbclim_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(watershed_mbclim_annual_binned, f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_moving, f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_moving, f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_moving, f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_moving, f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_fixed, f)     
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_fixed, f)  
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_fixed, f)  
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_fixed, f)  
                    # Runoff: precipitation
                    with open(pickle_fp_reg + fn_reg_prec_monthly, 'wb') as f:
                        pickle.dump(reg_prec_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_prec_monthly, 'wb') as f:
                        pickle.dump(regO2_prec_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'wb') as f:
                        pickle.dump(watershed_prec_monthly, f)
                    # Runoff: off-glacier precipitation
                    with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_prec_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_prec_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(watershed_offglac_prec_monthly, f)
                    # Runoff: off-glacier melt
                    with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_melt_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_melt_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(watershed_offglac_melt_monthly, f)
                    # Runoff: off-glacier refreeze
                    with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_refreeze_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_refreeze_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(watershed_offglac_refreeze_monthly, f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'wb') as f:
                        pickle.dump(reg_ela_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'wb') as f:
                        pickle.dump(regO2_ela_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'wb') as f:
                        pickle.dump(watershed_ela_annual, f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'wb') as f:
                        pickle.dump(reg_aar_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'wb') as f:
                        pickle.dump(regO2_aar_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'wb') as f:
                        pickle.dump(watershed_aar_annual, f)
                        
                # ----- OTHERWISE LOAD THE PROCESSED DATASETS -----
                else:
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                        reg_vol_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'rb') as f:
                        regO2_vol_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'rb') as f:
                        watershed_vol_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                        degid_vol_annual = pickle.load(f)
                    # Volume below sea level
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'rb') as f:
                        regO2_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'rb') as f:
                        watershed_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'rb') as f:
                        degid_vol_annual_bwl = pickle.load(f)
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                        reg_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'rb') as f:
                        regO2_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'rb') as f:
                        watershed_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'rb') as f:
                        degid_vol_annual_bd = pickle.load(f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                        reg_area_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'rb') as f:
                        regO2_area_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'rb') as f:
                        watershed_area_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                        degid_area_annual = pickle.load(f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                        reg_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'rb') as f:
                        regO2_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'rb') as f:
                        watershed_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'rb') as f:
                        degid_area_annual_bd = pickle.load(f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                        reg_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'rb') as f:
                        regO2_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'rb') as f:
                        watershed_vol_annual_binned = pickle.load(f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                        reg_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'rb') as f:
                        regO2_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'rb') as f:
                        watershed_vol_annual_binned_bd = pickle.load(f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                        reg_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'rb') as f:
                        regO2_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'rb') as f:
                        watershed_area_annual_binned = pickle.load(f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                        reg_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'rb') as f:
                        regO2_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'rb') as f:
                        watershed_area_annual_binned_bd = pickle.load(f)
                    # Mass balance: accumulation
                    with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                        reg_acc_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_acc_monthly, 'rb') as f:
                        regO2_acc_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_acc_monthly, 'rb') as f:
                        watershed_acc_monthly = pickle.load(f)
                    # Mass balance: refreeze
                    with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                        reg_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_refreeze_monthly, 'rb') as f:
                        regO2_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_refreeze_monthly, 'rb') as f:
                        watershed_refreeze_monthly = pickle.load(f)
                    # Mass balance: melt
                    with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                        reg_melt_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_melt_monthly, 'rb') as f:
                        regO2_melt_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_melt_monthly, 'rb') as f:
                        watershed_melt_monthly = pickle.load(f)
                    # Mass balance: frontal ablation
                    with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                        reg_frontalablation_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_frontalablation_monthly, 'rb') as f:
                        regO2_frontalablation_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_frontalablation_monthly, 'rb') as f:
                        watershed_frontalablation_monthly = pickle.load(f)
                    # Mass balance: total mass balance
                    with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
                        reg_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_massbaltotal_monthly, 'rb') as f:
                        regO2_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_massbaltotal_monthly, 'rb') as f:
                        watershed_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_massbaltotal_monthly, 'rb') as f:
                        degid_massbaltotal_monthly = pickle.load(f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                        reg_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'rb') as f:
                        regO2_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'rb') as f:
                        watershed_mbclim_annual_binned = pickle.load(f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                        reg_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'rb') as f:
                        regO2_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                        watershed_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'rb') as f:
                        degid_runoff_monthly_moving = pickle.load(f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                        reg_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'rb') as f:
                        regO2_runoff_monthly_fixed= pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                        watershed_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'rb') as f:
                        degid_runoff_monthly_fixed = pickle.load(f)
                    # Runoff: precipitation
                    with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
                        reg_prec_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_prec_monthly, 'rb') as f:
                        regO2_prec_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'rb') as f:
                        watershed_prec_monthly= pickle.load(f)
                    # Runoff: off-glacier precipitation
                    with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
                        reg_offglac_prec_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_prec_monthly, 'rb') as f:
                        regO2_offglac_prec_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'rb') as f:
                        watershed_offglac_prec_monthly = pickle.load(f)
                    # Runoff: off-glacier melt
                    with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
                        reg_offglac_melt_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_melt_monthly, 'rb') as f:
                        regO2_offglac_melt_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'rb') as f:
                        watershed_offglac_melt_monthly = pickle.load(f)
                    # Runoff: off-glacier refreeze
                    with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
                        reg_offglac_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_refreeze_monthly, 'rb') as f:
                        regO2_offglac_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'rb') as f:
                        watershed_offglac_refreeze_monthly = pickle.load(f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                        reg_ela_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'rb') as f:
                        regO2_ela_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'rb') as f:
                        watershed_ela_annual = pickle.load(f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
                        reg_aar_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'rb') as f:
                        regO2_aar_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'rb') as f:
                        watershed_aar_annual = pickle.load(f)
                        
                    # Years
                    if years is None:
                        for nglac, glacno in enumerate(glacno_list[0:1]):
                            # Filenames
                            netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                            netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                            ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                
                            # Years
                            years = ds_stats.year.values
        
                    
                #%%
                if args.option_plot:
                    # ===== REGIONAL PLOTS =====
                    fig_fp_reg = fig_fp + str(reg).zfill(2) + '/'
                    if not os.path.exists(fig_fp_reg):
                        os.makedirs(fig_fp_reg)
                        
                    # ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
                    fig, ax = plt.subplots(3, 4, squeeze=False, sharex=False, sharey=False, 
                                           gridspec_kw = {'wspace':0.7, 'hspace':0.5})
                    label= gcm_name + ' ' + rcp
                    
                    # VOLUME CHANGE
                    ax[0,0].plot(years, reg_vol_annual/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_vol_annual_bwl is None:
                        ax[0,0].plot(years, reg_vol_annual_bwl/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle='--', zorder=4, label='bwl')
                    if not reg_vol_annual_bd is None:
                        ax[0,0].plot(years, reg_vol_annual_bd/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,0].set_ylabel('Volume (km$^{3}$)')
                    ax[0,0].set_xlim(years.min(), years.max())
                    ax[0,0].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,0].set_ylim(0,reg_vol_annual.max()*1.05/1e9)
                    ax[0,0].tick_params(direction='inout', right=True)
                    ax[0,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )        
                    
    
                    # AREA CHANGE
                    ax[0,1].plot(years, reg_area_annual/1e6, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_area_annual_bd is None:
                        ax[0,1].plot(years, reg_area_annual_bd/1e6, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,1].set_ylabel('Area (km$^{2}$)')
                    ax[0,1].set_xlim(years.min(), years.max())
                    ax[0,1].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,1].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,1].set_ylim(0,reg_area_annual.max()*1.05/1e6)
                    ax[0,1].tick_params(direction='inout', right=True)
                    ax[0,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )    
                    
                    
                    # MASS BALANCE
                    reg_mbmwea_annual = ((reg_vol_annual[1:] - reg_vol_annual[:-1]) / reg_area_annual[:-1] * 
                                         pygem_prms.density_ice / pygem_prms.density_water)
                    ax[0,2].plot(years[0:-1], reg_mbmwea_annual, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    ax[0,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                    ax[0,2].set_xlim(years.min(), years[0:-1].max())
                    ax[0,2].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,2].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,2].tick_params(direction='inout', right=True)
                    
                    
                    # RUNOFF CHANGE 
                    reg_runoff_annual_fixed = reg_runoff_monthly_fixed.reshape(-1,12).sum(axis=1)
                    reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
                    ax[0,3].set_ylabel('Runoff (km$^{3}$)')
                    ax[0,3].set_xlim(years.min(), years[0:-1].max())
                    ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
                    ax[0,3].tick_params(direction='inout', right=True)
                    
                    
    
                    
                    # BINNED VOLUME
                    elev_bin_major = 1000
                    elev_bin_minor = 250
                    ymin = np.floor(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][0]] / elev_bin_major) * elev_bin_major
                    ymax = np.ceil(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][-1]] / elev_bin_major) * elev_bin_major
                    ax[1,0].plot(reg_vol_annual_binned[:,0]/1e9, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,0].plot(reg_vol_annual_binned[:,-1]/1e9, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_vol_annual_bd is None:
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,0]/1e9, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,-1]/1e9, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,0].set_ylabel('Elevation (m)')
                    ax[1,0].set_xlabel('Volume (km$^{3}$)')
                    ax[1,0].set_xlim(0, reg_vol_annual_binned.max()/1e9)
                    ax[1,0].set_ylim(ymin, ymax)
                    ax[1,0].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,0].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,0].tick_params(direction='inout', right=True)
                    ax[1,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # BINNED AREA
                    ax[1,1].plot(reg_area_annual_binned[:,0]/1e6, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,1].plot(reg_area_annual_binned[:,-1]/1e6, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_area_annual_binned_bd is None:
                        ax[1,1].plot(reg_area_annual_binned_bd[:,0]/1e6, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,1].plot(reg_area_annual_binned_bd[:,-1]/1e6, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,1].set_ylabel('Elevation (m)')
                    ax[1,1].set_xlabel('Area (km$^{2}$)')
                    ax[1,1].set_xlim(0, reg_area_annual_binned.max()/1e6)
                    ax[1,1].set_ylim(ymin, ymax)
                    ax[1,1].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,1].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,1].tick_params(direction='inout', right=True)
                    ax[1,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # CLIMATIC MASS BALANCE GRADIENT
                    reg_mbclim_annual_binned_mwea = reg_mbclim_annual_binned / reg_area_annual_binned
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,0], elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,-2], elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].set_ylabel('Elevation (m)')
                    ax[1,2].set_xlabel('$b_{clim}$ (m w.e. yr$^{-1}$)')
                    ax[1,2].set_ylim(ymin, ymax)
                    ax[1,2].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,2].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,2].tick_params(direction='inout', right=True)           
                    ax[1,2].axvline(0, color='k', linewidth=0.25)
                    
                    
                    # RUNOFF COMPONENTS
    #                reg_offglac_melt_annual = reg_offglac_melt_monthly.reshape(-1,12).sum(axis=1)
    #                reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
    #                ax[0,3].set_ylabel('Runoff (km$^{3}$)')
    #                ax[0,3].set_xlim(years.min(), years[0:-1].max())
    #                ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
    #                ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
    #                ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
    #                ax[0,3].tick_params(direction='inout', right=True)
                    
                    
                    # ELA
                    ela_min = np.floor(np.min(reg_ela_annual[0:-1]) / 100) * 100
                    ela_max = np.ceil(np.max(reg_ela_annual[0:-1]) / 100) * 100
                    ax[2,0].plot(years[0:-1], reg_ela_annual[0:-1], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,0].set_ylabel('ELA (m)')
                    ax[2,0].set_xlim(years.min(), years[0:-1].max())
    #                ax[2,0].set_ylim(ela_min, ela_max)
                    ax[2,0].tick_params(direction='inout', right=True)
                    
                    
                    # AAR
                    ax[2,1].plot(years, reg_aar_annual, color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,1].set_ylabel('AAR (-)')
                    ax[2,1].set_ylim(0,1)
                    ax[2,1].set_xlim(years.min(), years[0:-1].max())
                    ax[2,1].tick_params(direction='inout', right=True)
                    
                    
                    # MASS BALANCE COMPONENTS
                    # - these are only meant for monthly and/or relative purposes 
                    #   mass balance from volume change should be used for annual changes
                    reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
                    # Refreeze
                    reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
                    # Melt
                    reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
                    # Frontal Ablation
                    reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
                    # Periods
                    if reg_acc_annual.shape[0] == 101:
                        period_yrs = 20
                        periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
                        reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
                        
                        # Convert to mwea
                        reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
                        reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
                        reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
                        reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
                        reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
                        reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
                    else:
                        assert True==False, 'Set up for different time periods'
    
                    # Plot
                    ax[2,2].bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='refreeze', zorder=2)
                    ax[2,2].bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='acc', zorder=3)
                    if not reg_frontalablation_periods_mwea.sum() == 0:
                        ax[2,2].bar(periods, -reg_frontalablation_periods_mwea, color='#83439A', width=period_yrs/2-1, label='frontal ablation', zorder=3)
                    ax[2,2].bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='melt', zorder=2)
                    ax[2,2].bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='total', zorder=1)
                    ax[2,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                    ax[2,2].set_xlim(years.min(), years[0:-1].max())
                    ax[2,2].xaxis.set_major_locator(MultipleLocator(100))
                    ax[2,2].xaxis.set_minor_locator(MultipleLocator(20))
                    ax[2,2].yaxis.set_major_locator(MultipleLocator(1))
                    ax[2,2].yaxis.set_minor_locator(MultipleLocator(0.25))
                    ax[2,2].tick_params(direction='inout', right=True)
                    ax[2,2].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                                   loc=(1.2,0.25)) 
                    
                    
                    # Remove plot in lower right
                    fig.delaxes(ax[2,3])
                    
                    
                    # Title
                    fig.text(0.5, 0.95, rgi_reg_dict[reg] + ' (' + gcm_name + ' ' + rcp + ')', size=12, ha='center', va='top',)
                    
                    # Save figure
                    fig_fn = str(reg) + '_allplots_' + str(years.min()) + '-' + str(years.max()) + '_' + gcm_name + '_' + rcp + '.png'
                    fig.set_size_inches(8,6)
                    fig.savefig(fig_fp_reg + fig_fn, bbox_inches='tight', dpi=300)


if option_process_data_nodebris:

    overwrite_pickle = False
    
    grouping = 'all'

    fig_fp = netcdf_fp_cmip5 + '/../analysis-nodebris/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = netcdf_fp_cmip5 + '/../analysis-nodebris/csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = fig_fp + '../pickle-nodebris/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
        
#    def mwea_to_gta(mwea, area):
#        return mwea * pygem_prms.density_water * area / 1e12
    
    #%%
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')

        # ===== EXPORT RESULTS =====
        success_fullfn = csv_fp + 'CMIP5_success.csv'
        success_cns = ['O1Region', 'count_success', 'count', 'count_%', 'reg_area_km2_success', 'reg_area_km2', 'reg_area_%']
        success_df_single = pd.DataFrame(np.zeros((1,len(success_cns))), columns=success_cns)
        success_df_single.loc[0,:] = [reg, main_glac_rgi.shape[0], main_glac_rgi_all.shape[0],
                                      np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,2),
                                      np.round(main_glac_rgi.Area.sum(),2), np.round(main_glac_rgi_all.Area.sum(),2),
                                      np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,2)]
        if os.path.exists(success_fullfn):
            success_df = pd.read_csv(success_fullfn)
            
            # Add or overwrite existing file
            success_idx = np.where((success_df.O1Region == reg))[0]
            if len(success_idx) > 0:
                success_df.loc[success_idx,:] = success_df_single.values
            else:
                success_df = pd.concat([success_df, success_df_single], axis=0)
                
        else:
            success_df = success_df_single
            
        success_df = success_df.sort_values('O1Region', ascending=True)
        success_df.reset_index(inplace=True, drop=True)
        success_df.to_csv(success_fullfn, index=False)                
        
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size) * degree_size
        main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size) * degree_size
        deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi))]
        main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
        
        # River Basin
        watershed_dict_fn = pygem_prms.main_directory + '/../qgis_datasets/rgi60_watershed_dict.csv'
        watershed_csv = pd.read_csv(watershed_dict_fn)
        watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
        main_glac_rgi['watershed'] = main_glac_rgi.RGIId.map(watershed_dict)
        if len(np.where(main_glac_rgi.watershed.isnull())[0]) > 0:
            main_glac_rgi.loc[np.where(main_glac_rgi.watershed.isnull())[0],'watershed'] = 'nan'
        
        #%%
        # Unique Groups
        # O2 Regions
        unique_regO2s = np.unique(main_glac_rgi['O2Region'])
        
        # Degrees
        if main_glac_rgi['deg_id'].isnull().all():
            unique_degids = None
        else:
            unique_degids = np.unique(main_glac_rgi['deg_id'])
        
        # Watersheds
        if main_glac_rgi['watershed'].isnull().all():
            unique_watersheds = None
        else:
            unique_watersheds = np.unique(main_glac_rgi['watershed'])

        # Elevation bins
        elev_bin_size = 10
        zmax = int(np.ceil(main_glac_rgi.Zmax.max() / elev_bin_size) * elev_bin_size) + 500
        elev_bins = np.arange(0,zmax,elev_bin_size)
        elev_bins = np.insert(elev_bins, 0, -1000)
        
        
        # Pickle datasets
        # Glacier list
        fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
        if not os.path.exists(pickle_fp + str(reg).zfill(2) + '/'):
            os.makedirs(pickle_fp + str(reg).zfill(2) + '/')
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'wb') as f:
            pickle.dump(glacno_list, f)
        
        # O2Region dict
        fn_unique_regO2s = 'R' + str(reg) + '_unique_regO2s.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_regO2s, 'wb') as f:
            pickle.dump(unique_regO2s, f)      
        # Watershed dict
        fn_unique_watersheds = 'R' + str(reg) + '_unique_watersheds.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_watersheds, 'wb') as f:
            pickle.dump(unique_watersheds, f) 
        # Degree ID dict
        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_degids, 'wb') as f:
            pickle.dump(unique_degids, f)
        
        fn_elev_bins = 'R' + str(reg) + '_elev_bins.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_elev_bins, 'wb') as f:
            pickle.dump(elev_bins, f)
        
        #%%
        years = None        
        for gcm_name in gcm_names:
            for rcp in rcps:

                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_reg):
                    os.makedirs(pickle_fp_reg)
                pickle_fp_regO2 =  pickle_fp + str(reg).zfill(2) + '/O2Regions/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_regO2):
                    os.makedirs(pickle_fp_regO2)
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_watershed):
                    os.makedirs(pickle_fp_watershed)
                pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_degid):
                    os.makedirs(pickle_fp_degid)
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                regO2_rcp_gcm_str = 'R' + str(reg) + '_O2Regions_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name
                
                # Volume
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                fn_regO2_vol_annual = regO2_rcp_gcm_str + '_vol_annual.pkl'
                fn_watershed_vol_annual = watershed_rcp_gcm_str + '_vol_annual.pkl'
                fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                # Volume below sea level 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_regO2_vol_annual_bwl = regO2_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_watershed_vol_annual_bwl = watershed_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_degid_vol_annual_bwl = degid_rcp_gcm_str + '_vol_annual_bwl.pkl'
                # Volume below debris
                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_regO2_vol_annual_bd = regO2_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_watershed_vol_annual_bd = watershed_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_degid_vol_annual_bd = degid_rcp_gcm_str + '_vol_annual_bd.pkl'
                # Area 
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_regO2_area_annual = regO2_rcp_gcm_str + '_area_annual.pkl'
                fn_watershed_area_annual = watershed_rcp_gcm_str + '_area_annual.pkl'
                fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                # Area below debris
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_regO2_area_annual_bd = regO2_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_watershed_area_annual_bd = watershed_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_degid_area_annual_bd = degid_rcp_gcm_str + '_area_annual_bd.pkl'
                # Binned Volume
                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_regO2_vol_annual_binned = regO2_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_watershed_vol_annual_binned = watershed_rcp_gcm_str + '_vol_annual_binned.pkl'
                # Binned Volume below debris
                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_regO2_vol_annual_binned_bd = regO2_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_watershed_vol_annual_binned_bd = watershed_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                # Binned Area
                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_regO2_area_annual_binned = regO2_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_watershed_area_annual_binned = watershed_rcp_gcm_str + '_area_annual_binned.pkl'
                # Binned Area below debris
                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_regO2_area_annual_binned_bd = regO2_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_watershed_area_annual_binned_bd = watershed_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                # Binned Climatic Mass Balance
                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_regO2_mbclim_annual_binned = regO2_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                # Runoff: moving-gauged
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_regO2_runoff_monthly_moving = regO2_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_degid_runoff_monthly_moving = degid_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                # Runoff: fixed-gauged
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_regO2_runoff_monthly_fixed = regO2_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_degid_runoff_monthly_fixed = degid_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                # ELA
                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                fn_regO2_ela_annual = regO2_rcp_gcm_str + '_ela_annual.pkl'
                fn_watershed_ela_annual = watershed_rcp_gcm_str + '_ela_annual.pkl'
                # AAR
                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                fn_regO2_aar_annual = regO2_rcp_gcm_str + '_aar_annual.pkl'
                fn_watershed_aar_annual = watershed_rcp_gcm_str + '_aar_annual.pkl'
                    
                if not os.path.exists(pickle_fp_reg + fn_reg_vol_annual) or overwrite_pickle:

                    # Entire region
                    years = None
                    reg_vol_annual = None
                    reg_vol_annual_bwl = None
                    reg_vol_annual_bd = None
                    reg_area_annual = None
                    reg_area_annual_bd = None
                    reg_vol_annual_binned = None
                    reg_vol_annual_binned_bd = None
                    reg_area_annual_binned = None
                    reg_area_annual_binned_bd = None
                    reg_mbclim_annual_binned = None
                    reg_runoff_monthly_fixed = None
                    reg_runoff_monthly_moving = None
                    reg_ela_annual = None
                    reg_ela_annual_area = None # used for weighted area calculations
                    reg_area_annual_acc = None
                    reg_area_annual_frombins = None
                    
                    # Subregion groups
                    regO2_vol_annual = None
                    regO2_vol_annual_bwl = None
                    regO2_vol_annual_bd = None
                    regO2_area_annual = None
                    regO2_area_annual_bd = None
                    regO2_vol_annual_binned = None
                    regO2_vol_annual_binned_bd = None
                    regO2_area_annual_binned = None
                    regO2_area_annual_binned_bd = None
                    regO2_mbclim_annual_binned = None
                    regO2_runoff_monthly_fixed = None
                    regO2_runoff_monthly_moving = None
                    regO2_ela_annual = None
                    regO2_ela_annual_area = None # used for weighted area calculations
                    regO2_area_annual_acc = None
                    regO2_area_annual_frombins = None
                    
                    # Watershed groups
                    watershed_vol_annual = None
                    watershed_vol_annual_bwl = None
                    watershed_vol_annual_bd = None
                    watershed_area_annual = None
                    watershed_area_annual_bd = None
                    watershed_vol_annual_binned = None
                    watershed_vol_annual_binned_bd = None
                    watershed_area_annual_binned = None
                    watershed_area_annual_binned_bd = None
                    watershed_mbclim_annual_binned = None
                    watershed_runoff_monthly_fixed = None
                    watershed_runoff_monthly_moving = None
                    watershed_ela_annual = None
                    watershed_ela_annual_area = None # used for weighted area calculations
                    watershed_area_annual_acc = None
                    watershed_area_annual_frombins = None
    
                    # Degree groups                
                    degid_vol_annual = None
                    degid_vol_annual_bwl = None
                    degid_vol_annual_bd = None
                    degid_area_annual = None
                    degid_area_annual_bd = None
                    degid_runoff_monthly_fixed = None
                    degid_runoff_monthly_moving = None
    
    
                    for nglac, glacno in enumerate(glacno_list):
                        if nglac%10 == 0:
                            print(gcm_name, rcp, glacno)
                        
                        # Group indices
                        glac_idx = np.where(main_glac_rgi['glacno'] == glacno)[0][0]
                        regO2 = main_glac_rgi.loc[glac_idx, 'O2Region']
                        regO2_idx = np.where(regO2 == unique_regO2s)[0][0]
                        watershed = main_glac_rgi.loc[glac_idx,'watershed']
                        watershed_idx = np.where(watershed == unique_watersheds)
                        degid = main_glac_rgi.loc[glac_idx, 'deg_id']
                        degid_idx = np.where(degid == unique_degids)[0][0]
                        
                        # Filenames
                        nsim_strs = ['50', '1', '100', '150', '200', '250']
                        ds_binned = None
                        nset = -1
                        while ds_binned is None and nset <= len(nsim_strs):
                            nset += 1
                            nsim_str = nsim_strs[nset]
                            
                            try:
                                netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                netcdf_fn_binned = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
        
                                netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                
                                # Open files
                                ds_binned = xr.open_dataset(netcdf_fp_binned + '/' + netcdf_fn_binned)
                                ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                            except:
                                ds_binned = None
                            
                        # Years
                        if years is None:
                            years = ds_stats.year.values
                            
                                
                        # ----- 1. Volume (m3) vs. Year ----- 
                        glac_vol_annual = ds_stats.glac_volume_annual.values[0,:]
                        # All
                        if reg_vol_annual is None:
                            reg_vol_annual = glac_vol_annual
                        else:
                            reg_vol_annual += glac_vol_annual
                        # O2Region
                        if regO2_vol_annual is None:
                            regO2_vol_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_vol_annual[regO2_idx,:] = glac_vol_annual
                        else:
                            regO2_vol_annual[regO2_idx,:] += glac_vol_annual
                        # Watershed
                        if watershed_vol_annual is None:
                            watershed_vol_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_vol_annual[watershed_idx,:] = glac_vol_annual
                        else:
                            watershed_vol_annual[watershed_idx,:] += glac_vol_annual
                        # DegId
                        if degid_vol_annual is None:
                            degid_vol_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_vol_annual[degid_idx,:] = glac_vol_annual
                        else:
                            degid_vol_annual[degid_idx,:] += glac_vol_annual
                            
                            
                        # ----- 2. Volume below-sea-level (m3) vs. Year ----- 
                        #  - initial elevation is stored
                        #  - bed elevation is constant in time
                        #  - assume sea level is at 0 m a.s.l.
                        z_sealevel = 0
                        bin_z_init = ds_binned.bin_surface_h_initial.values[0,:]
                        bin_thick_annual = ds_binned.bin_thick_annual.values[0,:,:]
                        bin_z_bed = bin_z_init - bin_thick_annual[:,0]
                        # Annual surface height
                        bin_z_surf_annual = bin_z_bed[:,np.newaxis] + bin_thick_annual
                        
                        # Annual volume (m3)
                        bin_vol_annual = ds_binned.bin_volume_annual.values[0,:,:]
                        # Annual area (m2)
                        bin_area_annual = np.zeros(bin_vol_annual.shape)
                        bin_area_annual[bin_vol_annual > 0] = (
                                bin_vol_annual[bin_vol_annual > 0] / bin_thick_annual[bin_vol_annual > 0])
                        
                        # Processed based on OGGM's _vol_below_level function
                        bwl = (bin_z_bed[:,np.newaxis] < 0) & (bin_thick_annual > 0)
                        if bwl.any():
                            # Annual surface height (max of sea level for calcs)
                            bin_z_surf_annual_bwl = bin_z_surf_annual.copy()
                            bin_z_surf_annual_bwl[bin_z_surf_annual_bwl > z_sealevel] = z_sealevel
                            # Annual thickness below sea level (m)
                            bin_thick_annual_bwl = bin_thick_annual.copy()
                            bin_thick_annual_bwl = bin_z_surf_annual_bwl - bin_z_bed[:,np.newaxis]
                            bin_thick_annual_bwl[~bwl] = 0
                            # Annual volume below sea level (m3)
                            bin_vol_annual_bwl = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bwl[bwl] = bin_thick_annual_bwl[bwl] * bin_area_annual[bwl]
                            glac_vol_annual_bwl = bin_vol_annual_bwl.sum(0)
                            
                            # All
                            if reg_vol_annual_bwl is None:
                                reg_vol_annual_bwl = glac_vol_annual_bwl
                            else:
                                reg_vol_annual_bwl += glac_vol_annual_bwl
                            # O2Region
                            if regO2_vol_annual_bwl is None:
                                regO2_vol_annual_bwl = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bwl[regO2_idx,:] = glac_vol_annual_bwl
                            else:
                                regO2_vol_annual_bwl[regO2_idx,:] += glac_vol_annual_bwl
                            # Watershed
                            if watershed_vol_annual_bwl is None:
                                watershed_vol_annual_bwl = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bwl[watershed_idx,:] = glac_vol_annual_bwl
                            else:
                                watershed_vol_annual_bwl[watershed_idx,:] += glac_vol_annual_bwl
                            # DegId
                            if degid_vol_annual_bwl is None:
                                degid_vol_annual_bwl = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bwl[degid_idx,:] = glac_vol_annual_bwl
                            else:
                                degid_vol_annual_bwl[degid_idx,:] += glac_vol_annual_bwl
                        
    
                        # ----- 3. Volume below-debris vs. Time ----- 
                        gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
                        fls = gdir.read_pickle('inversion_flowlines')
                        bin_debris_hd = np.zeros(bin_z_init.shape)
                        bin_debris_ed = np.zeros(bin_z_init.shape) + 1
                        if 'debris_hd' in dir(fls[0]):
                            bin_debris_hd[0:fls[0].debris_hd.shape[0]] = fls[0].debris_hd
                            bin_debris_ed[0:fls[0].debris_hd.shape[0]] = fls[0].debris_ed
                        if bin_debris_hd.sum() > 0:
                            bin_vol_annual_bd = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bd[bin_debris_hd > 0, :] = bin_vol_annual[bin_debris_hd > 0, :]
                            glac_vol_annual_bd = bin_vol_annual_bd.sum(0)
                            
                            # All
                            if reg_vol_annual_bd is None:
                                reg_vol_annual_bd = glac_vol_annual_bd
                            else:
                                reg_vol_annual_bd += glac_vol_annual_bd
                            # O2Region
                            if regO2_vol_annual_bd is None:
                                regO2_vol_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bd[regO2_idx,:] = glac_vol_annual_bd
                            else:
                                regO2_vol_annual_bd[regO2_idx,:] += glac_vol_annual_bd
                            # Watershed
                            if watershed_vol_annual_bd is None:
                                watershed_vol_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bd[watershed_idx,:] = glac_vol_annual_bd
                            else:
                                watershed_vol_annual_bd[watershed_idx,:] += glac_vol_annual_bd
                            # DegId
                            if degid_vol_annual_bd is None:
                                degid_vol_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bd[degid_idx,:] = glac_vol_annual_bd
                            else:
                                degid_vol_annual_bd[degid_idx,:] += glac_vol_annual_bd
                        
                        
                        # ----- 4. Area vs. Time ----- 
                        glac_area_annual = ds_stats.glac_area_annual.values[0,:]
                        # All
                        if reg_area_annual is None:
                            reg_area_annual = glac_area_annual
                        else:
                            reg_area_annual += glac_area_annual
                        # O2Region
                        if regO2_area_annual is None:
                            regO2_area_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual[regO2_idx,:] = glac_area_annual
                        else:
                            regO2_area_annual[regO2_idx,:] += glac_area_annual
                        # Watershed
                        if watershed_area_annual is None:
                            watershed_area_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual[watershed_idx,:] = glac_area_annual
                        else:
                            watershed_area_annual[watershed_idx,:] += glac_area_annual
                        # DegId
                        if degid_area_annual is None:
                            degid_area_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_area_annual[degid_idx,:] = glac_area_annual
                        else:
                            degid_area_annual[degid_idx,:] += glac_area_annual
                        
                        
                        # ----- 5. Area below-debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            bin_area_annual_bd = np.zeros(bin_area_annual.shape)
                            bin_area_annual_bd[bin_debris_hd > 0, :] = bin_area_annual[bin_debris_hd > 0, :]
                            glac_area_annual_bd = bin_area_annual_bd.sum(0)
                            
                            # All
                            if reg_area_annual_bd is None:
                                reg_area_annual_bd = glac_area_annual_bd
                            else:
                                reg_area_annual_bd += glac_area_annual_bd
                            # O2Region
                            if regO2_area_annual_bd is None:
                                regO2_area_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_area_annual_bd[regO2_idx,:] = glac_area_annual_bd
                            else:
                                regO2_area_annual_bd[regO2_idx,:] += glac_area_annual_bd
                            # Watershed
                            if watershed_area_annual_bd is None:
                                watershed_area_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_area_annual_bd[watershed_idx,:] = glac_area_annual_bd
                            else:
                                watershed_area_annual_bd[watershed_idx,:] += glac_area_annual_bd
                            # DegId
                            if degid_area_annual_bd is None:
                                degid_area_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_area_annual_bd[degid_idx,:] = glac_area_annual_bd
                            else:
                                degid_area_annual_bd[degid_idx,:] += glac_area_annual_bd
                        
                        
                        # ----- 6. Binned glacier volume vs. Time ----- 
                        bin_vol_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_vol_annual[:,ncol])
                            bin_vol_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_vol_annual_binned is None:
                            reg_vol_annual_binned = bin_vol_annual_10m
                        else:
                            reg_vol_annual_binned += bin_vol_annual_10m
                        # O2Region
                        if regO2_vol_annual_binned is None:
                            regO2_vol_annual_binned = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                            regO2_vol_annual_binned[regO2_idx,:,:] = bin_vol_annual_10m
                        else:
                            regO2_vol_annual_binned[regO2_idx,:,:] += bin_vol_annual_10m
                        # Watershed
                        if watershed_vol_annual_binned is None:
                            watershed_vol_annual_binned = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                            watershed_vol_annual_binned[watershed_idx,:,:] = bin_vol_annual_10m
                        else:
                            watershed_vol_annual_binned[watershed_idx,:,:] += bin_vol_annual_10m
                        
    
                        # ----- 7. Binned glacier volume below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_vol_annual_10m_bd = bin_vol_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_vol_annual_binned_bd is None:
                                reg_vol_annual_binned_bd = bin_vol_annual_10m_bd
                            else:
                                reg_vol_annual_binned_bd += bin_vol_annual_10m_bd
                            # O2Region
                            if regO2_vol_annual_binned_bd is None:
                                regO2_vol_annual_binned_bd = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] += bin_vol_annual_10m_bd
                            # Watershed
                            if watershed_vol_annual_binned_bd is None:
                                watershed_vol_annual_binned_bd = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] += bin_vol_annual_10m_bd
    
    
                        # ----- 8. Binned glacier area vs. Time ----- 
                        bin_area_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_area_annual[:,ncol])
                            bin_area_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_area_annual_binned is None:
                            reg_area_annual_binned = bin_area_annual_10m
                        else:
                            reg_area_annual_binned += bin_area_annual_10m
                        # O2Region
                        if regO2_area_annual_binned is None:
                            regO2_area_annual_binned = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                            regO2_area_annual_binned[regO2_idx,:,:] = bin_area_annual_10m
                        else:
                            regO2_area_annual_binned[regO2_idx,:,:] += bin_area_annual_10m
                        # Watershed
                        if watershed_area_annual_binned is None:
                            watershed_area_annual_binned = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                            watershed_area_annual_binned[watershed_idx,:,:] = bin_area_annual_10m
                        else:
                            watershed_area_annual_binned[watershed_idx,:,:] += bin_area_annual_10m
    
    
                        
                        # ----- 9. Binned glacier area below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_area_annual_10m_bd = bin_area_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_area_annual_binned_bd is None:
                                reg_area_annual_binned_bd = bin_area_annual_10m_bd
                            else:
                                reg_area_annual_binned_bd += bin_area_annual_10m_bd
                            # O2Region
                            if regO2_area_annual_binned_bd is None:
                                regO2_area_annual_binned_bd = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                                regO2_area_annual_binned_bd[regO2_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                regO2_area_annual_binned_bd[regO2_idx,:,:] += bin_area_annual_10m_bd
                            # Watershed
                            if watershed_area_annual_binned_bd is None:
                                watershed_area_annual_binned_bd = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                                watershed_area_annual_binned_bd[watershed_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                watershed_area_annual_binned_bd[watershed_idx,:,:] += bin_area_annual_10m_bd


                        # ----- 11. Binned Climatic Mass Balance vs. Time -----
                        # - Various mass balance datasets may have slight mismatch due to averaging
                        #   ex. mbclim_annual was reported in mwe, so the area average will cause difference
                        #   ex. mbtotal_monthly was averaged on a monthly basis, so the temporal average will cause difference
                        bin_mbclim_annual = ds_binned.bin_massbalclim_annual.values[0,:,:]
                        bin_mbclim_annual_m3we = bin_mbclim_annual * bin_area_annual
    
    #                    glac_massbaltotal_annual_0 = bin_mbclim_annual_m3we.sum(0)
    #                    glac_massbaltotal_annual_1 = glac_massbaltotal_monthly.reshape(-1,12).sum(1)
    #                    glac_massbaltotal_annual_2 = ((glac_vol_annual[1:] - glac_vol_annual[0:-1]) * 
    #                                                  pygem_prms.density_ice / pygem_prms.density_water)
                        
                        bin_mbclim_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_mbclim_annual_m3we[:,ncol])
                            bin_mbclim_annual_10m[:,ncol] = bin_counts
                        # All
                        if reg_mbclim_annual_binned is None:
                            reg_mbclim_annual_binned = bin_mbclim_annual_10m
                        else:
                            reg_mbclim_annual_binned += bin_mbclim_annual_10m
                        # O2Region
                        if regO2_mbclim_annual_binned is None:
                            regO2_mbclim_annual_binned = np.zeros((len(unique_regO2s), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            regO2_mbclim_annual_binned[regO2_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            regO2_mbclim_annual_binned[regO2_idx,:,:] += bin_mbclim_annual_10m
                        # Watershed
                        if watershed_mbclim_annual_binned is None:
                            watershed_mbclim_annual_binned = np.zeros((len(unique_watersheds), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            watershed_mbclim_annual_binned[watershed_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            watershed_mbclim_annual_binned[watershed_idx,:,:] += bin_mbclim_annual_10m
    
                        
                        # ----- 12. Runoff vs. Time -----
                        glac_runoff_monthly = ds_stats.glac_runoff_monthly.values[0,:]
                        # Moving-gauge Runoff vs. Time
                        # All
                        if reg_runoff_monthly_moving is None:
                            reg_runoff_monthly_moving = glac_runoff_monthly
                        else:
                            reg_runoff_monthly_moving += glac_runoff_monthly
                        # O2Region
                        if regO2_runoff_monthly_moving is None:
                            regO2_runoff_monthly_moving = np.zeros((len(unique_regO2s),len(ds_stats.time.values)))
                            regO2_runoff_monthly_moving[regO2_idx,:] = glac_runoff_monthly
                        else:
                            regO2_runoff_monthly_moving[regO2_idx,:] += glac_runoff_monthly
                        # watershed
                        if watershed_runoff_monthly_moving is None:
                            watershed_runoff_monthly_moving = np.zeros((len(unique_watersheds),len(ds_stats.time.values)))
                            watershed_runoff_monthly_moving[watershed_idx,:] = glac_runoff_monthly
                        else:
                            watershed_runoff_monthly_moving[watershed_idx,:] += glac_runoff_monthly
                        # DegId
                        if degid_runoff_monthly_moving is None:
                            degid_runoff_monthly_moving = np.zeros((len(unique_degids),len(ds_stats.time.values)))
                            degid_runoff_monthly_moving[degid_idx,:] = glac_runoff_monthly
                        else:
                            degid_runoff_monthly_moving[degid_idx,:] += glac_runoff_monthly
                            
                        # Fixed-gauge Runoff vs. Time
                        offglac_runoff_monthly = ds_stats.offglac_runoff_monthly.values[0,:]
                        glac_runoff_monthly_fixed = glac_runoff_monthly + offglac_runoff_monthly
                        # All
                        if reg_runoff_monthly_fixed is None:
                            reg_runoff_monthly_fixed = glac_runoff_monthly_fixed
                        else:
                            reg_runoff_monthly_fixed += glac_runoff_monthly_fixed
                        # O2Region
                        if regO2_runoff_monthly_fixed is None:
                            regO2_runoff_monthly_fixed = np.zeros((len(unique_regO2s),len(ds_stats.time.values)))
                            regO2_runoff_monthly_fixed[regO2_idx,:] = glac_runoff_monthly_fixed
                        else:
                            regO2_runoff_monthly_fixed[regO2_idx,:] += glac_runoff_monthly_fixed
                        # Watershed
                        if watershed_runoff_monthly_fixed is None:
                            watershed_runoff_monthly_fixed = np.zeros((len(unique_watersheds),len(ds_stats.time.values)))
                            watershed_runoff_monthly_fixed[watershed_idx,:] = glac_runoff_monthly_fixed
                        else:
                            watershed_runoff_monthly_fixed[watershed_idx,:] += glac_runoff_monthly_fixed
                        # DegId
                        if degid_runoff_monthly_fixed is None:
                            degid_runoff_monthly_fixed = np.zeros((len(unique_degids),len(ds_stats.time.values)))
                            degid_runoff_monthly_fixed[degid_idx,:] = glac_runoff_monthly_fixed
                        else:
                            degid_runoff_monthly_fixed[degid_idx,:] += glac_runoff_monthly_fixed

    
                        # ----- 13. ELA vs. Time -----
                        glac_ela_annual = ds_stats.glac_ELA_annual.values[0,:]
                        if np.isnan(glac_ela_annual).any():
                            # Quality control nan values 
                            #  - replace with max elev because occur when entire glacier has neg mb
                            bin_z_surf_annual_glaconly = bin_z_surf_annual.copy()
                            bin_z_surf_annual_glaconly[bin_thick_annual == 0] = np.nan
                            zmax_annual = np.nanmax(bin_z_surf_annual_glaconly, axis=0)
                            glac_ela_annual[np.isnan(glac_ela_annual)] = zmax_annual[np.isnan(glac_ela_annual)]
    
                        # Area-weighted ELA
                        # All
                        if reg_ela_annual is None:
                            reg_ela_annual = glac_ela_annual
                            reg_ela_annual_area = glac_area_annual.copy()
                        else:
                            # Use index to avoid dividing by 0 when glacier completely melts                            
                            ela_idx = np.where(reg_ela_annual_area + glac_area_annual > 0)[0]
                            reg_ela_annual[ela_idx] = (
                                    (reg_ela_annual[ela_idx] * reg_ela_annual_area[ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                    (reg_ela_annual_area[ela_idx] + glac_area_annual[ela_idx]))
                            reg_ela_annual_area += glac_area_annual
                        
                        # O2Region
                        if regO2_ela_annual is None:
                            regO2_ela_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual[regO2_idx,:] = glac_ela_annual
                            regO2_ela_annual_area = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual_area[regO2_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(regO2_ela_annual_area[regO2_idx,:] + glac_area_annual > 0)[0]
                            regO2_ela_annual[regO2_idx,ela_idx] = (
                                    (regO2_ela_annual[regO2_idx,ela_idx] * regO2_ela_annual_area[regO2_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (regO2_ela_annual_area[regO2_idx,ela_idx] + glac_area_annual[ela_idx]))
                            regO2_ela_annual_area[regO2_idx,:] += glac_area_annual
                        
                        # Watershed
                        if watershed_ela_annual is None:
                            watershed_ela_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual[watershed_idx,:] = glac_ela_annual
                            watershed_ela_annual_area = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual_area[watershed_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(watershed_ela_annual_area[watershed_idx,:] + glac_area_annual > 0)[0]
                            watershed_ela_annual[watershed_idx,ela_idx] = (
                                    (watershed_ela_annual[watershed_idx,ela_idx] * watershed_ela_annual_area[watershed_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (watershed_ela_annual_area[watershed_idx,ela_idx] + glac_area_annual[ela_idx]))
                            watershed_ela_annual_area[watershed_idx,:] += glac_area_annual
                        

                        # ----- 14. AAR vs. Time -----
                        #  - averaging issue with bin_area_annual.sum(0) != glac_area_annual
                        #  - hence only use these 
                        bin_area_annual_acc = bin_area_annual.copy()
                        bin_area_annual_acc[bin_mbclim_annual <= 0] = 0
                        glac_area_annual_acc = bin_area_annual_acc.sum(0)
                        glac_area_annual_frombins = bin_area_annual.sum(0)
                        
                        # All
                        if reg_area_annual_acc is None:
                            reg_area_annual_acc = glac_area_annual_acc.copy()
                            reg_area_annual_frombins = glac_area_annual_frombins.copy()
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        else:
                            reg_area_annual_acc += glac_area_annual_acc
                            reg_area_annual_frombins += glac_area_annual_frombins
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        # O2Regions
                        if regO2_area_annual_acc is None:
                            regO2_area_annual_acc = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_acc[regO2_idx,:] = glac_area_annual_acc.copy()
                            regO2_area_annual_frombins = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_frombins[regO2_idx,:] = glac_area_annual_frombins.copy()
                            regO2_aar_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        else:
                            regO2_area_annual_acc[regO2_idx,:] += glac_area_annual_acc
                            regO2_area_annual_frombins[regO2_idx,:] += glac_area_annual_frombins
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        # Watersheds
                        if watershed_area_annual_acc is None:
                            watershed_area_annual_acc = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_acc[watershed_idx,:] = glac_area_annual_acc.copy()
                            watershed_area_annual_frombins = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_frombins[watershed_idx,:] = glac_area_annual_frombins.copy()
                            watershed_aar_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        else:
                            watershed_area_annual_acc[watershed_idx,:] += glac_area_annual_acc
                            watershed_area_annual_frombins[watershed_idx,:] += glac_area_annual_frombins
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        
                    # ===== PICKLE DATASETS =====
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'wb') as f:
                        pickle.dump(reg_vol_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'wb') as f:
                        pickle.dump(regO2_vol_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'wb') as f:
                        pickle.dump(watershed_vol_annual, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'wb') as f:
                        pickle.dump(degid_vol_annual, f)
                    # Volume below sea level 
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'wb') as f:
                        pickle.dump(reg_vol_annual_bwl, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bwl, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bwl, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'wb') as f:
                        pickle.dump(degid_vol_annual_bwl, f) 
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'wb') as f:
                        pickle.dump(degid_vol_annual_bd, f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'wb') as f:
                        pickle.dump(reg_area_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'wb') as f:
                        pickle.dump(regO2_area_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'wb') as f:
                        pickle.dump(watershed_area_annual, f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'wb') as f:
                        pickle.dump(degid_area_annual, f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'wb') as f:
                        pickle.dump(degid_area_annual_bd, f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned, f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned_bd, f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'wb') as f:
                        pickle.dump(reg_area_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned, f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned_bd, f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(reg_mbclim_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(regO2_mbclim_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(watershed_mbclim_annual_binned, f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_moving, f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_moving, f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_moving, f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_moving, f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_fixed, f)     
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_fixed, f)  
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_fixed, f)  
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_fixed, f)  
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'wb') as f:
                        pickle.dump(reg_ela_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'wb') as f:
                        pickle.dump(regO2_ela_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'wb') as f:
                        pickle.dump(watershed_ela_annual, f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'wb') as f:
                        pickle.dump(reg_aar_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'wb') as f:
                        pickle.dump(regO2_aar_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'wb') as f:
                        pickle.dump(watershed_aar_annual, f)
                        
                # ----- OTHERWISE LOAD THE PROCESSED DATASETS -----
                else:
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                        reg_vol_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'rb') as f:
                        regO2_vol_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'rb') as f:
                        watershed_vol_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                        degid_vol_annual = pickle.load(f)
                    # Volume below sea level
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'rb') as f:
                        regO2_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'rb') as f:
                        watershed_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'rb') as f:
                        degid_vol_annual_bwl = pickle.load(f)
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                        reg_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'rb') as f:
                        regO2_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'rb') as f:
                        watershed_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'rb') as f:
                        degid_vol_annual_bd = pickle.load(f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                        reg_area_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'rb') as f:
                        regO2_area_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'rb') as f:
                        watershed_area_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                        degid_area_annual = pickle.load(f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                        reg_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'rb') as f:
                        regO2_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'rb') as f:
                        watershed_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'rb') as f:
                        degid_area_annual_bd = pickle.load(f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                        reg_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'rb') as f:
                        regO2_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'rb') as f:
                        watershed_vol_annual_binned = pickle.load(f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                        reg_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'rb') as f:
                        regO2_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'rb') as f:
                        watershed_vol_annual_binned_bd = pickle.load(f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                        reg_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'rb') as f:
                        regO2_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'rb') as f:
                        watershed_area_annual_binned = pickle.load(f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                        reg_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'rb') as f:
                        regO2_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'rb') as f:
                        watershed_area_annual_binned_bd = pickle.load(f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                        reg_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'rb') as f:
                        regO2_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'rb') as f:
                        watershed_mbclim_annual_binned = pickle.load(f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                        reg_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'rb') as f:
                        regO2_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                        watershed_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'rb') as f:
                        degid_runoff_monthly_moving = pickle.load(f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                        reg_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'rb') as f:
                        regO2_runoff_monthly_fixed= pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                        watershed_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'rb') as f:
                        degid_runoff_monthly_fixed = pickle.load(f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                        reg_ela_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'rb') as f:
                        regO2_ela_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'rb') as f:
                        watershed_ela_annual = pickle.load(f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
                        reg_aar_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'rb') as f:
                        regO2_aar_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'rb') as f:
                        watershed_aar_annual = pickle.load(f)
                        
                    # Years
                    if years is None:
                        for nglac, glacno in enumerate(glacno_list[0:1]):
                            # Filenames
                            netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                            netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                            ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                
                            # Years
                            years = ds_stats.year.values
        
                    
                #%%
                if args.option_plot:
                    # ===== REGIONAL PLOTS =====
                    fig_fp_reg = fig_fp + str(reg).zfill(2) + '/'
                    if not os.path.exists(fig_fp_reg):
                        os.makedirs(fig_fp_reg)
                        
                    # ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
                    fig, ax = plt.subplots(3, 4, squeeze=False, sharex=False, sharey=False, 
                                           gridspec_kw = {'wspace':0.7, 'hspace':0.5})
                    label= gcm_name + ' ' + rcp
                    
                    # VOLUME CHANGE
                    ax[0,0].plot(years, reg_vol_annual/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_vol_annual_bwl is None:
                        ax[0,0].plot(years, reg_vol_annual_bwl/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle='--', zorder=4, label='bwl')
                    if not reg_vol_annual_bd is None:
                        ax[0,0].plot(years, reg_vol_annual_bd/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,0].set_ylabel('Volume (km$^{3}$)')
                    ax[0,0].set_xlim(years.min(), years.max())
                    ax[0,0].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,0].set_ylim(0,reg_vol_annual.max()*1.05/1e9)
                    ax[0,0].tick_params(direction='inout', right=True)
                    ax[0,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )        
                    
    
                    # AREA CHANGE
                    ax[0,1].plot(years, reg_area_annual/1e6, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_area_annual_bd is None:
                        ax[0,1].plot(years, reg_area_annual_bd/1e6, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,1].set_ylabel('Area (km$^{2}$)')
                    ax[0,1].set_xlim(years.min(), years.max())
                    ax[0,1].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,1].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,1].set_ylim(0,reg_area_annual.max()*1.05/1e6)
                    ax[0,1].tick_params(direction='inout', right=True)
                    ax[0,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )    
                    
                    
                    # MASS BALANCE
                    reg_mbmwea_annual = ((reg_vol_annual[1:] - reg_vol_annual[:-1]) / reg_area_annual[:-1] * 
                                         pygem_prms.density_ice / pygem_prms.density_water)
                    ax[0,2].plot(years[0:-1], reg_mbmwea_annual, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    ax[0,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                    ax[0,2].set_xlim(years.min(), years[0:-1].max())
                    ax[0,2].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,2].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,2].tick_params(direction='inout', right=True)
                    
                    
                    # RUNOFF CHANGE 
                    reg_runoff_annual_fixed = reg_runoff_monthly_fixed.reshape(-1,12).sum(axis=1)
                    reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
                    ax[0,3].set_ylabel('Runoff (km$^{3}$)')
                    ax[0,3].set_xlim(years.min(), years[0:-1].max())
                    ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
                    ax[0,3].tick_params(direction='inout', right=True)
                    
                    
                    # BINNED VOLUME
                    elev_bin_major = 1000
                    elev_bin_minor = 250
                    ymin = np.floor(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][0]] / elev_bin_major) * elev_bin_major
                    ymax = np.ceil(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][-1]] / elev_bin_major) * elev_bin_major
                    ax[1,0].plot(reg_vol_annual_binned[:,0]/1e9, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,0].plot(reg_vol_annual_binned[:,-1]/1e9, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_vol_annual_bd is None:
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,0]/1e9, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,-1]/1e9, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,0].set_ylabel('Elevation (m)')
                    ax[1,0].set_xlabel('Volume (km$^{3}$)')
                    ax[1,0].set_xlim(0, reg_vol_annual_binned.max()/1e9)
                    ax[1,0].set_ylim(ymin, ymax)
                    ax[1,0].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,0].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,0].tick_params(direction='inout', right=True)
                    ax[1,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # BINNED AREA
                    ax[1,1].plot(reg_area_annual_binned[:,0]/1e6, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,1].plot(reg_area_annual_binned[:,-1]/1e6, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_area_annual_binned_bd is None:
                        ax[1,1].plot(reg_area_annual_binned_bd[:,0]/1e6, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,1].plot(reg_area_annual_binned_bd[:,-1]/1e6, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,1].set_ylabel('Elevation (m)')
                    ax[1,1].set_xlabel('Area (km$^{2}$)')
                    ax[1,1].set_xlim(0, reg_area_annual_binned.max()/1e6)
                    ax[1,1].set_ylim(ymin, ymax)
                    ax[1,1].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,1].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,1].tick_params(direction='inout', right=True)
                    ax[1,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # CLIMATIC MASS BALANCE GRADIENT
                    reg_mbclim_annual_binned_mwea = reg_mbclim_annual_binned / reg_area_annual_binned
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,0], elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,-2], elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].set_ylabel('Elevation (m)')
                    ax[1,2].set_xlabel('$b_{clim}$ (m w.e. yr$^{-1}$)')
                    ax[1,2].set_ylim(ymin, ymax)
                    ax[1,2].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,2].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,2].tick_params(direction='inout', right=True)           
                    ax[1,2].axvline(0, color='k', linewidth=0.25)
                    
                    
                    # RUNOFF COMPONENTS
    #                reg_offglac_melt_annual = reg_offglac_melt_monthly.reshape(-1,12).sum(axis=1)
    #                reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
    #                ax[0,3].set_ylabel('Runoff (km$^{3}$)')
    #                ax[0,3].set_xlim(years.min(), years[0:-1].max())
    #                ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
    #                ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
    #                ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
    #                ax[0,3].tick_params(direction='inout', right=True)
                    
                    
                    # ELA
                    ela_min = np.floor(np.min(reg_ela_annual[0:-1]) / 100) * 100
                    ela_max = np.ceil(np.max(reg_ela_annual[0:-1]) / 100) * 100
                    ax[2,0].plot(years[0:-1], reg_ela_annual[0:-1], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,0].set_ylabel('ELA (m)')
                    ax[2,0].set_xlim(years.min(), years[0:-1].max())
    #                ax[2,0].set_ylim(ela_min, ela_max)
                    ax[2,0].tick_params(direction='inout', right=True)
                    
                    
                    # AAR
                    ax[2,1].plot(years, reg_aar_annual, color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,1].set_ylabel('AAR (-)')
                    ax[2,1].set_ylim(0,1)
                    ax[2,1].set_xlim(years.min(), years[0:-1].max())
                    ax[2,1].tick_params(direction='inout', right=True)
                    
                    
#                    # MASS BALANCE COMPONENTS
#                    # - these are only meant for monthly and/or relative purposes 
#                    #   mass balance from volume change should be used for annual changes
#                    reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
#                    # Refreeze
#                    reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
#                    # Melt
#                    reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
#                    # Frontal Ablation
#                    reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
#                    # Periods
#                    if reg_acc_annual.shape[0] == 101:
#                        period_yrs = 20
#                        periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
#                        reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
#                        
#                        # Convert to mwea
#                        reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
#                        reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
#                        reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
#                        reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
#                        reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
#                        reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
#                    else:
#                        assert True==False, 'Set up for different time periods'
#    
#                    # Plot
#                    ax[2,2].bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='refreeze', zorder=2)
#                    ax[2,2].bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='acc', zorder=3)
#                    if not reg_frontalablation_periods_mwea.sum() == 0:
#                        ax[2,2].bar(periods, -reg_frontalablation_periods_mwea, color='#83439A', width=period_yrs/2-1, label='frontal ablation', zorder=3)
#                    ax[2,2].bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='melt', zorder=2)
#                    ax[2,2].bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='total', zorder=1)
#                    ax[2,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
#                    ax[2,2].set_xlim(years.min(), years[0:-1].max())
#                    ax[2,2].xaxis.set_major_locator(MultipleLocator(100))
#                    ax[2,2].xaxis.set_minor_locator(MultipleLocator(20))
#                    ax[2,2].yaxis.set_major_locator(MultipleLocator(1))
#                    ax[2,2].yaxis.set_minor_locator(MultipleLocator(0.25))
#                    ax[2,2].tick_params(direction='inout', right=True)
#                    ax[2,2].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
#                                   loc=(1.2,0.25)) 
                    
                    
                    # Remove plot in lower right
                    fig.delaxes(ax[2,3])
                    
                    
                    # Title
                    fig.text(0.5, 0.95, rgi_reg_dict[reg] + ' (' + gcm_name + ' ' + rcp + ')', size=12, ha='center', va='top',)
                    
                    # Save figure
                    fig_fn = str(reg) + '_allplots_' + str(years.min()) + '-' + str(years.max()) + '_' + gcm_name + '_' + rcp + '.png'
                    fig.set_size_inches(8,6)
                    fig.savefig(fig_fp_reg + fig_fn, bbox_inches='tight', dpi=300)
            

#%%
if option_calving_mbclim_era5:
    
    calving_rgiid_fn = '/Users/drounce/Documents/HiMAT/calving_data/rgiids_for_will.csv'
    sim_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_Will-calving/'
    rgiid_df = pd.read_csv(calving_rgiid_fn)
    rgiids = list(rgiid_df.rgiid.values)
    glac_str = [x.split('-')[1] for x in rgiids]
    glac_no_list = sorted([str(int(x.split('.')[0])) + '.' + x.split('.')[1] for x in glac_str])

    # Load glaciers
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glac_no_list)

    #%%
    mbclim_term_decadal_all = None
    glacno_list_all = []
    for nglac, glacno in enumerate(glac_no_list[0:1]):
        if nglac%10 == 0:
            print(nglac, glacno)
        
        reg = glacno.split('.')[0]
        
        # Filepath and filename
        era5_fp_binned = sim_fp + str(reg).zfill(2) + '/ERA5/binned/'
        glac_fn_binned = glacno + '_ERA5_MCMC_ba1_50sets_2000_2019_binned.nc'
        
        if os.path.exists(era5_fp_binned + glac_fn_binned):
            ds = xr.open_dataset(era5_fp_binned + glac_fn_binned)
    
            mbclim_binned = ds.bin_massbalclim_annual.values[0,:,:-1]
            mbclim_binned_mad = ds.bin_massbalclim_annual_mad.values[0,:,:-1]
            
            # Terminus average (lower 2 bins)
            mbclim_term = mbclim_binned[-2:,:].mean(0)
            mbclim_term_mad = mbclim_binned_mad[-2:,:].mean(0)
            # Decadal average at the terminus (mwea)
            mbclim_term_decadal = mbclim_term.reshape(-1,10).mean(1)
            mbclim_term_decadal_mad = mbclim_term_mad.reshape(-1,10).mean(1)
                
            mbclim_term_decadal_output = np.concatenate((mbclim_term_decadal[np.newaxis,:], 
                                                         mbclim_term_decadal_mad[np.newaxis,:]), axis=1)

            if mbclim_term_decadal_all is None:
                mbclim_term_decadal_all = mbclim_term_decadal_output   
            else:
                mbclim_term_decadal_all = np.concatenate((mbclim_term_decadal_all, 
                                                          mbclim_term_decadal_output), axis=0)
            glacno_list_all.append(glacno)
                    
    mbclim_term_df_cns = ['RGIId', 'mbclim_mwea_2000_2009', 'mbclim_mwea_2010_2019', 
                          'mbclim_mwea_2000_2009_mad', 'mbclim_mwea_2010_2019_mad']
    mbclim_term_df = pd.DataFrame(np.zeros((len(glacno_list_all), mbclim_term_decadal_all.shape[1] +1)), columns=mbclim_term_df_cns)
    mbclim_term_df.loc[:,'RGIId'] = glacno_list_all
    mbclim_term_df.loc[:,mbclim_term_df_cns[1:]] = mbclim_term_decadal_all
    
    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    mbclim_term_df_fn = 'ERA5_2000_2019_calving_mbclim_term_mwea.csv'
    mbclim_term_df.to_csv(csv_fp + mbclim_term_df_fn, index=False)


#%%
if option_multigcm_plots_reg:
    
    netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + '/figures/'
    csv_fp = analysis_fp + '/csv/'
    pickle_fp = analysis_fp + '/pickle/'
    
#    fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
#    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
#    pickle_fp = fig_fp + '../pickle/'
    
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
#    rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    
#    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
    
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_bwl = {}
    reg_area_all = {} 
    reg_runoff_all = {}
    reg_melt_all = {}
    reg_acc_all = {}
    reg_refreeze_all = {}
    reg_fa_all = {}
    
    # Set up Global region
    reg_vol_all['all'] = {}
    reg_vol_all_bwl['all'] = {}
    reg_area_all['all'] = {}
    reg_runoff_all['all'] = {}
    reg_melt_all['all'] = {}
    reg_acc_all['all'] = {}
    reg_refreeze_all['all'] = {}
    reg_fa_all['all'] = {}
    
    reg_vol_all['all_no519'] = {}
    reg_vol_all_bwl['all_no519'] = {}
    reg_area_all['all_no519'] = {}
    reg_runoff_all['all_no519'] = {}
    reg_melt_all['all_no519'] = {}
    reg_acc_all['all_no519'] = {}
    reg_refreeze_all['all_no519'] = {}
    reg_fa_all['all_no519'] = {}
    for rcp in rcps:
        reg_vol_all['all'][rcp] = {}
        reg_vol_all_bwl['all'][rcp] = {}
        reg_area_all['all'][rcp] = {}
        reg_runoff_all['all'][rcp] = {}
        reg_melt_all['all'][rcp] = {}
        reg_acc_all['all'][rcp] = {}
        reg_refreeze_all['all'][rcp] = {}
        reg_fa_all['all'][rcp] = {}
        
        reg_vol_all['all_no519'][rcp] = {}
        reg_vol_all_bwl['all_no519'][rcp] = {}
        reg_area_all['all_no519'][rcp] = {}
        reg_runoff_all['all_no519'][rcp] = {}
        reg_melt_all['all_no519'][rcp] = {}
        reg_acc_all['all_no519'][rcp] = {}
        reg_refreeze_all['all_no519'][rcp] = {}
        reg_fa_all['all_no519'][rcp] = {}
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            reg_vol_all['all'][rcp][gcm_name] = None
            reg_vol_all_bwl['all'][rcp][gcm_name] = None
            reg_area_all['all'][rcp][gcm_name] = None
            reg_runoff_all['all'][rcp][gcm_name] = None
            reg_melt_all['all'][rcp][gcm_name] = None
            reg_acc_all['all'][rcp][gcm_name] = None
            reg_refreeze_all['all'][rcp][gcm_name] = None
            reg_fa_all['all'][rcp][gcm_name] = None
            
            reg_vol_all['all_no519'][rcp][gcm_name] = None
            reg_vol_all_bwl['all_no519'][rcp][gcm_name] = None
            reg_area_all['all_no519'][rcp][gcm_name] = None
            reg_runoff_all['all_no519'][rcp][gcm_name] = None
            reg_melt_all['all_no519'][rcp][gcm_name] = None
            reg_acc_all['all_no519'][rcp][gcm_name] = None
            reg_refreeze_all['all_no519'][rcp][gcm_name] = None
            reg_fa_all['all_no519'][rcp][gcm_name] = None
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_bwl[reg] = {}
        reg_area_all[reg] = {}
        reg_runoff_all[reg] = {}
        reg_melt_all[reg] = {}
        reg_acc_all[reg] = {}
        reg_refreeze_all[reg] = {}
        reg_fa_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_bwl[reg][rcp] = {}
            reg_area_all[reg][rcp] = {}
            reg_runoff_all[reg][rcp] = {}
            reg_melt_all[reg][rcp] = {}
            reg_acc_all[reg][rcp] = {}
            reg_refreeze_all[reg][rcp] = {}
            reg_fa_all[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
                    pickle_fp_reg =  (netcdf_fp_cmip5_land + '../analysis/pickle/' + str(reg).zfill(2) + 
                                      '/O1Regions/' + gcm_name + '/' + rcp + '/')                    
                else:
                    pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                
                # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                # Volume below sea level
                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                    reg_vol_annual_bwl = pickle.load(f)
                # Volume below debris
                with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                    reg_vol_annual_bd = pickle.load(f)
                # Area 
                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                    reg_area_annual = pickle.load(f)
                # Area below debris
                with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                    reg_area_annual_bd = pickle.load(f)
                # Binned Volume
                with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                    reg_vol_annual_binned = pickle.load(f)
                # Binned Volume below debris
                with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                    reg_vol_annual_binned_bd = pickle.load(f)
                # Binned Area
                with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                    reg_area_annual_binned = pickle.load(f)
                # Binned Area below debris
                with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                    reg_area_annual_binned_bd = pickle.load(f)
                # Mass balance: accumulation
                with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                    reg_acc_monthly = pickle.load(f)
                # Mass balance: refreeze
                with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                    reg_refreeze_monthly = pickle.load(f)
                # Mass balance: melt
                with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                    reg_melt_monthly = pickle.load(f)
                # Mass balance: frontal ablation
                with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                    reg_frontalablation_monthly = pickle.load(f)
                # Mass balance: total mass balance
                with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
                    reg_massbaltotal_monthly = pickle.load(f)
                # Binned Climatic Mass Balance
                with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                    reg_mbclim_annual_binned = pickle.load(f)
                # Runoff: moving-gauged
                with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                    reg_runoff_monthly_moving = pickle.load(f)
                # Runoff: fixed-gauged
                with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                    reg_runoff_monthly_fixed = pickle.load(f)
                # Runoff: precipitation
                with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
                    reg_prec_monthly = pickle.load(f)
                # Runoff: off-glacier precipitation
                with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
                    reg_offglac_prec_monthly = pickle.load(f)
                # Runoff: off-glacier melt
                with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
                    reg_offglac_melt_monthly = pickle.load(f)
                # Runoff: off-glacier refreeze
                with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
                    reg_offglac_refreeze_monthly = pickle.load(f)
                # ELA
                with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                    reg_ela_annual = pickle.load(f)
#                # AAR
#                with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
#                    reg_aar_annual = pickle.load(f)
                    
                    
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                if reg_vol_annual_bwl is None:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = np.zeros(reg_vol_annual.shape)
                else:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = reg_vol_annual_bwl
                reg_area_all[reg][rcp][gcm_name] = reg_area_annual  
                reg_runoff_all[reg][rcp][gcm_name] = reg_runoff_monthly_fixed
                reg_melt_all[reg][rcp][gcm_name] = reg_melt_monthly
                reg_acc_all[reg][rcp][gcm_name] = reg_acc_monthly
                reg_refreeze_all[reg][rcp][gcm_name] = reg_refreeze_monthly
                reg_fa_all[reg][rcp][gcm_name] = reg_frontalablation_monthly
                
                if reg_vol_all['all'][rcp][gcm_name] is None:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all[reg][rcp][gcm_name]
                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all[reg][rcp][gcm_name]
                    reg_melt_all['all'][rcp][gcm_name] = reg_melt_all[reg][rcp][gcm_name]
                    reg_acc_all['all'][rcp][gcm_name] = reg_acc_all[reg][rcp][gcm_name]
                    reg_refreeze_all['all'][rcp][gcm_name] = reg_refreeze_all[reg][rcp][gcm_name]
                    reg_fa_all['all'][rcp][gcm_name] = reg_fa_all[reg][rcp][gcm_name]
                    
                    if reg not in [5,19]:
                        reg_vol_all['all_no519'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
                        reg_vol_all_bwl['all_no519'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
                        reg_area_all['all_no519'][rcp][gcm_name] = reg_area_all[reg][rcp][gcm_name]
                        reg_runoff_all['all_no519'][rcp][gcm_name] = reg_runoff_all[reg][rcp][gcm_name]
                        reg_melt_all['all_no519'][rcp][gcm_name] = reg_melt_all[reg][rcp][gcm_name]
                        reg_acc_all['all_no519'][rcp][gcm_name] = reg_acc_all[reg][rcp][gcm_name]
                        reg_refreeze_all['all_no519'][rcp][gcm_name] = reg_refreeze_all[reg][rcp][gcm_name]
                        reg_fa_all['all_no519'][rcp][gcm_name] = reg_fa_all[reg][rcp][gcm_name]
                    
                else:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all['all'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl['all'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all['all'][rcp][gcm_name] + reg_area_all[reg][rcp][gcm_name]
                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all['all'][rcp][gcm_name] + reg_runoff_all[reg][rcp][gcm_name]
                    reg_melt_all['all'][rcp][gcm_name] = reg_melt_all['all'][rcp][gcm_name] + reg_melt_all[reg][rcp][gcm_name]
                    reg_acc_all['all'][rcp][gcm_name] = reg_acc_all['all'][rcp][gcm_name] + reg_acc_all[reg][rcp][gcm_name]
                    reg_refreeze_all['all'][rcp][gcm_name] = reg_refreeze_all['all'][rcp][gcm_name] + reg_refreeze_all[reg][rcp][gcm_name]
                    reg_fa_all['all'][rcp][gcm_name] = reg_fa_all['all'][rcp][gcm_name] + reg_fa_all[reg][rcp][gcm_name]
                    
                    if reg not in [5,19]:
                        reg_vol_all['all_no519'][rcp][gcm_name] = reg_vol_all['all_no519'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
                        reg_vol_all_bwl['all_no519'][rcp][gcm_name] = reg_vol_all_bwl['all_no519'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
                        reg_area_all['all_no519'][rcp][gcm_name] = reg_area_all['all_no519'][rcp][gcm_name] + reg_area_all[reg][rcp][gcm_name]
                        reg_runoff_all['all_no519'][rcp][gcm_name] = reg_runoff_all['all_no519'][rcp][gcm_name] + reg_runoff_all[reg][rcp][gcm_name]
                        reg_melt_all['all_no519'][rcp][gcm_name] = reg_melt_all['all_no519'][rcp][gcm_name] + reg_melt_all[reg][rcp][gcm_name]
                        reg_acc_all['all_no519'][rcp][gcm_name] = reg_acc_all['all_no519'][rcp][gcm_name] + reg_acc_all[reg][rcp][gcm_name]
                        reg_refreeze_all['all_no519'][rcp][gcm_name] = reg_refreeze_all['all_no519'][rcp][gcm_name] + reg_refreeze_all[reg][rcp][gcm_name]
                        reg_fa_all['all_no519'][rcp][gcm_name] = reg_fa_all['all_no519'][rcp][gcm_name] + reg_fa_all[reg][rcp][gcm_name]
                
             
    #%%
    regions.append('all_no519')
    regions.append('all')
    # MULTI-GCM STATISTICS
    ds_multigcm_vol = {}
    ds_multigcm_vol_bsl = {}
    ds_multigcm_area = {}
    ds_multigcm_runoff = {}
    ds_multigcm_melt = {}
    ds_multigcm_acc = {}
    ds_multigcm_refreeze = {}
    ds_multigcm_fa = {}
    for reg in regions:
        ds_multigcm_vol[reg] = {}
        ds_multigcm_vol_bsl[reg] = {}
        ds_multigcm_area[reg] = {}
        ds_multigcm_runoff[reg] = {}
        ds_multigcm_melt[reg] = {}
        ds_multigcm_acc[reg] = {}
        ds_multigcm_refreeze[reg] = {}
        ds_multigcm_fa[reg] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)
    
                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_bsl_gcm = reg_vol_all_bwl[reg][rcp][gcm_name]
                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
                reg_runoff_gcm = reg_runoff_all[reg][rcp][gcm_name]
                reg_melt_gcm = reg_melt_all[reg][rcp][gcm_name]
                reg_acc_gcm = reg_acc_all[reg][rcp][gcm_name]
                reg_refreeze_gcm = reg_refreeze_all[reg][rcp][gcm_name]
                reg_fa_gcm = reg_fa_all[reg][rcp][gcm_name]
    
                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm   
                    reg_area_gcm_all = reg_area_gcm
                    reg_runoff_gcm_all = reg_runoff_gcm
                    reg_melt_gcm_all = reg_melt_gcm
                    reg_acc_gcm_all = reg_acc_gcm
                    reg_refreeze_gcm_all = reg_refreeze_gcm
                    reg_fa_gcm_all = reg_fa_gcm
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
                    reg_runoff_gcm_all = np.vstack((reg_runoff_gcm_all, reg_runoff_gcm))
                    reg_melt_gcm_all = np.vstack((reg_melt_gcm_all, reg_melt_gcm))
                    reg_acc_gcm_all = np.vstack((reg_acc_gcm_all, reg_acc_gcm))
                    reg_refreeze_gcm_all = np.vstack((reg_refreeze_gcm_all, reg_refreeze_gcm))
                    reg_fa_gcm_all = np.vstack((reg_fa_gcm_all, reg_fa_gcm))
            
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_vol_bsl[reg][rcp] = reg_vol_bsl_gcm_all
            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
            ds_multigcm_runoff[reg][rcp] = reg_runoff_gcm_all
            ds_multigcm_melt[reg][rcp] = reg_melt_gcm_all
            ds_multigcm_acc[reg][rcp] = reg_acc_gcm_all
            ds_multigcm_refreeze[reg][rcp] = reg_refreeze_gcm_all
            ds_multigcm_fa[reg][rcp] = reg_fa_gcm_all
    



        #%%
        # ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
    
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            ax[0,0].plot(years, reg_vol_med, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_vol_med + 1.96*reg_vol_mad, 
                                     reg_vol_med - 1.96*reg_vol_mad, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
               
        ax[0,0].set_ylabel('Volume (m$^{3}$)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
    #                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    #                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        if 'rcp26' in rcps and 'ssp126' in rcps:
            scenario_str = 'rcps_ssps'
        elif 'rcp26' in rcps:
            scenario_str = 'rcps'
        elif 'ssp126' in rcps:
            scenario_str = 'ssps'
        fig_fn = (str(reg) + '_volchange_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
                  '-' + scenario_str + '.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        
        #%%
        # ----- FIGURE: NORMALIZED VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
    
        normyear_idx = np.where(years == normyear)[0][0]
    
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            ax[0,0].plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1, zorder=4, label=rcp)
            
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                     reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax[0,0].set_ylabel('Mass (rel. to 2015)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].set_ylim(0,1)
        ax[0,0].xaxis.set_major_locator(MultipleLocator(20))
        ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
        ax[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
        ax[0,0].tick_params(direction='inout', right=True)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
    #                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    #                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = (str(reg) + '_volchangenorm_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
                  '-' + scenario_str + '.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    
    #%% ----- MULTI-GCM STATISTICS -----    
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms',
                          'Marzeion_slr_mmsle_mean', 'Edwards_slr_mmsle_mean',
                          'slr_mmSLE_med', 'slr_mmSLE_95', 'slr_mmSLE_mean', 'slr_mmSLE_std', 'slr_mmSLE_mad',
                          'mb_mmSLE_med', 'mb_mmSLE_95', 'mb_mmSLE_mean', 'mb_mmSLE_std', 'mb_mmSLE_mad', 
                          'slr_correction_mmSLE',
                          'slr_2090-2100_mmSLEyr_med', 'slr_2090-2100_mmSLEyr_mean', 'slr_2090-2100_mmSLEyr_std', 'slr_2090-2100_mmSLEyr_mad', 
                          'slr_max_mmSLEyr_med', 'slr_max_mmSLEyr_mean', 'slr_max_mmSLEyr_std', 'slr_max_mmSLEyr_mad', 
#                          'slr_mmSLE_fromlost_med', 'slr_mmSLE_fromlost_mean', 'slr_mmSLE_fromlost_std', 'slr_mmSLE_fromlost_mad',
                          'yr_max_slr_med', 'yr_max_slr_mean', 'yr_max_slr_std', 'yr_max_slr_mad',
                          'vol_lost_%_med', 'vol_lost_%_95', 'vol_lost_%_mean', 'vol_lost_%_std', 'vol_lost_%_mad',
                          'Marzeion_vol_lost_%_mean',
                          'area_lost_%_med', 'area_lost_%_mean', 'area_lost_%_std', 'area_lost_%_mad',
#                          'count_lost_med', 'count_lost_mean', 'count_lost_std', 'count_lost_mad',
#                          'count_lost_%_med', 'count_lost_%_mean', 'count_lost_%_std', 'count_lost_%_mad',
                          'mb_2090-2100_mmwea_med', 'mb_2090-2100_mmwea_mean', 'mb_2090-2100_mmwea_std', 'mb_2090-2100_mmwea_mad',
                          'mb_max_mmwea_med', 'mb_max_mmwea_mean', 'mb_max_mmwea_std', 'mb_max_mmwea_mad']
    
    
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(stats_overview_cns))), columns=stats_overview_cns)
    
    stats_overview_df.loc[0,'Edwards_slr_mmsle_mean'] = 80
    stats_overview_df.loc[1,'Edwards_slr_mmsle_mean'] = 115
    stats_overview_df.loc[3,'Edwards_slr_mmsle_mean'] = 170
    stats_overview_df.loc[4,'Marzeion_slr_mmsle_mean'] = 79
    stats_overview_df.loc[5,'Marzeion_slr_mmsle_mean'] = 119
    stats_overview_df.loc[6,'Marzeion_slr_mmsle_mean'] = 159
    
    stats_overview_df.loc[4,'Marzeion_vol_lost_%_mean'] = 18
    stats_overview_df.loc[5,'Marzeion_vol_lost_%_mean'] = 27
    stats_overview_df.loc[6,'Marzeion_vol_lost_%_mean'] = 36
    
    
    ncount = 0
    regions_overview = regions
    if 'all' in regions and 'all_no519' in regions:
        regions_overview = regions[-2:] + regions[0:-2]
    for nreg, reg in enumerate(regions_overview):
        for rcp in rcps:
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]
            
            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl, option='oggm')
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_med = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_mean = np.mean(reg_slr_cum, axis=0)
            reg_slr_cum_std = np.std(reg_slr_cum, axis=0)
            reg_slr_cum_mad = median_abs_deviation(reg_slr_cum, axis=0)
            
            # Mass change in SLE
            reg_slr_nocorrection = slr_mmSLEyr(reg_vol, reg_vol_bsl, option='None')
            reg_slr_nocorrection_cum_raw = np.cumsum(reg_slr_nocorrection, axis=1)
            reg_slr_nocorrection_cum = reg_slr_nocorrection_cum_raw - reg_slr_nocorrection_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_nocorrection_cum_med = np.median(reg_slr_nocorrection_cum, axis=0)
            reg_slr_nocorrection_cum_mean = np.mean(reg_slr_nocorrection_cum, axis=0)
            reg_slr_nocorrection_cum_std = np.std(reg_slr_nocorrection_cum, axis=0)
            reg_slr_nocorrection_cum_mad = median_abs_deviation(reg_slr_nocorrection_cum, axis=0)
            
#            # Cumulative Sea-level change from lost glaciers [mm SLE]
#            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
#            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
#            reg_slr_lost = ds_multigcm_glac_lost_slr[reg][rcp]
#            reg_slr_lost_cum_raw = np.cumsum(reg_slr_lost, axis=1)
#            reg_slr_lost_cum = reg_slr_lost_cum_raw - reg_slr_lost_cum_raw[:,normyear_idx][:,np.newaxis]
#            
#            # See how much not accounting for bsl affects results and correct as an approximation
#            reg_slr_nobsl = slr_mmSLEyr(reg_vol, np.zeros(reg_vol.shape))
#            reg_slr_nobsl_cum_raw = np.cumsum(reg_slr_nobsl, axis=1)
#            reg_slr_nobsl_cum = reg_slr_nobsl_cum_raw - reg_slr_nobsl_cum_raw[:,normyear_idx][:,np.newaxis]
#            reg_slr_nobsl_cum_med = np.median(reg_slr_nobsl_cum, axis=0)
#            
#            reg_slr_lost_cum = reg_slr_lost_cum * reg_slr_cum_med[-1] / reg_slr_nobsl_cum_med[-1]
#            
#            reg_slr_lost_cum_med = np.median(reg_slr_lost_cum, axis=0)
#            reg_slr_lost_cum_mean = np.mean(reg_slr_lost_cum, axis=0)
#            reg_slr_lost_cum_std = np.std(reg_slr_lost_cum, axis=0)
#            reg_slr_lost_cum_mad = median_abs_deviation(reg_slr_lost_cum, axis=0)
            
            # Sea-level change rate [mm SLE yr-1]
            reg_slr_20902100_med = np.median(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_mean = np.mean(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_std = np.std(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_mad = median_abs_deviation(reg_slr[:,-10:], axis=(0,1))
            
            # Sea-level change max rate
            reg_slr_med_raw = np.median(reg_slr, axis=0)
            reg_slr_mean_raw = np.mean(reg_slr, axis=0)
            reg_slr_std_raw = np.std(reg_slr, axis=0)
            reg_slr_mad_raw = median_abs_deviation(reg_slr, axis=0) 
            reg_slr_med = uniform_filter(reg_slr_med_raw, size=(11))
            slr_max_idx = np.where(reg_slr_med == reg_slr_med.max())[0]
            reg_slr_mean = uniform_filter(reg_slr_mean_raw, size=(11))
            reg_slr_std = uniform_filter(reg_slr_std_raw, size=(11))
            reg_slr_mad = uniform_filter(reg_slr_mad_raw, size=(11))
            
            # Year of maximum sea-level change rate
            #  - use a median filter to sort through the peaks which otherwise don't get smoothed with mean
            reg_slr_uniformfilter = np.zeros(reg_slr.shape)
            for nrow in np.arange(reg_slr.shape[0]):
                reg_slr_uniformfilter[nrow,:] = generic_filter(reg_slr[nrow,:], np.median, size=(11))
            reg_yr_slr_max = years[np.argmax(reg_slr_uniformfilter, axis=1)]
            reg_yr_slr_max[reg_yr_slr_max < normyear] = normyear
            reg_yr_slr_max_med = np.median(reg_yr_slr_max)
            reg_yr_slr_max_mean = np.mean(reg_yr_slr_max)
            reg_yr_slr_max_std = np.std(reg_yr_slr_max)
            reg_yr_slr_max_mad = median_abs_deviation(reg_yr_slr_max)
#            if reg in ['all']:
#                print('\n ', deg_group)
#                print('  ', reg_yr_slr_max_med, reg_yr_slr_max_mean, reg_yr_slr_max_std, reg_yr_slr_max_mad)
            
            # Volume lost [%]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_std = np.std(reg_vol, axis=0)
            reg_vol_mean = np.mean(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            reg_vol_lost_med_norm = (1 - reg_vol_med / reg_vol_med[normyear_idx]) * 100
            reg_vol_lost_mean_norm = (1 - reg_vol_mean / reg_vol_mean[normyear_idx]) * 100
            reg_vol_lost_std_norm = reg_vol_std / reg_vol_med[normyear_idx] * 100
            reg_vol_lost_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx] * 100
            
            # Area lost [%]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_std = np.std(reg_area, axis=0)
            reg_area_mean = np.mean(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            reg_area_lost_med_norm = (1 - reg_area_med / reg_area_med[normyear_idx]) * 100
            reg_area_lost_mean_norm = (1 - reg_area_mean / reg_area_mean[normyear_idx]) * 100
            reg_area_lost_std_norm = reg_area_std / reg_area_med[normyear_idx] * 100
            reg_area_lost_mad_norm = reg_area_mad / reg_area_med[normyear_idx] * 100
            
#            # Glaciers lost [count]
#            reg_glac_lost = ds_multigcm_glac_lost_bydeg[reg][deg_group]
#            reg_glac_lost_med = np.median(reg_glac_lost, axis=0)
#            reg_glac_lost_mean = np.mean(reg_glac_lost, axis=0)
#            reg_glac_lost_std = np.std(reg_glac_lost, axis=0)
#            reg_glac_lost_mad = median_abs_deviation(reg_glac_lost, axis=0)
#            
#            # Glaciers lost [%]
#            glac_count_total = np.median(temp_dev_df['Glac_count-' + str(reg)])
#            reg_glac_lost_med_norm = np.median(reg_glac_lost, axis=0) / glac_count_total * 100
#            reg_glac_lost_mean_norm = np.mean(reg_glac_lost, axis=0) / glac_count_total * 100
#            reg_glac_lost_std_norm = np.std(reg_glac_lost, axis=0) / glac_count_total * 100
#            reg_glac_lost_mad_norm = median_abs_deviation(reg_glac_lost, axis=0) / glac_count_total * 100
            
            # Specific mass balance [kg m-2 yr-1 or mm w.e. yr-1]
            reg_mass = reg_vol * pygem_prms.density_ice
            reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
            reg_mb_20902100_med = np.median(reg_mb[:,-10:], axis=(0,1))
            reg_mb_20902100_mean = np.mean(reg_mb[:,-10:], axis=(0,1))
            reg_mb_20902100_std = np.std(reg_mb[:,-10:], axis=(0,1))
            reg_mb_20902100_mad = median_abs_deviation(reg_mb[:,-10:], axis=(0,1))
            
            # Mass balance max rate
            reg_mb_med_raw = np.median(reg_mb, axis=0)
            reg_mb_mean_raw = np.mean(reg_mb, axis=0)
            reg_mb_std_raw = np.std(reg_mb, axis=0)
            reg_mb_mad_raw = median_abs_deviation(reg_mb, axis=0) 
            reg_mb_med = uniform_filter(reg_mb_med_raw, size=(11))
            mb_max_idx = np.where(reg_mb_med == reg_mb_med.min())[0]
            reg_mb_mean = uniform_filter(reg_mb_mean_raw, size=(11))
            reg_mb_std = uniform_filter(reg_mb_std_raw, size=(11))
            reg_mb_mad = uniform_filter(reg_mb_mad_raw, size=(11))
            
#            if reg in ['all']:
#                print(reg_mb_med)
#                print(reg_mb_med.min(), reg_mb_med[mb_max_idx])
            
            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = rcp
            stats_overview_df.loc[ncount,'n_gcms'] = reg_vol.shape[0]
            stats_overview_df.loc[ncount,'slr_mmSLE_med'] = reg_slr_cum_med[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_mean'] = reg_slr_cum_mean[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_std'] = reg_slr_cum_std[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_mad'] = reg_slr_cum_mad[-1]
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_med'] = np.max(reg_slr_med)
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_mean'] = np.max(reg_slr_mean)
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_std'] = reg_slr_std[slr_max_idx]
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_mad'] = reg_slr_mad[slr_max_idx]
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_med'] = reg_slr_20902100_med
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_mean'] = reg_slr_20902100_mean
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_std'] = reg_slr_20902100_std
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_mad'] = reg_slr_20902100_mad
            stats_overview_df.loc[ncount,'mb_mmSLE_med'] = reg_slr_nocorrection_cum_med[-1]
            stats_overview_df.loc[ncount,'mb_mmSLE_mean'] = reg_slr_nocorrection_cum_mean[-1]
            stats_overview_df.loc[ncount,'mb_mmSLE_std'] = reg_slr_nocorrection_cum_std[-1]
            stats_overview_df.loc[ncount,'mb_mmSLE_mad'] = reg_slr_nocorrection_cum_mad[-1]
#            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_med'] = reg_slr_lost_cum_med[-1]
#            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_mean'] = reg_slr_lost_cum_mean[-1]
#            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_std'] = reg_slr_lost_cum_std[-1]
#            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_mad'] = reg_slr_lost_cum_mad[-1]
            stats_overview_df.loc[ncount,'yr_max_slr_med'] = reg_yr_slr_max_med
            stats_overview_df.loc[ncount,'yr_max_slr_mean'] = reg_yr_slr_max_mean
            stats_overview_df.loc[ncount,'yr_max_slr_std'] = reg_yr_slr_max_std
            stats_overview_df.loc[ncount,'yr_max_slr_mad'] = reg_yr_slr_max_mad
            stats_overview_df.loc[ncount,'vol_lost_%_med'] = reg_vol_lost_med_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_mean'] = reg_vol_lost_mean_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_std'] = reg_vol_lost_std_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_mad'] = reg_vol_lost_mad_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_med'] = reg_area_lost_med_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_mean'] = reg_area_lost_mean_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_std'] = reg_area_lost_std_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_mad'] = reg_area_lost_mad_norm[-1]
#            stats_overview_df.loc[ncount,'count_lost_med'] = reg_glac_lost_med[-1]
#            stats_overview_df.loc[ncount,'count_lost_mean'] = reg_glac_lost_mean[-1]
#            stats_overview_df.loc[ncount,'count_lost_std'] = reg_glac_lost_std[-1]
#            stats_overview_df.loc[ncount,'count_lost_mad'] = reg_glac_lost_mad[-1]
#            stats_overview_df.loc[ncount,'count_lost_%_med'] = reg_glac_lost_med_norm[-1]
#            stats_overview_df.loc[ncount,'count_lost_%_mean'] = reg_glac_lost_mean_norm[-1]
#            stats_overview_df.loc[ncount,'count_lost_%_std'] = reg_glac_lost_std_norm[-1]
#            stats_overview_df.loc[ncount,'count_lost_%_mad'] = reg_glac_lost_mad_norm[-1]
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_med'] = reg_mb_20902100_med
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_mean'] = reg_mb_20902100_mean
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_std'] = reg_mb_20902100_std
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_mad'] = reg_mb_20902100_mad
            stats_overview_df.loc[ncount,'mb_max_mmwea_med'] = np.min(reg_mb_med)
            stats_overview_df.loc[ncount,'mb_max_mmwea_mean'] = np.min(reg_mb_mean)
            stats_overview_df.loc[ncount,'mb_max_mmwea_std'] = reg_mb_std[mb_max_idx]
            stats_overview_df.loc[ncount,'mb_max_mmwea_mad'] = reg_mb_mad[mb_max_idx]
            
            ncount += 1
    
    stats_overview_df['slr_correction_mmSLE'] = stats_overview_df['mb_mmSLE_med'] - stats_overview_df['slr_mmSLE_med']
    stats_overview_df['slr_mmSLE_95'] = 1.96*stats_overview_df['slr_mmSLE_std']
    stats_overview_df['mb_mmSLE_95'] = 1.96*stats_overview_df['mb_mmSLE_std']
    stats_overview_df['vol_lost_%_95'] = 1.96*stats_overview_df['vol_lost_%_std']
    stats_overview_df['slr_Marzeion_dif%'] = 100* stats_overview_df['slr_mmSLE_med'] / stats_overview_df['Marzeion_slr_mmsle_mean']
    stats_overview_df['slr_Edwards_dif%'] = 100 * stats_overview_df['slr_mmSLE_med'] / stats_overview_df['Edwards_slr_mmsle_mean']
    stats_overview_df.to_csv(csv_fp + 'stats_overview.csv', index=False)
    
    regions.remove('all_no519')
    #%%
    # ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    # Record max specific mass balance
    vol_lost_cns = ['Region']
    for rcp in rcps:
        vol_lost_cns.append('vol_lost_%_' + rcp)
        vol_lost_cns.append('vol_lost_mad_%_' + rcp)
    vol_lost_df = pd.DataFrame(np.zeros((len(regions),len(rcps)*2+1)), columns=vol_lost_cns)
    vol_lost_df['Region'] = regions
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            # Record statistics
            reg_idx = regions.index(reg)
            vol_lost_df.loc[reg_idx,'vol_lost_%_' + rcp] = (1 - reg_vol_med_norm[-1])*100
            vol_lost_df.loc[reg_idx,'vol_lost_mad_%_' + rcp] = reg_vol_mad_norm[-1]*100
            
            ax.plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years, 
                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Mass (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    vol_lost_fn = 'vol_lost_norm_statistics.csv'
    vol_lost_df.to_csv(csv_fp + vol_lost_fn, index=False)
    
    
    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED AREA CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_area = ds_multigcm_area[reg][rcp]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            
            reg_area_med_norm = reg_area_med / reg_area_med[normyear_idx]
            reg_area_mad_norm = reg_area_mad / reg_area_med[normyear_idx]
            
            ax.plot(years, reg_area_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years, 
                                reg_area_med_norm + 1.96*reg_area_mad_norm, 
                                reg_area_med_norm - 1.96*reg_area_mad_norm, 
                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Area (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_areachange_norm_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: ALL SPECIFIC MASS LOSS RATES -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_mass = reg_vol * pygem_prms.density_ice
            reg_area = ds_multigcm_area[reg][rcp]

            # Specific mass change rate
            reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
            reg_mb_med = np.median(reg_mb, axis=0)
            reg_mb_mad = median_abs_deviation(reg_mb, axis=0)
            
            ax.plot(years[0:-1], reg_mb_med / 1000, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_mb_med + 1.96*reg_mb_mad) / 1000, 
                                (reg_mb_med - 1.96*reg_mb_mad) / 1000, 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M/$\Delta$t (m w.e. yr$^{-1}$)')
            ax.set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
        ax.set_ylim(-5.5,0.5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_mb_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: ALL SPECIFIC MASS LOSS RATES - w running mean! -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    # Record max specific mass balance
    mb_mwea_max_cns = ['Region']
    for rcp in rcps:
        mb_mwea_max_cns.append('mb_mwea_' + rcp)
        mb_mwea_max_cns.append('mb_mwea_mad_' + rcp)
        mb_mwea_max_cns.append('year_' + rcp)
    mb_mwea_max_df = pd.DataFrame(np.zeros((len(regions),len(rcps)*3+1)), columns=mb_mwea_max_cns)
    mb_mwea_max_df['Region'] = regions
    
    mb_gta_max_cns = ['Region']
    for rcp in rcps:
        mb_gta_max_cns.append('mb_gta_' + rcp)
        mb_gta_max_cns.append('mb_gta_mad_' + rcp)
        mb_gta_max_cns.append('year_' + rcp)
    mb_gta_max_df = pd.DataFrame(np.zeros((len(regions),len(rcps)*3+1)), columns=mb_gta_max_cns)
    mb_gta_max_df['Region'] = regions
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_mass = reg_vol * pygem_prms.density_ice
            reg_area = ds_multigcm_area[reg][rcp]

            # Specific mass change rate
            reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
            reg_mb_med_raw = np.median(reg_mb, axis=0)
            reg_mb_mad_raw = median_abs_deviation(reg_mb, axis=0)            
            reg_mb_med = uniform_filter(reg_mb_med_raw, size=(11))
            reg_mb_mad = uniform_filter(reg_mb_mad_raw, size=(11))
            
            # Record max mb
            reg_mb_gta = (reg_mass[:,1:] - reg_mass[:,0:-1]) / 1e12
            reg_mb_gta_med_raw = np.median(reg_mb_gta, axis=0)
            reg_mb_gta_mad_raw = median_abs_deviation(reg_mb_gta, axis=0)            
            reg_mb_gta_med = uniform_filter(reg_mb_gta_med_raw, size=(11))
            reg_mb_gta_mad = uniform_filter(reg_mb_gta_mad_raw, size=(11))
            
            mb_df_idx = regions.index(reg)
            mb_max_idx = np.where(reg_mb_gta_med == reg_mb_gta_med.min())
            mb_gta_max_df.loc[mb_df_idx,'mb_gta_' + rcp] = reg_mb_gta_med.min()
            mb_gta_max_df.loc[mb_df_idx,'mb_gta_mad_' + rcp] = reg_mb_gta_mad[mb_max_idx]
            mb_gta_max_df.loc[mb_df_idx,'year_' + rcp] = years[0:-1][mb_max_idx]
            
#            print(reg, rcp, np.round(reg_mb_gta_med.sum()), 'Gt')
            
            # Record max mb_mwea
            mb_mwea_max_idx = np.where(reg_mb_med == reg_mb_med.min())
            mb_mwea_max_df.loc[mb_df_idx,'mb_mwea_' + rcp] = reg_mb_med.min() / 1000
            mb_mwea_max_df.loc[mb_df_idx,'mb_mwea_mad_' + rcp] = reg_mb_mad[mb_mwea_max_idx] / 1000
            mb_mwea_max_df.loc[mb_df_idx,'year_' + rcp] = years[0:-1][mb_mwea_max_idx]
            
            # Plot
            ax.plot(years[0:-1], reg_mb_med / 1000, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_mb_med + 1.96*reg_mb_mad) / 1000, 
                                (reg_mb_med - 1.96*reg_mb_mad) / 1000, 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M/$\Delta$t (m w.e. yr$^{-1}$)')
            ax.set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
        ax.set_ylim(-5.5,0.5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_mb_11yrmean_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    mb_mwea_max_fn = 'mb_mwea_max_statistics.csv'
    mb_mwea_max_df.to_csv(csv_fp + mb_mwea_max_fn, index=False)
    
    mb_gta_max_fn = 'mb_gta_max_statistics.csv'
    mb_gta_max_df.to_csv(csv_fp + mb_gta_max_fn, index=False)
    
    #%% ----- FIGURE: ANNUAL SEA-LEVEL RISE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    # Record max mm SLE yr
    mb_mmslea_max_cns = ['Region']
    for rcp in rcps:
        mb_mmslea_max_cns.append('mb_mmSLEyr_' + rcp)
        mb_mmslea_max_cns.append('mb_mmSLEyr_mad_' + rcp)
        mb_mmslea_max_cns.append('year_' + rcp)
        mb_mmslea_max_cns.append('mb_mmSLEyr_2100_' + rcp)
        mb_mmslea_max_cns.append('mb_mmSLEyr_mad_2100_' + rcp)
    mb_mmslea_max_df = pd.DataFrame(np.zeros((len(regions),len(rcps)*5+1)), columns=mb_mmslea_max_cns)
    mb_mmslea_max_df['Region'] = regions
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]

            # Sea-level change [mm SLE yr-1]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            #  - assume all mass loss contributes to sea level rise
#                reg_slr = -1*(reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000
            
            reg_slr_med_raw = np.median(reg_slr, axis=0)
            reg_slr_mad_raw = median_abs_deviation(reg_slr, axis=0) 
            reg_slr_med = uniform_filter(reg_slr_med_raw, size=(11))
            reg_slr_mad = uniform_filter(reg_slr_mad_raw, size=(11))
            
            print(reg, rcp, np.max(reg_slr_med), np.where(reg_slr_med == reg_slr_med.max()))
            
            # Record max mm_SLE
            mb_df_idx = regions.index(reg)
            mb_mmslea_max_idx = np.where(reg_slr_med == reg_slr_med.max())
            mb_mmslea_max_df.loc[mb_df_idx,'mb_mmSLEyr_' + rcp] = reg_slr_med.max()
            mb_mmslea_max_df.loc[mb_df_idx,'mb_mmSLEyr_mad_' + rcp] = reg_slr_mad[mb_mwea_max_idx]
            mb_mmslea_max_df.loc[mb_df_idx,'year_' + rcp] = years[0:-1][mb_mmslea_max_idx]
            mb_mmslea_max_df.loc[mb_df_idx,'mb_mmSLEyr_2100_' + rcp] = reg_slr_med[-1]
            mb_mmslea_max_df.loc[mb_df_idx,'mb_mmSLEyr_mad_2100_' + rcp] = reg_slr_mad[-1]
            
            ax.plot(years[0:-1], reg_slr_med, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_slr_med + 1.96*reg_slr_mad), 
                                (reg_slr_med - 1.96*reg_slr_mad), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                
            
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('$\Delta$M/$\Delta$t\n(mm SLE yr$^{-1}$)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in [19, 3, 1]:
            ax.set_ylim(-0.01,0.9)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))    
        elif reg in [5, 9, 4, 7]:
            ax.set_ylim(-0.01,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [17, 13, 6, 14, 2, 13]:
            ax.set_ylim(-0.01,0.175)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        elif reg in [8, 10, 11, 16, 12, 18]:
            ax.set_ylim(-0.005,0.025)
            ax.yaxis.set_major_locator(MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(MultipleLocator(0.005)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_SLR_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    mb_slr_max_fn = 'mb_mmSLEyr_max_statistics.csv'
    mb_mmslea_max_df.to_csv(csv_fp + mb_slr_max_fn, index=False)
    
    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE -----
    slr_cum_cns = ['Region', 'Scenario', 'med_mmSLE', 'mean_mmSLE', 'std_mmSLE', 'mad_mmSLE']
    slr_cum_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(slr_cum_cns))), columns=slr_cum_cns)
    ncount = 0
    
    all_slr_cum_df = pd.DataFrame(np.zeros((len(rcps),len(years)-1)), columns=years[:-1])
    
    for nreg, reg in enumerate(regions):
        for nrcp, rcp in enumerate(rcps):
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]
            
            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_med = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_mean = np.mean(reg_slr_cum, axis=0)
            reg_slr_cum_std = np.std(reg_slr_cum, axis=0)
            reg_slr_cum_mad = median_abs_deviation(reg_slr_cum, axis=0)
            
            slr_cum_df.loc[ncount,'Region'] = reg
            slr_cum_df.loc[ncount,'Scenario'] = rcp
            slr_cum_df.loc[ncount,'med_mmSLE'] = reg_slr_cum_med[-1]
            slr_cum_df.loc[ncount,'mean_mmSLE'] = reg_slr_cum_mean[-1]
            slr_cum_df.loc[ncount,'std_mmSLE'] = reg_slr_cum_std[-1]
            slr_cum_df.loc[ncount,'mad_mmSLE'] = reg_slr_cum_mad[-1]
            
            if reg in ['all']:
                print(reg, rcp)
                all_slr_cum_df.iloc[nrcp,:] = reg_slr_cum_med
            
            ncount += 1
    slr_cum_df.to_csv(csv_fp + 'SLR_cum_2100_rel2015.csv', index=False)
    
    all_slr_cum_df.index = rcps
    all_slr_cum_df.to_csv(csv_fp + 'all_SLR_cum_mmSLE_2100_rel2015_timeseries.csv')
    #%%
            
#            print(reg, 'SLR (mm SLE):', rcp, np.round(reg_slr_cum_avg[-1],2), '+/-', np.round(reg_slr_cum_var[-1],2))
        
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    # Record max mm SLE yr
    slrcum_multigcm_cns = ['Region']
    for rcp in rcps:
        slrcum_multigcm_cns.append('slr_mmSLE_' + rcp)
        slrcum_multigcm_cns.append('slr_mmSLE_std_' + rcp)
    slrcum_multigcm_df = pd.DataFrame(np.zeros((len(regions),len(rcps)*2+1)), columns=slrcum_multigcm_cns)
    slrcum_multigcm_df['Region'] = regions
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
#            reg_vol_med[normyear_idx]
#            print('NORMALIZE ALL TO BE 0 AT NORMYEAR!')
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
#            reg_slr_cum_avg = np.mean(reg_slr_cum, axis=0)
            reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
#            reg_slr_cum_var = median_abs_deviation(reg_slr_cum, axis=0)
            
            # Record cumulative SLR
            slr_idx = regions.index(reg)
            slrcum_multigcm_df.loc[slr_idx,'slr_mmSLE_' + rcp] = reg_slr_cum_avg[-1]
            slrcum_multigcm_df.loc[slr_idx,'slr_mmSLE_std_' + rcp] = reg_slr_cum_var[-1]
            

            ax.plot(years[0:-1], reg_slr_cum_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
#                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                ax.fill_between(years[0:-1], 
                                (reg_slr_cum_avg + reg_slr_cum_var), 
                                (reg_slr_cum_avg - reg_slr_cum_var), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
            if reg in ['all']:
                print('global glacier SLR (mm SLE):', rcp, np.round(reg_slr_cum_avg[-1],2), '+/-', np.round(reg_slr_cum_var[-1],2))
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M (mm SLE)')
            ax.set_ylabel('SLR (mm SLE)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in ['all']:
            ax.set_ylim(0,250)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        if reg in [19, 3, 1]:
            ax.set_ylim(0,42)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))    
        elif reg in [5, 9, 4, 7]:
            ax.set_ylim(0,27)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,11)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1)) 
        elif reg in [15, 2]:
            ax.set_ylim(0,3)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
        elif reg in [8]:
            ax.set_ylim(0,0.8)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [10, 11, 16, 12, 18]:
            ax.set_ylim(0,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_SLR-cum_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    all_idx = list(slrcum_multigcm_df['Region']).index('all')
    for rcp in rcps:
        slrcum_multigcm_df['slr_mmSLE_' + rcp + '_%'] = (
                slrcum_multigcm_df['slr_mmSLE_' + rcp] / slrcum_multigcm_df.loc[all_idx,'slr_mmSLE_' + rcp] * 100)
        slrcum_multigcm_df['slr_mmSLE_std_' + rcp + '_%'] =  (
                slrcum_multigcm_df['slr_mmSLE_std_' + rcp] / slrcum_multigcm_df.loc[all_idx,'slr_mmSLE_std_' + rcp] * 100)
    slr_cum_multigcm_fn = 'SLR_cum_multigcm_statistics.csv'
    slrcum_multigcm_df.to_csv(csv_fp + slr_cum_multigcm_fn, index=False)
    
    
    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE w BOX AND WHISKERS -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=8,wspace=0.66,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:3])
    ax1b = fig.add_subplot(gs[0:2,2:4])
    ax2 = fig.add_subplot(gs[0,6:])
    ax3 = fig.add_subplot(gs[1,4:6])
    ax4 = fig.add_subplot(gs[1,6:])
    ax5 = fig.add_subplot(gs[2,0:2])
    ax6 = fig.add_subplot(gs[2,2:4])
    ax7 = fig.add_subplot(gs[2,4:6])
    ax8 = fig.add_subplot(gs[2,6:])
    ax9 = fig.add_subplot(gs[3,0:2])
    ax10 = fig.add_subplot(gs[3,2:4])
    ax11 = fig.add_subplot(gs[3,4:6])
    ax12 = fig.add_subplot(gs[3,6:])
    ax13 = fig.add_subplot(gs[4,0:2])
    ax14 = fig.add_subplot(gs[4,2:4])
    ax15 = fig.add_subplot(gs[4,4:6])
    ax16 = fig.add_subplot(gs[4,6:])
    ax17 = fig.add_subplot(gs[5,0:2])
    ax18 = fig.add_subplot(gs[5,2:4])
    ax19 = fig.add_subplot(gs[5,4:6])
    ax20 = fig.add_subplot(gs[5,6:])
    
    data_boxplot = []
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
            
            ax.plot(years[0:-1], reg_slr_cum_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
#                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                ax.fill_between(years[0:-1], 
                                (reg_slr_cum_avg + reg_slr_cum_var), 
                                (reg_slr_cum_avg - reg_slr_cum_var), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
            
            # Aggregate boxplot data
            if reg in ['all']:
                data_boxplot.append(reg_slr_cum[:,-1])
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M (mm SLE)')
            ax.set_ylabel('SLR (mm SLE)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in ['all']:
            ax.set_ylim(0,250)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        if reg in [19, 3, 1]:
            ax.set_ylim(0,42)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))    
        elif reg in [5, 9, 4, 7]:
            ax.set_ylim(0,27)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,11)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1)) 
        elif reg in [15, 2]:
            ax.set_ylim(0,3)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
        elif reg in [8]:
            ax.set_ylim(0,0.8)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [10, 11, 16, 12, 18]:
            ax.set_ylim(0,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                leg_cols = 2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                leg_cols = 2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                leg_cols = 1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                leg_cols = 1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                leg_cols = 1
            ax.legend(loc=(-1.34,0.2), labels=labels, fontsize=10, ncol=leg_cols, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    
#        bp_lc = []
#        for rcp in rcps:
#            bp_lc.append(rcp_colordict[rcp])
    bp = ax1b.boxplot(data_boxplot)
    ax1b.set_ylim(0,250)
    for nbox, box in enumerate(bp['boxes']):
        rcp = rcps[nbox]
        # change outline color
        box.set(color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1)
    for nitem, item in enumerate(bp['medians']):
        rcp = rcps[nitem]
        # change outline color
        item.set(color=rcp_colordict[rcp], linewidth=1)
    for nitem, item in enumerate(bp['whiskers']):
        rcp = rcps[int(np.floor(nitem/2))]
        # change outline color
        item.set(color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1)
    for nitem, item in enumerate(bp['caps']):
        rcp = rcps[int(np.floor(nitem/2))]
        # change outline color
        item.set(color=rcp_colordict[rcp], linewidth=1)
    # turn off axes
    ax1b.get_yaxis().set_visible(False)
    ax1b.get_xaxis().set_visible(False)
    ax1b.axis('off')
    ax1b.set_xlim(-5,8)
            
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_SLR-cum_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd-BoxWhisker.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
#%% ----- FIGURE: MASS BALANCE COMOPNENTS -----
    for rcp in rcps:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[0,3])
        ax3 = fig.add_subplot(gs[1,2])
        ax4 = fig.add_subplot(gs[1,3])
        ax5 = fig.add_subplot(gs[2,0])
        ax6 = fig.add_subplot(gs[2,1])
        ax7 = fig.add_subplot(gs[2,2])
        ax8 = fig.add_subplot(gs[2,3])
        ax9 = fig.add_subplot(gs[3,0])
        ax10 = fig.add_subplot(gs[3,1])
        ax11 = fig.add_subplot(gs[3,2])
        ax12 = fig.add_subplot(gs[3,3])
        ax13 = fig.add_subplot(gs[4,0])
        ax14 = fig.add_subplot(gs[4,1])
        ax15 = fig.add_subplot(gs[4,2])
        ax16 = fig.add_subplot(gs[4,3])
        ax17 = fig.add_subplot(gs[5,0])
        ax18 = fig.add_subplot(gs[5,1])
        ax19 = fig.add_subplot(gs[5,2])
        ax20 = fig.add_subplot(gs[5,3])
        
        regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
            
            reg = regions_ordered[nax]
            
            # MASS BALANCE COMPONENTS
            # - these are only meant for monthly and/or relative purposes 
            #   mass balance from volume change should be used for annual changes
            reg_area_annual = np.median(ds_multigcm_area[reg][rcp],axis=0)
            reg_melt_monthly = np.median(ds_multigcm_melt[reg][rcp],axis=0)
            reg_acc_monthly = np.median(ds_multigcm_acc[reg][rcp],axis=0)
            reg_refreeze_monthly = np.median(ds_multigcm_refreeze[reg][rcp],axis=0)
            reg_frontalablation_monthly = np.median(ds_multigcm_fa[reg][rcp],axis=0)
            
            
            reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
            # Refreeze
            reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
            # Melt
            reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
            # Frontal Ablation
            reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
            # Periods
            if reg_acc_annual.shape[0] == 101:
                period_yrs = 20
                periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
                reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
                
                # Convert to mwea
                reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
                reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
                reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
                reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
                reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
                reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
            else:
                assert True==False, 'Set up for different time periods'

            # Plot
            ax.bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='Refreeze', zorder=2)
            ax.bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='Accumulation', zorder=3)
            if not reg_frontalablation_periods_mwea.sum() == 0:
                ax.bar(periods, -reg_frontalablation_periods_mwea, color='#04D8B2', width=period_yrs/2-1, label='Frontal ablation', zorder=3)
            ax.bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='Melt', zorder=2)
            ax.bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='Mass balance (total)', zorder=1)
            
            ax.set_xlim(years.min(), years[0:-1].max())
            ax.set_ylim(-5.5,2.75)
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            
            if ax in [ax1,ax5,ax9,ax13,ax17]:
                ax.set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)')
                
            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)
            
            if nax == 0:
                ax.legend(loc=(1.07,0.65), fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25,
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          ) 
        # Save figure
        fig_fn = ('mbcomponents_allregions_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
                  '-' + rcp + '.png')
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    
#%% ----- FIGURE: ANNUAL RUNOFF -----        
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        reg_runoff_annual_present_rcps_list = []
        for rcp in rcps:

            # Median and standard deviation
            reg_runoff_monthly = ds_multigcm_runoff[reg][rcp]
            reg_runoff_annual = np.zeros((reg_runoff_monthly.shape[0],101))
            for nrow in np.arange(0,reg_runoff_monthly.shape[0]):
                reg_runoff_annual[nrow,:] = reg_runoff_monthly[nrow,:].reshape(-1,12).sum(1)

            reg_runoff_annual_avg = np.median(reg_runoff_annual, axis=0)
            reg_runoff_annual_std = np.std(reg_runoff_annual, axis=0)

            reg_runoff_annual_present = reg_runoff_annual[:,0:16].mean(1)
            reg_runoff_norm = reg_runoff_annual / reg_runoff_annual_present[:,np.newaxis]
            reg_runoff_norm_avg = np.median(reg_runoff_norm, axis=0)
            reg_runoff_norm_var = np.std(reg_runoff_norm, axis=0)
            
            reg_runoff_annual_present_rcps_list.append(reg_runoff_annual_present.mean())
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(reg_runoff_annual_avg, years[0:-1], 11)
            
            ax.plot(years[0:-1], reg_runoff_norm_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            
            if 'ssp' in rcp:
                peakwater_yr_idx = np.where(peakwater_yr == years[0:-1])[0][0]
                ax.vlines(peakwater_yr, 0, reg_runoff_norm_avg[peakwater_yr_idx], 
                          color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                          linewidth=1, zorder=5)
                
            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
#                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                ax.fill_between(years[0:-1], 
                                (reg_runoff_norm_avg + reg_runoff_norm_var), 
                                (reg_runoff_norm_avg - reg_runoff_norm_var), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        runoff_gta_str = str(np.round(np.mean(reg_runoff_annual_present_rcps_list) * pygem_prms.density_water / 1e12,1))
        print(reg, runoff_gta_str, 'Gt/yr')
        ax.text(0.97, 0.97, runoff_gta_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes, zorder=6)

        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Runoff (-)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        if reg in [19, 3, 9, 7, 5]:
            ax.set_ylim(0,8)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))    
        else:
            ax.set_ylim(0,2.2)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2))
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('allregions_runoff_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: GLOBAL COMBINED -----
    add_rgi_glaciers = True
    add_rgi_regions = True
    
    class MidpointNormalize(mpl.colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # Note that I'm ignoring clipping and other edge cases here.
            result, is_scalar = self.process_value(value)
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)
        
#    rgi_reg_fig_dict = {'all':'Global',
#                        1:'Alaska (1)',
#                        2:'W Canada/USA (2)',
#                        3:'Arctic Canada\nNorth (3)',
#                        4:'Arctic Canada\nSouth (4)',
#                        5:'Greenland (5)',
#                        6:'Iceland (6)',
#                        7:'Svalbard (7)',
#                        8:'Scandinavia (8)',
#                        9:'Russian Arctic (9)',
#                        10:'North Asia (10)',
#                        11:'Central\nEurope (11)',
#                        12:'Caucasus\nMiddle East (12)',
#                        13:'Central Asia (13)',
#                        14:'South Asia\nWest (14)',
#                        15:'South Asia\nEast (15))',
#                        16:'Low Latitudes (16)',
#                        17:'Southern\nAndes (17)',
#                        18:'New Zealand (18)',
#                        19:'Antarctica/Subantarctic (19)'
#                        }
    rgi_reg_fig_dict = {'all':'Global',
                        1:'Alaska',
                        2:'W Canada/USA',
                        3:'Arctic Canada\nNorth',
                        4:'Arctic Canada\nSouth',
                        5:'Greenland',
                        6:'Iceland',
                        7:'Svalbard',
                        8:'Scandinavia',
                        9:'Russian Arctic',
                        10:'North Asia',
                        11:'Central\nEurope',
                        12:'Caucasus\nMiddle East',
                        13:'Central Asia',
                        14:'South Asia\nWest',
                        15:'South Asia\nEast',
                        16:'Low Latitudes',
                        17:'Southern\nAndes',
                        18:'New Zealand',
                        19:'Antarctica/Subantarctic'
                        }
    
    rcp_colordict = {'ssp119':'#76B8E5', 'ssp126':'#76B8E5', 'ssp245':'#F1EA8A', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
    rcp_namedict = {'ssp119':'SSP1-1.9',
                    'ssp126':'SSP1-2.6',
                    'ssp245':'SSP2-4.5',
                    'ssp370':'SSP3-7.0',
                    'ssp585':'SSP5-8.5'}
    
    pie_scenarios = ['ssp245']
    for pie_scenario in pie_scenarios:

        fig = plt.figure()
        ax_background = fig.add_axes([0,0.15,1,0.7], projection=ccrs.Robinson())
        ax_background.patch.set_facecolor('lightblue')
        ax_background.get_yaxis().set_visible(False)
        ax_background.get_xaxis().set_visible(False)
    #    ax_background.coastlines(color='white')
        ax_background.add_feature(cartopy.feature.LAND, color='white')
        
        ax_global_patch = fig.add_axes([0.08,0.145,0.19,0.38], facecolor='lightblue')
        ax_global_patch.get_yaxis().set_visible(False)
        ax_global_patch.get_xaxis().set_visible(False)
        
        # Add RGI glacier outlines
        if add_rgi_glaciers:
            shape_feature = ShapelyFeature(Reader(rgi_shp_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='indigo',linewidth=0.35,edgecolor='indigo')
            ax_background.add_feature(shape_feature)
            
        if add_rgi_regions:
            shape_feature = ShapelyFeature(Reader(rgi_regions_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='None',linewidth=0.35,edgecolor='k')
            ax_background.add_feature(shape_feature)
        
        regions_ordered = ['all',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        reg_pie_sizes = []
#        pie_scenario = 'ssp585'
        for reg in regions_ordered:
            reg_slr_cum_ssp585 = slr_cum_df.loc[(slr_cum_df.Region==reg) & (slr_cum_df.Scenario==pie_scenario),'med_mmSLE'].values[0]
            
            print(reg, np.round(reg_slr_cum_ssp585,2))
            
            if reg_slr_cum_ssp585 > 100:
                pie_size = 0.33
            elif reg_slr_cum_ssp585 > 25:
                pie_size = 0.2
            elif reg_slr_cum_ssp585 < 1:
                pie_size = 0.05
            else:
                pie_size = 0.05 + (reg_slr_cum_ssp585 - 1) / (25-1) * 0.15 
            reg_pie_sizes.append(pie_size)
        
        ax0 = fig.add_axes([0.125,0.18,0.1,0.04], facecolor='none')
        ax1 = fig.add_axes([0.09,0.73,0.1,0.04], facecolor='none')
        ax2 = fig.add_axes([0.1,0.59,0.1,0.04], facecolor='none')
        ax3 = fig.add_axes([0.21,0.875,0.1,0.04], facecolor='none')
        ax4 = fig.add_axes([0.265,0.60,0.1,0.04], facecolor='none')
        ax5 = fig.add_axes([0.33,0.875,0.1,0.04], facecolor='none')
        ax6 = fig.add_axes([0.37,0.64,0.1,0.04], facecolor='none')
        ax7 = fig.add_axes([0.44,0.875,0.1,0.04], facecolor='none')
        ax8 = fig.add_axes([0.55,0.875,0.1,0.04], facecolor='none')
        ax9 = fig.add_axes([0.68,0.875,0.1,0.04], facecolor='none')
        ax10 = fig.add_axes([0.8,0.78,0.1,0.04], facecolor='none')
        ax11 = fig.add_axes([0.425,0.55,0.1,0.04], facecolor='none')
        ax12 = fig.add_axes([0.53,0.5,0.1,0.04], facecolor='none')
        ax13 = fig.add_axes([0.8,0.62,0.1,0.04], facecolor='none')
        ax14 = fig.add_axes([0.65,0.43,0.1,0.04], facecolor='none')
        ax15 = fig.add_axes([0.77,0.47,0.1,0.04], facecolor='none')
        ax16 = fig.add_axes([0.435,0.40,0.1,0.04], facecolor='none')
        ax17 = fig.add_axes([0.36,0.295,0.1,0.04], facecolor='none')
        ax18 = fig.add_axes([0.715,0.3,0.1,0.04], facecolor='none')
        ax19 = fig.add_axes([0.55,0.19,0.1,0.04], facecolor='none')

        # Pie charts
        ax0b = fig.add_axes([0.01,0.205,reg_pie_sizes[0],reg_pie_sizes[0]], facecolor='none')
        ax1b = fig.add_axes([0.045,0.76,reg_pie_sizes[1],reg_pie_sizes[1]], facecolor='none')
        ax2b = fig.add_axes([0.12,0.63,reg_pie_sizes[2],reg_pie_sizes[2]], facecolor='none')
        ax3b = fig.add_axes([0.20,0.91,reg_pie_sizes[3],reg_pie_sizes[3]], facecolor='none')
        ax4b = fig.add_axes([0.26,0.635,reg_pie_sizes[4],reg_pie_sizes[4]], facecolor='none')
        ax5b = fig.add_axes([0.31,0.91,reg_pie_sizes[5],reg_pie_sizes[5]], facecolor='none')
        ax6b = fig.add_axes([0.385,0.678,reg_pie_sizes[6],reg_pie_sizes[6]], facecolor='none')
        ax7b = fig.add_axes([0.445,0.91,reg_pie_sizes[7],reg_pie_sizes[7]], facecolor='none')
        ax8b = fig.add_axes([0.573,0.912,reg_pie_sizes[8],reg_pie_sizes[8]], facecolor='none')
        ax9b = fig.add_axes([0.68,0.91,reg_pie_sizes[9],reg_pie_sizes[9]], facecolor='none')
        ax10b = fig.add_axes([0.823,0.817,reg_pie_sizes[10],reg_pie_sizes[10]], facecolor='none')
        ax11b = fig.add_axes([0.45,0.587,reg_pie_sizes[11],reg_pie_sizes[11]], facecolor='none')
        ax12b = fig.add_axes([0.551,0.538,reg_pie_sizes[12],reg_pie_sizes[12]], facecolor='none')
        ax13b = fig.add_axes([0.81,0.656,reg_pie_sizes[13],reg_pie_sizes[13]], facecolor='none')
        ax14b = fig.add_axes([0.663,0.467,reg_pie_sizes[14],reg_pie_sizes[14]], facecolor='none')
        ax15b = fig.add_axes([0.793,0.508,reg_pie_sizes[15],reg_pie_sizes[15]], facecolor='none')
        ax16b = fig.add_axes([0.46,0.438,reg_pie_sizes[16],reg_pie_sizes[16]], facecolor='none')
        ax17b = fig.add_axes([0.37,0.331,reg_pie_sizes[17],reg_pie_sizes[17]], facecolor='none')
        ax18b = fig.add_axes([0.74,0.339,reg_pie_sizes[18],reg_pie_sizes[18]], facecolor='none')
        ax19b = fig.add_axes([0.535,0.224,reg_pie_sizes[19],reg_pie_sizes[19]], facecolor='none')
        
        # ----- Heat map of specific mass balance (2015 - 2100) -----
        for nax, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10,
                                  ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19]):
            
            reg = regions_ordered[nax]
            
    #        cmap = 'RdYlBu'
            cmap = 'Greys_r'
    #        cmap = 'YlOrRd'
            norm_values = [-2.5,-1.5,-0.25]
            norm = MidpointNormalize(midpoint=norm_values[1], vmin=norm_values[0], vmax=norm_values[2])
            
            mesh = None
            for rcp in ['ssp245', 'ssp585']:
            
                # Median and absolute median deviation
                reg_vol = ds_multigcm_vol[reg][rcp]
                reg_mass = reg_vol * pygem_prms.density_ice
                reg_area = ds_multigcm_area[reg][rcp]
        
                # Specific mass change rate
                reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
                reg_mb_med = np.median(reg_mb, axis=0)
                reg_mb_mad = median_abs_deviation(reg_mb, axis=0)
                
                if mesh is None:
                    mesh = reg_mb_med[np.newaxis,normyear_idx:]
                else:
                    mesh = np.concatenate((mesh, reg_mb_med[np.newaxis,normyear_idx:]), axis=0)
                
            ax.imshow(mesh/1000, aspect='auto', cmap=cmap, norm=norm, interpolation='none')
            ax.hlines(0.5,0,mesh.shape[1]-1, color='k', linewidth=0.5, zorder=2)
            ax.get_yaxis().set_visible(False)
    #        ax.get_xaxis().set_visible(False)
            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.tick_params(axis='both', which='major', direction='inout', right=True, top=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True, top=True)
            ax.get_xaxis().set_ticks([])
            
            # Add region label
#            if ax in [ax0]:
#                ax.text(0.5, -0.14, rgi_reg_fig_dict[reg], size=8, horizontalalignment='center', 
#                        verticalalignment='top', transform=ax.transAxes)
#            else:
            ax.text(0.5, -0.14, rgi_reg_fig_dict[reg], size=8, horizontalalignment='center', 
                verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=2))
            
        # ----- Pie Chart of Volume Remaining by end of century -----
        wedge_size = 0.15
        for nax, ax in enumerate([ax0b, ax1b, ax2b, ax3b, ax4b, ax5b, ax6b, ax7b, ax8b, ax9b, ax10b,
                                  ax11b, ax12b, ax13b, ax14b, ax15b, ax16b, ax17b, ax18b, ax19b]):
            
            reg = regions_ordered[nax]
            
            ssp_vol_remaining_pies = []
            ssp_pie_radius = 1
            for nrcp, rcp in enumerate(['ssp585', 'ssp370', 'ssp245', 'ssp126']):
                
                # Median and absolute median deviation
                reg_vol = ds_multigcm_vol[reg][rcp]
                reg_vol_med = np.median(reg_vol, axis=0)
                reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
                
                reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
                reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
    
                ssp_vol_remaining_pies.append(reg_vol_med_norm[-1])
                
            # Single Pie Chart
    #        vol_lost = 1-np.min(ssp_vol_remaining_pies[0])
    #        ssp_pies = [1 - ssp_vol_remaining_pies[3],
    #                    ssp_vol_remaining_pies[3] - ssp_vol_remaining_pies[2],
    #                    ssp_vol_remaining_pies[2] - ssp_vol_remaining_pies[1],
    #                    ssp_vol_remaining_pies[1] - ssp_vol_remaining_pies[0],
    #                    ssp_vol_remaining_pies[0]]
    #        ssp_pie_colors = ['lightgray',rcp_colordict['ssp126'],rcp_colordict['ssp245'],
    #                          rcp_colordict['ssp370'], rcp_colordict['ssp585']]
    #        pie_slices, pie_labels = ax.pie(ssp_pies, counterclock=False, startangle=90, colors=ssp_pie_colors)
                # Nested Pie Charts
                ssp_pies = [1-ssp_vol_remaining_pies[nrcp], ssp_vol_remaining_pies[nrcp]]
                ssp_pie_colors = ['lightgray',rcp_colordict[rcp]]
                pie_slices, pie_labels = ax.pie(ssp_pies, radius=ssp_pie_radius, 
                                                counterclock=False, startangle=90, colors=ssp_pie_colors,
                                                wedgeprops=dict(width=wedge_size))
                ssp_pie_radius = ssp_pie_radius - wedge_size
            ssp_pie_radius_fill = 1 - wedge_size*len(['ssp585', 'ssp370', 'ssp245', 'ssp126'])
            wedge_size_fill = ssp_pie_radius_fill
            ssp_pies, ssp_pie_colors = [1], ['lightgray']
            pie_slices, pie_labels = ax.pie(ssp_pies, radius=ssp_pie_radius_fill, 
                                            counterclock=False, startangle=90, colors=ssp_pie_colors,
                                            wedgeprops=dict(width=wedge_size_fill))
            ax.axis('equal')
            
            # SLR 
            reg_slr_cum_ssp585 = slr_cum_df.loc[(slr_cum_df.Region==reg) & (slr_cum_df.Scenario==pie_scenario),'med_mmSLE'].values[0]
            if reg_slr_cum_ssp585 > 1:
                reg_slr_str = str(int(np.round(reg_slr_cum_ssp585)))
            else:
                reg_slr_str = ''
            if reg in ['all']:
                reg_slr_str += '\nmm SLE'
            ax.text(0.5, 0.5, reg_slr_str, size=10, color='k', horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes)
            
            # Add outer edge by adding new circle with desired properties
            center = pie_slices[0].center
            r = 1
            circle = mpl.patches.Circle(center, r, fill=False, edgecolor="k", linewidth=1)
            ax.add_patch(circle)
            
        # ----- LEGEND -----
        # Sized circles
#        ax_background.text(0.66,-0.2,'Sea level rise from\n2015-2100 for ' + rcp_namedict[pie_scenario] + '\n(mm SLE)', size=10, 
#                           horizontalalignment='center', verticalalignment='top', transform=ax_background.transAxes)
#        ax_circle1 = fig.add_axes([0.56,0.06,0.05,0.05], facecolor='none')
#        pie_slices, pie_labels = ax_circle1.pie([1], counterclock=False, startangle=90, colors=['white'],
#                                                wedgeprops=dict(edgecolor='k', linewidth=0.5))
#        ax_circle1.axis('equal')
#        ax_circle1.text(0.5,0.5,'1', size=8, horizontalalignment='center',  verticalalignment='center',
#                        transform=ax_circle1.transAxes)
#        
#        ax_circle2 = fig.add_axes([0.61,0.01,0.10625,0.10625], facecolor='none')
#        pie_slices, pie_labels = ax_circle2.pie([1], counterclock=False, startangle=90, colors=['white'],
#                                                wedgeprops=dict(edgecolor='k', linewidth=0.5))
#        ax_circle2.axis('equal')
#        ax_circle2.text(0.5,0.5,'10', size=8, horizontalalignment='center', verticalalignment='center', 
#                        transform=ax_circle2.transAxes)
    
        ax_circle3 = fig.add_axes([0.6,-0.02,0.13,0.13], facecolor='none')
#        ax_circle3 = fig.add_axes([0.68,-0.08,0.2,0.2], facecolor='none')
#        ax_circle3.text(0.5,0.5,'25', size=8, horizontalalignment='center', verticalalignment='center', transform=ax_circle3.transAxes)
        ssp_vol_remaining_pies = [0.6, 0.65, 0.7, 0.75]
        ssp_pie_radius = 1
        for nrcp, rcp in enumerate(['ssp585', 'ssp370', 'ssp245', 'ssp126']):
            # Nested Pie Charts
            ssp_pies = [1-ssp_vol_remaining_pies[nrcp], ssp_vol_remaining_pies[nrcp]]
            ssp_pie_colors = ['lightgray',rcp_colordict[rcp]]
            pie_slices, pie_labels = ax_circle3.pie(ssp_pies, radius=ssp_pie_radius, 
                                                    counterclock=False, startangle=90, colors=ssp_pie_colors,
                                                    wedgeprops=dict(width=wedge_size))
            ssp_pie_radius = ssp_pie_radius - wedge_size
        ssp_pie_radius_fill = 1 - wedge_size*len(['ssp585', 'ssp370', 'ssp245', 'ssp126'])
        wedge_size_fill = ssp_pie_radius_fill
        ssp_pies, ssp_pie_colors = [1], ['lightgray']
        pie_slices, pie_labels = ax_circle3.pie(ssp_pies, radius=ssp_pie_radius_fill, 
                                                counterclock=False, startangle=90, colors=ssp_pie_colors,
                                                wedgeprops=dict(width=wedge_size_fill))
        ax_circle3.axis('equal')
    
    #    ax_circle3 = fig.add_axes([0.75,0.01,0.2,0.2], facecolor='none')
    #    pie_slices, pie_labels = ax_circle3.pie([0.25,0.05,0.05,0.05,0.6], counterclock=False, startangle=90, colors=ssp_pie_colors)
        center = pie_slices[0].center
        r = 1
        circle = mpl.patches.Circle(center, r, fill=False, edgecolor="k", linewidth=1)
        ax_circle3.add_patch(circle)
        ax_circle3.text(0.79,0.4,'SSP1-2.6', color=rcp_colordict['ssp126'], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_circle3.text(0.76,0.25,'SSP2-4.5', color=rcp_colordict['ssp245'], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_circle3.text(0.72,0.1,'SSP3-7.0', color=rcp_colordict['ssp370'], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_circle3.text(0.62,-0.07,'SSP5-8.5', color=rcp_colordict['ssp585'], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_background.text(0.71,-0.05,'Mass at 2100 (rel. to 2015)', size=10, 
                           horizontalalignment='center', transform=ax_background.transAxes)
    
        # Heat maps
        ax_background.text(0.32,-0.05,'Annual mass balance (m w.e.)', size=10, 
                           horizontalalignment='center', transform=ax_background.transAxes)
        ax_heatmap = fig.add_axes([0.185,0.03,0.1,0.06], facecolor='none')
        ax_heatmap.hlines(0.5,2015,2100, color='k', linewidth=0.5, zorder=2)
        ax_heatmap.set_ylim(0,1)
        ax_heatmap.set_xlim(2015,2100)
        ax_heatmap.get_yaxis().set_visible(False)
        ax_heatmap.xaxis.set_major_locator(MultipleLocator(40))
        ax_heatmap.xaxis.set_minor_locator(MultipleLocator(10))
        ax_heatmap.tick_params(axis='both', which='major', direction='inout', right=True, top=True)
        ax_heatmap.tick_params(axis='both', which='minor', direction='in', right=True, top=True)
        ax_heatmap.text(0.5,0.71,'SSP2-4.5', size=8, 
                        horizontalalignment='center', verticalalignment='center', transform=ax_heatmap.transAxes)
        ax_heatmap.text(0.5,0.21,'SSP5-8.5', size=8, 
                        horizontalalignment='center', verticalalignment='center', transform=ax_heatmap.transAxes)
        
        # Heat map colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax = plt.axes([0.3, 0.06, 0.22, 0.015])
        cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='horizontal', extend='both')
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=2)
        cbar.ax.tick_params(labelsize=8)
        
        labels = []
        for n,label in enumerate(cax.xaxis.get_ticklabels()):
            label_str = str(label.get_text())
            labels.append(label_str.split('.')[0] + '.' + label_str.split('.')[1][0])
        cbar.ax.set_xticklabels(labels)

        for n, label in enumerate(cax.xaxis.get_ticklabels()):
            print(n, label)
            if n%2 != 0:
                label.set_visible(False)
#        ax_background.text(0.5, -0.12, 'Mass balance (m w.e. yr$^{-1}$)', size=10, horizontalalignment='center', 
#                           verticalalignment='center', transform=ax_background.transAxes)

        # Save figure
        fig_fn = ('map_regional_mb_and_volremain_' + pie_scenario + '.png')
        fig.set_size_inches(8.5,5)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: GLOBAL COMBINED -----
#    class MidpointNormalize(mpl.colors.Normalize):
#        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#            self.midpoint = midpoint
#            mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
#    
#        def __call__(self, value, clip=None):
#            # Note that I'm ignoring clipping and other edge cases here.
#            result, is_scalar = self.process_value(value)
#            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#            return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)
#        
##    rgi_reg_fig_dict = {'all':'Global',
##                        1:'Alaska (1)',
##                        2:'W Canada/USA (2)',
##                        3:'Arctic Canada\nNorth (3)',
##                        4:'Arctic Canada\nSouth (4)',
##                        5:'Greenland (5)',
##                        6:'Iceland (6)',
##                        7:'Svalbard (7)',
##                        8:'Scandinavia (8)',
##                        9:'Russian Arctic (9)',
##                        10:'North Asia (10)',
##                        11:'Central\nEurope (11)',
##                        12:'Caucasus\nMiddle East (12)',
##                        13:'Central Asia (13)',
##                        14:'South Asia\nWest (14)',
##                        15:'South Asia\nEast (15))',
##                        16:'Low Latitudes (16)',
##                        17:'Southern\nAndes (17)',
##                        18:'New Zealand (18)',
##                        19:'Antarctica/Subantarctic (19)'
##                        }
#
#    fig = plt.figure()
#    ax_background = fig.add_axes([0.35,0.25,0.3,0.25], projection=ccrs.Robinson())
#    ax_background.patch.set_facecolor('lightblue')
#    ax_background.get_yaxis().set_visible(False)
#    ax_background.get_xaxis().set_visible(False)
##    ax_background.coastlines(color='white')
##    ax_background.add_feature(cartopy.feature.LAND, color='white')
#    ax_background.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='gainsboro'))
#    ax_background.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', facecolor='dimgrey'))
#    
##    regions_ordered = ['all',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
##    for reg in regions_ordered:
#    
#    ax_ak = fig.add_axes([0.1,0.27,0.25,0.5], facecolor='none', projection=ccrs.Robinson())
#    ax_ak.set_extent([-175,-110,36,70], ccrs.Geodetic())
#    ax_hma = fig.add_axes([0.62,0.27,0.25,0.25], facecolor='none', projection=ccrs.Robinson())
#    ax_hma.set_extent([70,105,20,50], ccrs.Geodetic())
#    ax_arctic = fig.add_axes([0.27,0.75,0.7,0.25], facecolor='none', projection=ccrs.Robinson(),
#                             extent=[-90,120,54,90])
#    ax_europe = fig.add_axes([0.33,0.52,0.2,0.25], facecolor='none', projection=ccrs.Robinson(),
#                             extent=[-2,15,41,53])
#    ax_caucasus = fig.add_axes([0.54,0.52,0.15,0.25], facecolor='none', projection=ccrs.Robinson(),
#                               extent=[35,53,30,44])
#    ax_nz = fig.add_axes([0.88,0.27,0.12,0.25], facecolor='none', projection=ccrs.Robinson(),
#                         extent=[166,175,-47.5,-40])
#    for ax in [ax_ak, ax_hma, ax_arctic, ax_europe, ax_caucasus, ax_nz]:
#        ax.patch.set_facecolor('lightblue')
#        ax.get_yaxis().set_visible(False)
#        ax.get_xaxis().set_visible(False)
#        ax.add_feature(cartopy.feature.LAND, color='white')
#        ax.add_feature(cartopy.feature.BORDERS, linestyle=':', color='k', linewidth=0.5)
#        
#    # Save figure
#    fig_fn = ('map_subregional_test.png')
#    fig.set_size_inches(8.5,5)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    

#%% ===== MULTI-GCM PLOTS OF RUNOFF BY WATERSHED =====
if option_multigcm_plots_ws:
    
    fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
    pickle_fp = fig_fp + '../pickle/'
    
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
#    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    
    # Watersheds
#    ws_reg = 'AK'
#    ws_reg = 'HMA'
    ws_reg = 'Euro_Asia'
#    ws_reg = 'SA'
    # North America / Alaska watersheds
    if ws_reg == 'AK':
        watersheds = ['Yukon', 'Kuskowin', 'Nushagak', 'GHAASBasin325', 'Copper', 'Alsek', 'Mackenzie', 
                      'Taku', 'Stikine', 'Nass','Skeena', 'Fraser', 'Columbia', 'Nelson']
        regions = [1,2]
    # Europe and North Asia watersheds
    elif ws_reg == 'Euro_Asia':
#        watersheds = ['Danube', 'Po', 'Rhone', 'Rhine', 'Glama', 'Drammenselva', 'Lulealven', 'Kalix', 'Ob', 'Thjorsa', 'Olfusa']
#        regions = [6,8,10,11]
        watersheds = ['Danube', 'Po', 'Rhone', 'Rhine']
        regions = [11]
    # High Mountain Asia watersheds
    elif ws_reg == 'HMA':
        watersheds = ['Brahmaputra', 'Ganges', 'Indus', 'Amu_Darya', 'Syr_Darya', 'Ili', 'Tarim', 'Inner_Tibetan_Plateau', 
                      'Inner_Tibetan_Plateau_extended', 'Mekong', 'Salween',  'Yangtze', 'Yellow', 'Irrawaddy']
        regions = [13,14,15]
    # South America watersheds
    elif ws_reg == 'SA':
        watersheds = ['Amazon', 'Santa', 'Baker (Chile)', 'Colorado (Argentina)', 'Rapel', 'Bio Bio']
        regions = [16,17]
        print('\n\nNEED TO MANUALLY ADD MORE WATERSHEDS!\n\n')
    watersheds_dict_correct = {'GHAASBasin325':'Susitna',
                               'Amu_Darya':'Amu Darya', 
                               'Syr_Darya':'Syr Darya', 
                               'Inner_Tibetan_Plateau':'Inner TP', 
                               'Inner_Tibetan_Plateau_extended':'Inner TP ext'}
    
#    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_multigcm = fig_fp + 'multi_gcm_ws/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
    
    # Set up processing
    ws_runoff_all = {}
    for ws in watersheds:
        ws_runoff_all[ws] = {}
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
            ws_runoff_all[ws][rcp] = {}
            for gcm_name in gcm_names:
                ws_runoff_all[ws][rcp][gcm_name] = None
            
    for reg in regions:
        
        fnfull_watersheds = pickle_fp + str(reg).zfill(2) + '/R' + str(reg) + '_unique_watersheds.pkl'
        with open(fnfull_watersheds, 'rb') as f:
            watersheds_list = list(pickle.load(f))
        
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
        
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'       

                # Runoff: moving-gauged
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                    ws_runoff_monthly_moving = pickle.load(f)
                # Runoff: fixed-gauged
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                    ws_runoff_monthly_fixed = pickle.load(f)
                    
                for ws in watersheds:
                    if ws in watersheds_list:
                        ws_idx = watersheds_list.index(ws)
                        if ws_runoff_all[ws][rcp][gcm_name] is None:
                            ws_runoff_all[ws][rcp][gcm_name] = ws_runoff_monthly_fixed[ws_idx,:]
                        else:
                            ws_runoff_all[ws][rcp][gcm_name] = ws_runoff_all[ws][rcp][gcm_name] + ws_runoff_monthly_fixed[ws_idx,:]

    #%% MULTI-GCM STATISTICS
    ds_multigcm_runoff = {}
    for ws in watersheds:
        ds_multigcm_runoff[ws] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                ws_runoff_gcm = ws_runoff_all[ws][rcp][gcm_name]
    
                if ngcm == 0:
                    ws_runoff_gcm_all = ws_runoff_gcm
                else:
                    ws_runoff_gcm_all = np.vstack((ws_runoff_gcm_all, ws_runoff_gcm))

            ds_multigcm_runoff[ws][rcp] = ws_runoff_gcm_all

    #%% ----- FIGURE: ANNUAL RUNOFF -----
    # Peakwater
    pw_cns = ['Watershed', 'Scenario', 'runoff_Gtyr_ref', 'peakwater_yr', 'peakwater_chg_perc', '2100_chg_perc']
    pw_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(pw_cns))), columns=pw_cns)
    ncount = 0
    
    for nws, ws in enumerate(watersheds):
        for rcp in rcps:

            # Median and standard deviation
            ws_runoff_monthly = ds_multigcm_runoff[ws][rcp]
            ws_runoff_annual = np.zeros((ws_runoff_monthly.shape[0],101))
            for nrow_ws in np.arange(0,ws_runoff_monthly.shape[0]):
                ws_runoff_annual[nrow_ws,:] = ws_runoff_monthly[nrow_ws,:].reshape(-1,12).sum(1)

            ws_runoff_annual_avg = np.median(ws_runoff_annual, axis=0)
            ws_runoff_annual_std = np.std(ws_runoff_annual, axis=0)
            ws_runoff_annual_present = ws_runoff_annual[:,0:16].mean(1)
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_runoff_annual_avg, years[0:-1], 11)
            
            pw_df.loc[ncount,'Watershed'] = ws
            pw_df.loc[ncount,'Scenario'] = rcp
            pw_df.loc[ncount,'runoff_Gtyr_ref'] = ws_runoff_annual_present.mean()
            pw_df.loc[ncount,'peakwater_yr'] = peakwater_yr
            pw_df.loc[ncount,'peakwater_chg_perc'] = peakwater_chg
            pw_df.loc[ncount,'2100_chg_perc'] = runoff_chg
            
            ncount += 1
    pw_df.to_csv(csv_fp + 'Runoff_' + ws_reg + '_watersheds_' + str(startyear) + '-' + str(endyear) + '_multigcm.csv', index=False)
            
    #%%
    ncols = 4
    nrows = int(np.ceil(len(watersheds)/ncols))
    fig, ax = plt.subplots(ncols,nrows,gridspec_kw = {'wspace':0.3, 'hspace':0.3})
    
    nrow, ncol = 0,0
    for nws, ws in enumerate(watersheds):
        
        print(ws)
        
        ws_runoff_annual_present_rcps_list = []
        for rcp in rcps:

            # Median and standard deviation
            ws_runoff_monthly = ds_multigcm_runoff[ws][rcp]
            ws_runoff_annual = np.zeros((ws_runoff_monthly.shape[0],101))
            for nrow_ws in np.arange(0,ws_runoff_monthly.shape[0]):
                ws_runoff_annual[nrow_ws,:] = ws_runoff_monthly[nrow_ws,:].reshape(-1,12).sum(1)

            ws_runoff_annual_avg = np.median(ws_runoff_annual, axis=0)
            ws_runoff_annual_std = np.std(ws_runoff_annual, axis=0)

            ws_runoff_annual_present = ws_runoff_annual[:,0:16].mean(1)
            ws_runoff_norm = ws_runoff_annual / ws_runoff_annual_present[:,np.newaxis]
            ws_runoff_norm_avg = np.median(ws_runoff_norm, axis=0)
            ws_runoff_norm_var = np.std(ws_runoff_norm, axis=0)
            
            ws_runoff_annual_present_rcps_list.append(ws_runoff_annual_present.mean())
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_runoff_annual_avg, years[0:-1], 11)
            
            ax[nrow,ncol].plot(years[0:-1], ws_runoff_norm_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                                linewidth=1, zorder=4, label=rcp)
            
            if 'ssp' in rcp:
                peakwater_yr_idx = np.where(peakwater_yr == years[0:-1])[0][0]
                ax[nrow,ncol].vlines(peakwater_yr, 0, ws_runoff_norm_avg[peakwater_yr_idx], 
                                     color=rcp_colordict[rcp], linestyle='--', 
                                     linewidth=1, zorder=5)
                
            if rcp in rcps_plot_mad:
                ax[nrow,ncol].fill_between(years[0:-1], 
                                           (ws_runoff_norm_avg + ws_runoff_norm_var), 
                                           (ws_runoff_norm_avg - ws_runoff_norm_var), 
                                           alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        runoff_gta_str = str(np.round(np.mean(ws_runoff_annual_present_rcps_list) * pygem_prms.density_water / 1e12,1))
        print(ws, runoff_gta_str, 'Gt/yr')
        ax[nrow,ncol].text(0.97, 0.97, runoff_gta_str, size=10, horizontalalignment='right', 
                           verticalalignment='top', transform=ax[nrow,ncol].transAxes, zorder=6)

        if ncol == 0:
            ax[nrow,ncol].set_ylabel('Runoff (-)')
        ax[nrow,ncol].set_xlim(startyear, endyear)
        ax[nrow,ncol].xaxis.set_major_locator(MultipleLocator(40))
        ax[nrow,ncol].xaxis.set_minor_locator(MultipleLocator(10))


        ax[nrow,ncol].set_ylim(0,3)
        ax[nrow,ncol].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[nrow,ncol].yaxis.set_minor_locator(MultipleLocator(0.1))
            
        if ws in watersheds_dict_correct:
            ws_label = watersheds_dict_correct[ws]
        else:
            ws_label = ws
        ax[nrow,ncol].text(1, 1.1, ws_label, size=10, horizontalalignment='right', 
                           verticalalignment='top', transform=ax[nrow,ncol].transAxes)
        ax[nrow,ncol].tick_params(axis='both', which='major', direction='inout', right=True)
        ax[nrow,ncol].tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nrow == 0 and ncol == ncols-1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            ax[nrow,ncol].legend(loc=(1.1,0.3), labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                                 handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                                 )

        # Adjust row and column
        ncol += 1
        if ncol == ncols:
            nrow += 1
            ncol = 0
    
    for batman in np.arange(nrows*ncols-len(watersheds)):
        ax[nrow,ncol].get_yaxis().set_visible(False)
        ax[nrow,ncol].get_xaxis().set_visible(False)
        ax[nrow,ncol].axis('off')
        ncol += 1
        
    
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = (ws_reg + '_watersheds_runoff_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- MONTHLY RUNOFF CHANGE FIGURE -----
#    ncols = 4
#    nrows = int(np.ceil(len(watersheds)/ncols))
#    fig, ax = plt.subplots(nrows,ncols,gridspec_kw = {'wspace':0.3, 'hspace':0.3})
#    
#    nrow, ncol = 0,0
#    for nws, ws in enumerate(watersheds):
#        
#        print(ws)
#        
#        ws_runoff_annual_present_rcps_list = []
#        for rcp in rcps:
#
#            # Median and standard deviation
#            ws_runoff_monthly = ds_multigcm_runoff[ws][rcp]
#            ws_runoff_annual = np.zeros((ws_runoff_monthly.shape[0],101))
#            for nrow_ws in np.arange(0,ws_runoff_monthly.shape[0]):
#                ws_runoff_annual[nrow_ws,:] = ws_runoff_monthly[nrow_ws,:].reshape(-1,12).sum(1)
#
#            ws_runoff_annual_avg = np.median(ws_runoff_annual, axis=0)
#            ws_runoff_annual_std = np.std(ws_runoff_annual, axis=0)
#
#            ws_runoff_annual_present = ws_runoff_annual[:,0:16].mean(1)
#            ws_runoff_norm = ws_runoff_annual / ws_runoff_annual_present[:,np.newaxis]
#            ws_runoff_norm_avg = np.median(ws_runoff_norm, axis=0)
#            ws_runoff_norm_var = np.std(ws_runoff_norm, axis=0)
#            
#            ws_runoff_annual_present_rcps_list.append(ws_runoff_annual_present.mean())
#            
#            # Peakwater
#            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_runoff_annual_avg, years[0:-1], 11)
#            
#            ax[nrow,ncol].plot(years[0:-1], ws_runoff_norm_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
#                                linewidth=1, zorder=4, label=rcp)
#            
#            if 'ssp' in rcp:
#                peakwater_yr_idx = np.where(peakwater_yr == years[0:-1])[0][0]
#                ax[nrow,ncol].vlines(peakwater_yr, 0, ws_runoff_norm_avg[peakwater_yr_idx], 
#                                     color=rcp_colordict[rcp], linestyle='--', 
#                                     linewidth=1, zorder=5)
#                
#            if rcp in rcps_plot_mad:
#                ax[nrow,ncol].fill_between(years[0:-1], 
#                                           (ws_runoff_norm_avg + ws_runoff_norm_var), 
#                                           (ws_runoff_norm_avg - ws_runoff_norm_var), 
#                                           alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
#        
#        runoff_gta_str = str(np.round(np.mean(ws_runoff_annual_present_rcps_list) * pygem_prms.density_water / 1e12,1))
#        print(ws, runoff_gta_str, 'Gt/yr')
#        ax[nrow,ncol].text(0.97, 0.97, runoff_gta_str, size=10, horizontalalignment='right', 
#                           verticalalignment='top', transform=ax[nrow,ncol].transAxes, zorder=6)
#
#        if ncol == 0:
#            ax[nrow,ncol].set_ylabel('Runoff (-)')
#        ax[nrow,ncol].set_xlim(startyear, endyear)
#        ax[nrow,ncol].xaxis.set_major_locator(MultipleLocator(40))
#        ax[nrow,ncol].xaxis.set_minor_locator(MultipleLocator(10))
#
#
#        ax[nrow,ncol].set_ylim(0,3)
#        ax[nrow,ncol].yaxis.set_major_locator(MultipleLocator(0.5))
#        ax[nrow,ncol].yaxis.set_minor_locator(MultipleLocator(0.1))
#            
#        if ws in watersheds_dict_correct:
#            ws_label = watersheds_dict_correct[ws]
#        else:
#            ws_label = ws
#        ax[nrow,ncol].text(1, 1.1, ws_label, size=10, horizontalalignment='right', 
#                           verticalalignment='top', transform=ax[nrow,ncol].transAxes)
#        ax[nrow,ncol].tick_params(axis='both', which='major', direction='inout', right=True)
#        ax[nrow,ncol].tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        if nrow == 0 and ncol == ncols-1:
#            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'rcp26' in rcps and len(rcps) == 3:
#                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'ssp126' in rcp and len(rcps) == 4:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#            ax[nrow,ncol].legend(loc=(1.1,0.3), labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
#                                 handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
#                                 )
#
#        # Adjust row and column
#        ncol += 1
#        if ncol == ncols:
#            nrow += 1
#            ncol = 0
#    
#    for batman in np.arange(nrows*ncols-len(watersheds)):
#        ax[nrow,ncol].get_yaxis().set_visible(False)
#        ax[nrow,ncol].get_xaxis().set_visible(False)
#        ax[nrow,ncol].axis('off')
#        ncol += 1
#        
#    
#    # Save figure
#    if 'rcp26' in rcps and 'ssp126' in rcps:
#        scenario_str = 'rcps_ssps'
#    elif 'rcp26' in rcps:
#        scenario_str = 'rcps'
#    elif 'ssp126' in rcp:
#        scenario_str = 'ssps'
#    fig_fn = (ws_reg + '_watersheds_runoff_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
#              '-' + scenario_str + '-medstd.png')
#    fig.set_size_inches(8.5,11)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

#%% ---------------------------------------------------------------------------------------------------------
if option_glacier_cs_plots:
#    glac_nos = ['1.22193', '2.14297', '11.02739', '11.03005', '11.03643', '12.00080', 
#                '14.06794', '15.03733', '17.05076', '17.14140', '18.02342']
#    glac_nos = ['1.10689', '7.00025', '7.00238']
#    glac_nos = ['1.10689']
#    glac_nos = ['7.00025']
#    glac_nos = ['7.00027']
#    glac_nos = ['7.00238']
#    glac_nos = ['7.00240']
#    glac_nos = ['7.00293']
#    glac_nos = ['7.00892']
##    glac_nos = ['7.00893']
#    glac_nos = ['15.03733']
#    glac_nos = ['11.03005']
    glac_nos = ['19.00001']
    
    
    cleanice_fns = False
    debris_fns = False
    
    # Clean ice filenames
    if cleanice_fns:
        for glac_no in glac_nos:
            if int(glac_no.split('.')[0]) in [1,3,4,5,7,9,17,19]: 
                netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_calving/' # treated as clean ice
                fig_fp_multigcm = netcdf_fp_cmip5 + '/../../analysis/figures/ind_glaciers/'
                fa_label = ''
            else:
                netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/Output/simulations/' # treated as clean ice
                fig_fp_multigcm = netcdf_fp_cmip5 + '/../analysis/figures/ind_glaciers/'
                fa_label = ''
    elif debris_fns:
        netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind/' # treated as clean ice
        fig_fp_multigcm = netcdf_fp_cmip5 + '/../analysis/figures/ind_glaciers/'
        fa_label = ''
    else:
        # Calving filenames
        netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving/' # including calving
        fig_fp_multigcm = netcdf_fp_cmip5 + '/../analysis_calving/figures/ind_glaciers/'
        fa_label = 'Frontal ablation included'
    
    # Other filenames
#    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind/'
#    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind_calving/' # including calving
#    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-nodebris/'
#    fig_fp_multigcm = netcdf_fp_cmip5 + '/../analysis/figures/ind_glaciers/'
    
    cs_year = 2000
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    startyear_idx = np.where(years == startyear)[0][0]
    cs_idx = np.where(years == cs_year)[0][0]

    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
#    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps_plot_mad = ['ssp126', 'ssp585']
    
      
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
        
    glac_name_dict = {'1.10689':'Columbia',
                      '1.22193':'Kahiltna (Denali)',
                      '2.14297':'Emmons (Rainier)',
                      '7.00027':'Basin-3 of Austfonna Ice Cap',
                      '7.00238': 'Storbreen',
                      '11.02739':'Zmuttgletscher (Matterhorn)',
                      '11.03005':'Miage (Mont Blanc)',
                      '11.03643':'Mer de Glace (Mont Blanc)',
                      '12.00080':'Bolshoy Azau (Elbrus)',
                      '14.06794':'Baltoro (K2)',
                      '15.03733':'Khumbu (Everest)',
                      '17.05076':'Viedma (Fitz Roy)',
                      '17.14140':'Horcones Inferior (Aconcagua)',
                      '18.02342':'Tasman (Aoraki)'}
        

    # Set up processing
    glac_zbed_all = {}
    glac_thick_all = {}
    glac_zsurf_all = {}
    glac_vol_all = {}
    glac_multigcm_zbed = {}
    glac_multigcm_thick = {}
    glac_multigcm_zsurf = {}
    glac_multigcm_vol = {}
    for glac_no in glac_nos:

        gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
        
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        nfls = gdir.read_pickle('model_flowlines')
        
        x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx

        glac_idx = np.nonzero(nfls[0].thick)[0]
        xmax = np.ceil(x[glac_idx].max()/1000+0.5)*1000
        
#        vol_m3_init = ds_binned.bin_volume_annual[0,:,0].values
#        thick_init = ds_binned.bin_thick_annual[0,:,0].values
#        widths_m = nfls[0].widths_m
#        lengths_m = vol_m3_init / thick_init / widths_m
                                
                                
        glac_zbed_all[glac_no] = {}
        glac_thick_all[glac_no] = {}
        glac_zsurf_all[glac_no] = {}
        glac_vol_all[glac_no] = {}
        
        for rcp in rcps:
#        for rcp in rcps[0:1]:
            
            glac_zbed_all[glac_no][rcp] = {}
            glac_thick_all[glac_no][rcp] = {}
            glac_zsurf_all[glac_no][rcp] = {}
            glac_vol_all[glac_no][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
#            for gcm_name in gcm_names[0:1]:

                ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                ds_stats_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                for i in os.listdir(ds_stats_fp):
                    if i.startswith(glac_no):
                        ds_stats_fn = i
                
                ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)
                ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)

                thick = ds_binned.bin_thick_annual[0,:,:].values
                zsurf_init = ds_binned.bin_surface_h_initial[0].values
                zbed = zsurf_init - thick[:,cs_idx]
                vol = ds_stats.glac_volume_annual[0,:].values
                
                glac_thick_all[glac_no][rcp][gcm_name] = thick
                glac_zbed_all[glac_no][rcp][gcm_name] = zbed
                glac_zsurf_all[glac_no][rcp][gcm_name] = zbed[:,np.newaxis] + thick
                glac_vol_all[glac_no][rcp][gcm_name] = vol
                
                
        # MULTI-GCM STATISTICS
        glac_multigcm_zbed[glac_no] = {}
        glac_multigcm_thick[glac_no] = {}
        glac_multigcm_zsurf[glac_no] = {}
        glac_multigcm_vol[glac_no] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
#                print(rcp, gcm_name)
    
                glac_zbed_gcm = glac_zbed_all[glac_no][rcp][gcm_name]
                glac_thick_gcm = glac_thick_all[glac_no][rcp][gcm_name]
                glac_zsurf_gcm = glac_zsurf_all[glac_no][rcp][gcm_name]
                glac_vol_gcm = glac_vol_all[glac_no][rcp][gcm_name]
                
                if x.shape[0] > glac_zbed_gcm.shape[0]:
                    x = x[0:glac_zbed_gcm.shape[0]]
    
                if ngcm == 0:
                    glac_zbed_gcm_all = glac_zbed_gcm 
                    glac_thick_gcm_all = glac_thick_gcm[np.newaxis,:,:]
                    glac_zsurf_gcm_all = glac_zsurf_gcm[np.newaxis,:,:]
                    glac_vol_gcm_all = glac_vol_gcm[np.newaxis,:]
                else:
                    glac_zbed_gcm_all = np.vstack((glac_zbed_gcm_all, glac_zbed_gcm))
                    glac_thick_gcm_all = np.vstack((glac_thick_gcm_all, glac_thick_gcm[np.newaxis,:,:]))
                    glac_zsurf_gcm_all = np.vstack((glac_zsurf_gcm_all, glac_zsurf_gcm[np.newaxis,:,:]))
                    glac_vol_gcm_all = np.vstack((glac_vol_gcm_all, glac_vol_gcm[np.newaxis,:]))
            
            glac_multigcm_zbed[glac_no][rcp] = glac_zbed_gcm_all
            glac_multigcm_thick[glac_no][rcp] = glac_thick_gcm_all
            glac_multigcm_zsurf[glac_no][rcp] = glac_zsurf_gcm_all
            glac_multigcm_vol[glac_no][rcp] = glac_vol_gcm_all
    
    
        #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,0.65])
        ax.patch.set_facecolor('none')
        ax2 = fig.add_axes([0,0.67,1,0.35])
        ax2.patch.set_facecolor('none')
        ax3 = fig.add_axes([0.67,0.32,0.3,0.3])
        ax3.patch.set_facecolor('none')
        
        add_zbed = True
        ymin, ymax, thick_max = None, None, None
        for rcp in rcps:
            zbed_med = np.median(glac_multigcm_zbed[glac_no][rcp],axis=0)
            zbed_std = np.std(glac_multigcm_zbed[glac_no][rcp], axis=0)
            
            thick_med = np.median(glac_multigcm_thick[glac_no][rcp],axis=0)
            thick_std = np.std(glac_multigcm_thick[glac_no][rcp], axis=0)
            
            zsurf_med = np.median(glac_multigcm_zsurf[glac_no][rcp],axis=0)
            zsurf_std = np.std(glac_multigcm_zsurf[glac_no][rcp], axis=0)
            
            vol_med = np.median(glac_multigcm_vol[glac_no][rcp],axis=0)
            vol_std = np.std(glac_multigcm_vol[glac_no][rcp], axis=0)
            
            normyear_idx = np.where(years == normyear)[0][0]
            endyear_idx = np.where(years == endyear)[0][0]
            
            
            if add_zbed:
                ax.plot(x/1000, zbed_med[np.arange(len(x))],
                        color='k', linestyle='-', linewidth=1, zorder=5, label='zbed')
                ax.plot(x/1000, zsurf_med[np.arange(len(x)),normyear_idx], 
                             color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
                ax2.plot(x/1000, thick_med[np.arange(len(x)),normyear_idx], 
                         color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
                add_zbed = False
                
            ax.plot(x/1000, zsurf_med[np.arange(len(x)),endyear_idx], 
                         color=rcp_colordict[rcp], linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
            
            ax2.plot(x/1000, thick_med[np.arange(len(x)),endyear_idx],
                     color=rcp_colordict[rcp], linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
            
            ax3.plot(years, vol_med / vol_med[normyear_idx], color=rcp_colordict[rcp], 
                     linewidth=0.5, zorder=4, label=None)

            if rcp in rcps_plot_mad:
                ax3.fill_between(years, 
                                 (vol_med + vol_std)/vol_med[normyear_idx], 
                                 (vol_med - vol_std)/vol_med[normyear_idx],
                                 alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
            
            # ymin and ymax for bounds
            if ymin is None:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
                ymax = np.ceil(zsurf_med[:,endyear_idx].max()/100)*100
            if np.floor(zbed_med.min()/100)*100 < ymin:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
            if np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100 > ymax:
                ymax = np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100
            # thickness max for bounds  
            if thick_max is None:
                thick_max = np.ceil(thick_med.max()/10)*10
            if np.ceil(thick_med.max()/10)*10 > thick_max:
                thick_max = np.ceil(thick_med.max()/10)*10
        
        if ymin < 0:
            water_idx = np.where(zbed_med < 0)[0]
            # Add water level
            ax.plot(x[water_idx]/1000, np.zeros(x[water_idx].shape), color='aquamarine', linewidth=1)
        
        if xmax/1000 > 25:
            x_major, x_minor = 10, 2
        elif xmax/1000 > 15:
            x_major, x_minor = 5, 1
        else:
            x_major, x_minor = 2, 0.5
        
        y_major, y_minor = 500,100
        
        if thick_max > 200:
            thick_major, thick_minor = 100, 20
        else:
            thick_major, thick_minor = 50, 10
            
            
        # ----- GLACIER SPECIFIC PLOTS -----
        plot_legend = True
        add_glac_name = True
        if glac_no in ['1.10689']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -500, 3500
            y_major, y_minor = 1000, 200
            thick_max = 700
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                plot_legend = False
                add_glac_name = False
        elif glac_no in ['7.00238']:
            thick_max = 700
            thick_major, thick_minor = 200, 100
            ymin, ymax = -800, 1400
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                plot_legend = False
                add_glac_name = False
        elif glac_no in ['7.00027']:
            thick_max = 800
            thick_major, thick_minor = 200, 100
            ymin, ymax = -900, 1400
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                plot_legend = False
                add_glac_name = False
            
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0,xmax/1000)
        ax2.set_xlim(0,xmax/1000)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor)) 
        ax2.set_ylim(0,thick_max)
        ax2.yaxis.set_major_locator(MultipleLocator(thick_major))
#        ax2.yaxis.set_minor_locator(MultipleLocator(thick_minor))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.get_xaxis().set_visible(False)
            
        ax.set_ylabel('Elevation (m a.s.l.)')
        ax.set_xlabel('Distance along flowline (km)')
        ax2.set_ylabel('Ice thickness (m)', labelpad=10)
#        ax2.yaxis.set_label_position('right')
#        ax2.yaxis.tick_right()
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax.spines['top'].set_visible(False)
                
        if glac_no in glac_name_dict.keys():
            glac_name_text = glac_name_dict[glac_no]
        else:
             glac_name_text = glac_no
        
        if add_glac_name:
            ax2.text(0.98, 1.16, glac_name_text, size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax2.transAxes)
        ax2.text(0.02, 1.16, fa_label, size=10, horizontalalignment='left', 
                verticalalignment='top', transform=ax2.transAxes)
#        ax.legend(rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#                       handletextpad=0.25, borderpad=0, frameon=False)
        
        ax3.set_ylabel('Mass (-)')
        ax3.set_xlim(normyear, endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
        ax3.set_ylim(0,1.1)
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax3.tick_params(axis='both', which='major', direction='inout', right=True)
        ax3.tick_params(axis='both', which='minor', direction='in', right=True)
        vol_norm_gt = vol_med[normyear_idx] * pygem_prms.density_ice / 1e12
        if vol_norm_gt > 10:
            vol_norm_gt_str = str(int(np.round(vol_norm_gt,0))) + ' Gt'
        elif vol_norm_gt > 1:
            vol_norm_gt_str = str(np.round(vol_norm_gt,1)) + ' Gt'
        else:
            vol_norm_gt_str = str(np.round(vol_norm_gt,2)) + ' Gt'
        ax3.text(0.95, 0.95, vol_norm_gt_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax3.transAxes)
        
        # Legend
        if plot_legend:
            rcp_dict = {'ssp119': 'SSP1-1.9',
                        'ssp126': 'SSP1-2.6',
                        'ssp245': 'SSP2-4.5',
                        'ssp370': 'SSP3-7.0',
                        'ssp585': 'SSP5-8.5'}
            rcp_lines = []
            for rcp in rcps:
                line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=1)
                rcp_lines.append(line)
            rcp_labels = [rcp_dict[rcp] for rcp in rcps]
            ax2.legend(rcp_lines, rcp_labels, loc=(0.02,0.45), fontsize=8, labelspacing=0.25, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=1, columnspacing=0.5, frameon=False)
            
            other_labels = []
            other_lines = []
            # add years
            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=0)
            other_lines.append(line)
            other_labels.append('')
            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=1)
            other_lines.append(line)
            other_labels.append(str(normyear))
            line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
            other_lines.append(line)
            other_labels.append(str(endyear))
            if ymin < 0:
                line = Line2D([0,1],[0,1], color='aquamarine', linewidth=1)
                other_lines.append(line)
                other_labels.append('Sea level')
            line = Line2D([0,1],[0,1], color='k', linewidth=1)
            other_lines.append(line)
            other_labels.append('Bed')
            
            ax.legend(other_lines, other_labels, loc=(0.03,0.05), fontsize=8, labelspacing=0.25, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=1, columnspacing=0.5, frameon=False)
        
#        loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
#                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        
        
        # Save figure
        if 'rcp26' in rcps and 'ssp126' in rcps:
            scenario_str = 'rcps_ssps'
        elif 'rcp26' in rcps:
            scenario_str = 'rcps'
        elif 'ssp126' in rcps:
            scenario_str = 'ssps'
        fig_fn = (glac_no + '_profile_' + str(endyear) + '.png')
        if debris_fns:
            fig_fn = fig_fn.replace('.png','-wdebris.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        
#%%
if option_glacier_cs_plots_NSFANS:
    glac_nos = ['1.22193']

    netcdf_fp_list = ['/Users/drounce/Documents/HiMAT/Output/simulations-nsfans_control/',
                      '/Users/drounce/Documents/HiMAT/Output/simulations-nsfans_normal_tadjusted_v2/']
    
    fig_fp_multigcm = netcdf_fp_cmip5 + '/../../analysis/figures/ind_glaciers_v2/'

       
    cs_year = 2000
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    startyear_idx = np.where(years == startyear)[0][0]
    cs_idx = np.where(years == cs_year)[0][0]

    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp245']

    rcps_plot_mad = ['ssp245']

    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
        
    glac_name_dict = {'1.15645':'Kennicott',
                      '1.22193': 'Kahiltna'}
        
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,0.65])
    ax.patch.set_facecolor('none')
#    ax2 = fig.add_axes([0,0.67,1,0.35])
#    ax2.patch.set_facecolor('none')
    ax3 = fig.add_axes([0.6,0.32,0.37,0.3])
    ax3.patch.set_facecolor('none')
    
    rcp_lines = []
    rcp_labels = []
    
    for nfp, netcdf_fp_cmip5 in enumerate(netcdf_fp_list):

        # Set up processing
        glac_zbed_all = {}
        glac_thick_all = {}
        glac_zsurf_all = {}
        glac_vol_all = {}
        glac_multigcm_zbed = {}
        glac_multigcm_thick = {}
        glac_multigcm_zsurf = {}
        glac_multigcm_vol = {}
        for glac_no in glac_nos:
    
            gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
            
            tasks.init_present_time_glacier(gdir) # adds bins below
            debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
            nfls = gdir.read_pickle('model_flowlines')
            
            x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx
    
            glac_idx = np.nonzero(nfls[0].thick)[0]
            xmax = np.ceil(x[glac_idx].max()/1000+0.5)*1000
            
    #        vol_m3_init = ds_binned.bin_volume_annual[0,:,0].values
    #        thick_init = ds_binned.bin_thick_annual[0,:,0].values
    #        widths_m = nfls[0].widths_m
    #        lengths_m = vol_m3_init / thick_init / widths_m
                                    
                                    
            glac_zbed_all[glac_no] = {}
            glac_thick_all[glac_no] = {}
            glac_zsurf_all[glac_no] = {}
            glac_vol_all[glac_no] = {}
            
            for rcp in rcps:
    #        for rcp in rcps[0:1]:
                
                glac_zbed_all[glac_no][rcp] = {}
                glac_thick_all[glac_no][rcp] = {}
                glac_zsurf_all[glac_no][rcp] = {}
                glac_vol_all[glac_no][rcp] = {}
                
                if 'rcp' in rcp:
                    gcm_names = gcm_names_rcps
                elif 'ssp' in rcp:
                    gcm_names = gcm_names_ssps
                    
                for gcm_name in gcm_names:
    #            for gcm_name in gcm_names[0:1]:
    
                    ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                    for i in os.listdir(ds_binned_fp):
                        if i.startswith(glac_no):
                            ds_binned_fn = i
                    ds_stats_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                    for i in os.listdir(ds_stats_fp):
                        if i.startswith(glac_no):
                            ds_stats_fn = i
                    
                    ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)
                    ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)
    
                    thick = ds_binned.bin_thick_annual[0,:,:].values
                    zsurf_init = ds_binned.bin_surface_h_initial[0].values
                    zbed = zsurf_init - thick[:,cs_idx]
                    vol = ds_stats.glac_volume_annual[0,:].values
                    
                    glac_thick_all[glac_no][rcp][gcm_name] = thick
                    glac_zbed_all[glac_no][rcp][gcm_name] = zbed
                    glac_zsurf_all[glac_no][rcp][gcm_name] = zbed[:,np.newaxis] + thick
                    glac_vol_all[glac_no][rcp][gcm_name] = vol
                    
                    
            # MULTI-GCM STATISTICS
            glac_multigcm_zbed[glac_no] = {}
            glac_multigcm_thick[glac_no] = {}
            glac_multigcm_zsurf[glac_no] = {}
            glac_multigcm_vol[glac_no] = {}
            for rcp in rcps: 
                
                if 'rcp' in rcp:
                    gcm_names = gcm_names_rcps
                elif 'ssp' in rcp:
                    gcm_names = gcm_names_ssps
                    
                for ngcm, gcm_name in enumerate(gcm_names):
                    
    #                print(rcp, gcm_name)
        
                    glac_zbed_gcm = glac_zbed_all[glac_no][rcp][gcm_name]
                    glac_thick_gcm = glac_thick_all[glac_no][rcp][gcm_name]
                    glac_zsurf_gcm = glac_zsurf_all[glac_no][rcp][gcm_name]
                    glac_vol_gcm = glac_vol_all[glac_no][rcp][gcm_name]
                    
                    if x.shape[0] > glac_zbed_gcm.shape[0]:
                        x = x[0:glac_zbed_gcm.shape[0]]
        
                    if ngcm == 0:
                        glac_zbed_gcm_all = glac_zbed_gcm 
                        glac_thick_gcm_all = glac_thick_gcm[np.newaxis,:,:]
                        glac_zsurf_gcm_all = glac_zsurf_gcm[np.newaxis,:,:]
                        glac_vol_gcm_all = glac_vol_gcm[np.newaxis,:]
                    else:
                        glac_zbed_gcm_all = np.vstack((glac_zbed_gcm_all, glac_zbed_gcm))
                        glac_thick_gcm_all = np.vstack((glac_thick_gcm_all, glac_thick_gcm[np.newaxis,:,:]))
                        glac_zsurf_gcm_all = np.vstack((glac_zsurf_gcm_all, glac_zsurf_gcm[np.newaxis,:,:]))
                        glac_vol_gcm_all = np.vstack((glac_vol_gcm_all, glac_vol_gcm[np.newaxis,:]))
                
                glac_multigcm_zbed[glac_no][rcp] = glac_zbed_gcm_all
                glac_multigcm_thick[glac_no][rcp] = glac_thick_gcm_all
                glac_multigcm_zsurf[glac_no][rcp] = glac_zsurf_gcm_all
                glac_multigcm_vol[glac_no][rcp] = glac_vol_gcm_all
        
            #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
            if nfp == 0:
                add_zbed = True
                color = '#76B8E5'
                label = 'Control'
            else:
                add_zbed = False
                color = '#ED2024'
                label = '$T_{snow,adjusted}$'
#                label = 'Rain/snow temp. threshold adjusted'
            line = Line2D([0,1],[0,1], color=color, linewidth=1)
            rcp_lines.append(line)
            rcp_labels.append(label)
            
            ymin, ymax, thick_max = None, None, None
            for rcp in rcps:
                zbed_med = np.median(glac_multigcm_zbed[glac_no][rcp],axis=0)
                zbed_std = np.std(glac_multigcm_zbed[glac_no][rcp], axis=0)
                
                thick_med = np.median(glac_multigcm_thick[glac_no][rcp],axis=0)
                thick_std = np.std(glac_multigcm_thick[glac_no][rcp], axis=0)
                
                zsurf_med = np.median(glac_multigcm_zsurf[glac_no][rcp],axis=0)
                zsurf_std = np.std(glac_multigcm_zsurf[glac_no][rcp], axis=0)
                
                vol_med = np.median(glac_multigcm_vol[glac_no][rcp],axis=0)
                vol_std = np.std(glac_multigcm_vol[glac_no][rcp], axis=0)
                
                normyear_idx = np.where(years == normyear)[0][0]
                endyear_idx = np.where(years == endyear)[0][0]
                
                
                if add_zbed:
                    ax.plot(x/1000, zbed_med[np.arange(len(x))],
                            color='k', linestyle='-', linewidth=1, zorder=5, label='zbed')
                    ax.plot(x/1000, zsurf_med[np.arange(len(x)),normyear_idx], 
                                 color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
#                    ax2.plot(x/1000, thick_med[np.arange(len(x)),normyear_idx], 
#                             color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
                    add_zbed = False
                    
                ax.plot(x/1000, zsurf_med[np.arange(len(x)),endyear_idx], 
                             color=color, linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
                
#                ax2.plot(x/1000, thick_med[np.arange(len(x)),endyear_idx],
#                         color=color, linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
                
                ax3.plot(years, vol_med / vol_med[normyear_idx], color=color, 
                         linewidth=0.5, zorder=4, label=None)
    
                if rcp in rcps_plot_mad:
                    ax3.fill_between(years, 
                                     (vol_med + vol_std)/vol_med[normyear_idx], 
                                     (vol_med - vol_std)/vol_med[normyear_idx],
                                     alpha=0.2, facecolor=color, label=None)
                
                # ymin and ymax for bounds
                if ymin is None:
                    ymin = np.floor(zbed_med[glac_idx].min()/100)*100
                    ymax = np.ceil(zsurf_med[:,endyear_idx].max()/100)*100
                if np.floor(zbed_med.min()/100)*100 < ymin:
                    ymin = np.floor(zbed_med[glac_idx].min()/100)*100
                if np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100 > ymax:
                    ymax = np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100
                # thickness max for bounds  
                if thick_max is None:
                    thick_max = np.ceil(thick_med.max()/10)*10
                if np.ceil(thick_med.max()/10)*10 > thick_max:
                    thick_max = np.ceil(thick_med.max()/10)*10
            
#            if ymin < 0:
#                water_idx = np.where(zbed_med < 0)[0]
#                # Add water level
#                ax.plot(x[water_idx]/1000, np.zeros(x[water_idx].shape), color='aquamarine', linewidth=1)
            
            if xmax/1000 > 25:
                x_major, x_minor = 10, 2
            elif xmax/1000 > 15:
                x_major, x_minor = 5, 1
            else:
                x_major, x_minor = 2, 0.5
            
            y_major, y_minor = 500,100
            
            if thick_max > 200:
                thick_major, thick_minor = 100, 20
            else:
                thick_major, thick_minor = 50, 10
                
                
            # ----- GLACIER SPECIFIC PLOTS -----
            plot_legend = False
            add_glac_name = False
            if glac_no in ['1.22193']:
                thick_major, thick_minor = 200, 100
                ymin, ymax = -200, 4800
                y_major, y_minor = 1000, 200
                thick_max = 700
                if nfp==1:
                    plot_legend = True
                    add_glac_name = True
                
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0,xmax/1000)
#            ax2.set_xlim(0,xmax/1000)
            ax.xaxis.set_major_locator(MultipleLocator(x_major))
            ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
            ax.yaxis.set_major_locator(MultipleLocator(y_major))
            ax.yaxis.set_minor_locator(MultipleLocator(y_minor)) 
#            ax2.set_ylim(0,thick_max)
#            ax2.yaxis.set_major_locator(MultipleLocator(thick_major))
#    #        ax2.yaxis.set_minor_locator(MultipleLocator(thick_minor))
#            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
#            ax2.get_xaxis().set_visible(False)
                
            ax.set_ylabel('Elevation (m a.s.l.)', size=14)
            ax.set_xlabel('Distance along flowline (km)', size=14)
            ax.tick_params(which='major', direction='inout', right=False, labelsize=12)
            ax.tick_params(which='minor', direction='in', right=False, labelsize=12)
#            ax2.set_ylabel('Ice thickness (m)', labelpad=10)
#    #        ax2.yaxis.set_label_position('right')
#    #        ax2.yaxis.tick_right()
#            ax.tick_params(axis='both', which='major', direction='inout', right=True)
#            ax.tick_params(axis='both', which='minor', direction='in', right=True)
#            ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#            ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#    #        ax.spines['top'].set_visible(False)
                    
            if glac_no in glac_name_dict.keys():
                glac_name_text = glac_name_dict[glac_no]
            else:
                 glac_name_text = glac_no
            
            if add_glac_name:
                ax.text(0.03, 0.98, glac_name_text, size=12, horizontalalignment='left', 
                        verticalalignment='top', transform=ax.transAxes)
    #        ax.legend(rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    #                       handletextpad=0.25, borderpad=0, frameon=False)
            
            ax3.set_ylabel('Mass (-)', size=14)
            ax3.set_xlim(normyear, endyear)
            ax3.xaxis.set_major_locator(MultipleLocator(40))
            ax3.xaxis.set_minor_locator(MultipleLocator(10))
            ax3.set_ylim(0,1.1)
            ax3.yaxis.set_major_locator(MultipleLocator(0.5))
            ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax3.tick_params(axis='both', which='major', direction='inout', right=True, labelsize=12)
            ax3.tick_params(axis='both', which='minor', direction='in', right=True, labelsize=12)
            vol_norm_gt = vol_med[normyear_idx] * pygem_prms.density_ice / 1e12
            if vol_norm_gt > 10:
                vol_norm_gt_str = str(int(np.round(vol_norm_gt,0))) + ' Gt'
            elif vol_norm_gt > 1:
                vol_norm_gt_str = str(np.round(vol_norm_gt,1)) + ' Gt'
            else:
                vol_norm_gt_str = str(np.round(vol_norm_gt,2)) + ' Gt'
            if nfp == 0:
                ax3.text(0.95, 0.95, vol_norm_gt_str, size=12, horizontalalignment='right', 
                        verticalalignment='top', transform=ax3.transAxes)
            
            # Legend
            if plot_legend:
#                ax2.legend(rcp_lines, rcp_labels, loc=(0.02,0.45), fontsize=8, labelspacing=0.25, handlelength=1, 
#                          handletextpad=0.25, borderpad=0, ncol=1, columnspacing=0.5, frameon=False)
                other_lines = []
                other_labels = []
                # add years
                line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=1)
                other_lines.append(line)
                other_labels.append(str(normyear))
                line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
                other_lines.append(line)
                other_labels.append(str(endyear))
                line = Line2D([0,1],[0,1], color='k', linewidth=1)
                other_lines.append(line)
                other_labels.append('Bed')
                
                line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=0)
                other_lines.append(line)
                other_labels.append('')
                
                for rcp_label in rcp_labels:
                    other_labels.append(rcp_label)
                for rcp_line in rcp_lines:
                    other_lines.append(rcp_line)
                
                ax.legend(other_lines, other_labels, loc=(0.02,0.01), fontsize=12, labelspacing=0.25, handlelength=1, 
                          handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.5, frameon=False)
            
    #        loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
    #                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        
        
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = (glac_no + '_profile_' + str(endyear) + '_control_vs_tsnowthreshold.png')
    fig.set_size_inches(4.5,3)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    plt.show()


#%%   
if option_debris_comparison:
    
    pickle_fp = '/Users/drounce/Documents/HiMAT/spc_backup/analysis/pickle/'
    pickle_fp_nodebris = '/Users/drounce/Documents/HiMAT/spc_backup/analysis-nodebris/pickle-nodebris/'
    
    #%%
    fig_fp = pickle_fp + '/../figures/'
    
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    regions = [1, 2, 11, 12, 13, 14, 15, 16, 18]
#    regions = [14]
    
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_debriscompare = fig_fp + 'debris_compare/'
    if not os.path.exists(fig_fp_debriscompare):
        os.makedirs(fig_fp_debriscompare, exist_ok=True)
    
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_nodebris = {}
#    reg_vol_all_bwl = {}
#    reg_area_all = {} 
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_nodebris[reg] = {}
#        reg_vol_all_bwl[reg] = {}
#        reg_area_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_nodebris[reg][rcp] = {}
#            reg_vol_all_bwl[reg][rcp] = {}
#            reg_area_all[reg][rcp] = {}
            
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                
                 # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                    
                # ===== NO DEBRIS =====
                pickle_fp_reg_nodebris =  pickle_fp_nodebris + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                
                 # Volume
                with open(pickle_fp_reg_nodebris + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual_nodebris = pickle.load(f)
                    
                
                print(reg, gcm_name, rcp, 'difference [%]', np.round((reg_vol_annual_nodebris[50] - reg_vol_annual[50])/ reg_vol_annual[0] * 100,2))
                
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                reg_vol_all_nodebris[reg][rcp][gcm_name] = reg_vol_annual_nodebris
                
    
    # MULTI-GCM STATISTICS
    ds_multigcm_vol = {}
    ds_multigcm_vol_nodebris = {}
#    ds_multigcm_vol_bsl = {}
#    ds_multigcm_area = {}
    for reg in regions:
        ds_multigcm_vol[reg] = {}
        ds_multigcm_vol_nodebris[reg] = {}
#        ds_multigcm_vol_bsl[reg] = {}
#        ds_multigcm_area[reg] = {}
        for rcp in rcps: 
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)
    
                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_gcm_nodebris = reg_vol_all_nodebris[reg][rcp][gcm_name]
#                reg_vol_bsl_gcm = reg_vol_all_bwl[reg][rcp][gcm_name]
#                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
    
                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_vol_gcm_all_nodebris = reg_vol_gcm_nodebris
#                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm   
#                    reg_area_gcm_all = reg_area_gcm    
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_gcm_all_nodebris = np.vstack((reg_vol_gcm_all_nodebris, reg_vol_gcm_nodebris))
#                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
#                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
            
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_vol_nodebris[reg][rcp] = reg_vol_gcm_all_nodebris
#            ds_multigcm_vol_bsl[reg][rcp] = reg_vol_bsl_gcm_all
#            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
                
                
    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    normyear_idx = np.where(years == normyear)[0][0]
    
    for reg in regions:
        fig = plt.figure()    
        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[2:3,0:2])
        ax3 = fig.add_subplot(gs[3:4,0:2])

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_mean = np.median(reg_vol, axis=0)
            reg_vol_std = median_abs_deviation(reg_vol, axis=0)

            reg_vol_nodebris = ds_multigcm_vol_nodebris[reg][rcp]
            reg_vol_mean_nodebris = np.median(reg_vol_nodebris, axis=0)
            reg_vol_std_nodebris = median_abs_deviation(reg_vol_nodebris, axis=0)
            
            # Normalized
            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
            
            reg_vol_mean_norm_nodebris = reg_vol_mean_nodebris / reg_vol_mean_nodebris[normyear_idx]
            reg_vol_std_norm_nodebris = reg_vol_std_nodebris / reg_vol_mean_nodebris[normyear_idx]
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_mean_norm_nodebris.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nodebris[nyear])
                year_idx = np.where(reg_vol_mean_norm_nodebris[nyear] < reg_vol_mean_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year

            # Plot
            ax1.plot(years, reg_vol_mean_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            ax1.plot(years, reg_vol_mean_norm_nodebris, color=rcp_colordict[rcp], linestyle=':', 
                    linewidth=1, zorder=3)
            ax2.plot(years, (reg_vol_mean_norm - reg_vol_mean_norm_nodebris)*100, color=rcp_colordict[rcp], 
                     linewidth=1, zorder=4)
            ax3.plot(years, reg_vol_delay, color=rcp_colordict[rcp], linewidth=1, zorder=4)
#            if rcp in rcps_plot_mad:
#                ax.fill_between(years, 
#                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
#                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
#                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax1.set_ylabel('Mass (rel. to 2015)')
        ax1.set_xlim(startyear, endyear)
        ax1.xaxis.set_major_locator(MultipleLocator(40))
        ax1.xaxis.set_minor_locator(MultipleLocator(10))
        ax1.set_ylim(0,1.1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.tick_params(axis='both', which='major', direction='inout', right=True)
        ax1.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Difference (%)')
        ax2.set_xlim(startyear, endyear)
        ax2.xaxis.set_major_locator(MultipleLocator(40))
        ax2.xaxis.set_minor_locator(MultipleLocator(10))
        if reg in [1,2,11]:
            ax2.set_ylim(0,7)
        elif reg in [18]:
            ax2.set_ylim(0,14)
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax3.set_ylabel('Difference (yrs)')
        ax3.set_xlim(startyear, endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [1,2,11]:
#            ax2.set_ylim(0,7)
#        elif reg in [18]:
#            ax2.set_ylim(0,14)
#        ax2.yaxis.set_major_locator(MultipleLocator(5))
#        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax1.text(1, 1.16, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax1.transAxes)
        ax1.axes.xaxis.set_ticklabels([])
            
        # Save figure
        fig_fn = (str(reg).zfill(2) + '_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'debriscompare.png')
        fig.set_size_inches(4,4)
        fig.savefig(fig_fp_debriscompare + fig_fn, bbox_inches='tight', dpi=300)
        
    #%%
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    ax9 = fig.add_subplot(gs[2,2])
    
    regions_ordered = [1,2,11,13,14,15,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
        
        reg = regions_ordered[nax]            
    
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_mean = np.median(reg_vol, axis=0)
            reg_vol_std = median_abs_deviation(reg_vol, axis=0)

            reg_vol_nodebris = ds_multigcm_vol_nodebris[reg][rcp]
            reg_vol_mean_nodebris = np.median(reg_vol_nodebris, axis=0)
            reg_vol_std_nodebris = median_abs_deviation(reg_vol_nodebris, axis=0)
            
            # Normalized
            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
            
            reg_vol_mean_norm_nodebris = reg_vol_mean_nodebris / reg_vol_mean_nodebris[normyear_idx]
            reg_vol_std_norm_nodebris = reg_vol_std_nodebris / reg_vol_mean_nodebris[normyear_idx]
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_mean_norm_nodebris.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nodebris[nyear])
                year_idx = np.where(reg_vol_mean_norm_nodebris[nyear] < reg_vol_mean_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year

            # Plot
            ax.plot(years, reg_vol_mean_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            ax.plot(years, reg_vol_mean_norm_nodebris, color=rcp_colordict[rcp], linestyle=':', 
                    linewidth=1, zorder=3, label=None)
#            ax2.plot(years, (reg_vol_mean_norm - reg_vol_mean_norm_nodebris)*100, color=rcp_colordict[rcp], 
#                     linewidth=1, zorder=4)
#            ax3.plot(years, reg_vol_delay, color=rcp_colordict[rcp], linewidth=1, zorder=4)
#            if rcp in rcps_plot_mad:
#                ax.fill_between(years, 
#                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
#                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
#                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1, ax4, ax7]:
            ax.set_ylabel('Mass (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        # Legend
        if ax == ax3:            
            rcp_dict = {'ssp119': 'SSP1-1.9',
                        'ssp126': 'SSP1-2.6',
                        'ssp245': 'SSP2-4.5',
                        'ssp370': 'SSP3-7.0',
                        'ssp585': 'SSP5-8.5'}
            rcp_lines = []
            rcp_labels = []
            line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
            rcp_lines.append(line)
            rcp_labels.append('Debris')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=1)
            rcp_lines.append(line)
            rcp_labels.append('Clean')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
            rcp_lines.append(line)
            rcp_labels.append(' ')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
            rcp_lines.append(line)
            rcp_labels.append(' ')
            
            for rcp in rcps:
                line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=1)
                rcp_lines.append(line)
                rcp_labels.append(rcp_dict[rcp])
            
            ax.legend(rcp_lines, rcp_labels, loc=(0.13,0.45), fontsize=8, labelspacing=0.2, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.3, frameon=False)
        
        
#        ax2.yaxis.set_label_position("right")
#        ax2.set_ylabel('Difference (%)')
#        ax2.set_xlim(startyear, endyear)
#        ax2.xaxis.set_major_locator(MultipleLocator(40))
#        ax2.xaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [1,2,11]:
#            ax2.set_ylim(0,7)
#        elif reg in [18]:
#            ax2.set_ylim(0,14)
#        ax2.yaxis.set_major_locator(MultipleLocator(5))
#        ax2.yaxis.set_minor_locator(MultipleLocator(1))
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)

        ax.text(1, 1.14, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes)
    

    # Save figure
    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'debriscompare.png')
    fig.set_size_inches(6.5,5.5)
    fig.savefig(fig_fp_debriscompare + fig_fn, bbox_inches='tight', dpi=300)
    

#if option_calving_comparison:
#    
#    pickle_fp = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving_v3/pickle/'
#    pickle_fp_nocalving = '/Users/drounce/Documents/HiMAT/spc_backup/analysis/pickle/'
#    
#    fig_fp = pickle_fp + '/../figures/'
#    
#    startyear = 2015
#    endyear = 2100
#    years = np.arange(2000,2101+1)
#    
#    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
#                 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
#    
#    regions = [1, 3, 4, 5, 7, 9, 17, 19]
#    
#    rcps_plot_mad = ['ssp126', 'ssp585']
#    
#    fig_fp_calvingcompare = fig_fp + 'calving_compare/'
#    if not os.path.exists(fig_fp_calvingcompare):
#        os.makedirs(fig_fp_calvingcompare, exist_ok=True)
#    
#    # Set up processing
#    reg_vol_all = {}
#    reg_vol_all_nocalving = {}
##    reg_vol_all_bwl = {}
##    reg_area_all = {} 
#            
#    for reg in regions:
#    
#        reg_vol_all[reg] = {}
#        reg_vol_all_nocalving[reg] = {}
##        reg_vol_all_bwl[reg] = {}
##        reg_area_all[reg] = {}
#        
#        for rcp in rcps:
#            reg_vol_all[reg][rcp] = {}
#            reg_vol_all_nocalving[reg][rcp] = {}
##            reg_vol_all_bwl[reg][rcp] = {}
##            reg_area_all[reg][rcp] = {}
#            
#                
#            for gcm_name in gcm_names:
#                
#                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
#                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
#                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
#                # Region string prefix
#                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
#                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
#                
#                # Filenames
#                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
##                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
##                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
##                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
##                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
##                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
##                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
##                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
##                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
##                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
##                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
##                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
##                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
##                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
##                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
##                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
##                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
##                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
##                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
##                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
##                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
##                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
##                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
##                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
##                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
##                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
##                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
##                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
##                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
##                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
#                
#                 # Volume
#                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
#                    reg_vol_annual = pickle.load(f)
##                # Volume below sea level
##                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
##                    reg_vol_annual_bwl = pickle.load(f)
##                # Volume below debris
##                with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
##                    reg_vol_annual_bd = pickle.load(f)
##                # Area 
##                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
##                    reg_area_annual = pickle.load(f)
##                # Area below debris
##                with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
##                    reg_area_annual_bd = pickle.load(f)
##                # Binned Volume
##                with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
##                    reg_vol_annual_binned = pickle.load(f)
##                # Binned Volume below debris
##                with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
##                    reg_vol_annual_binned_bd = pickle.load(f)
##                # Binned Area
##                with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
##                    reg_area_annual_binned = pickle.load(f)
##                # Binned Area below debris
##                with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
##                    reg_area_annual_binned_bd = pickle.load(f)
##                # Mass balance: accumulation
##                with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
##                    reg_acc_monthly = pickle.load(f)
##                # Mass balance: refreeze
##                with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
##                    reg_refreeze_monthly = pickle.load(f)
##                # Mass balance: melt
##                with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
##                    reg_melt_monthly = pickle.load(f)
##                # Mass balance: frontal ablation
##                with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
##                    reg_frontalablation_monthly = pickle.load(f)
##                # Mass balance: total mass balance
##                with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
##                    reg_massbaltotal_monthly = pickle.load(f)
##                # Binned Climatic Mass Balance
##                with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
##                    reg_mbclim_annual_binned = pickle.load(f)
##                # Runoff: moving-gauged
##                with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
##                    reg_runoff_monthly_moving = pickle.load(f)
##                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
##                    watershed_runoff_monthly_moving = pickle.load(f)
##                # Runoff: fixed-gauged
##                with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
##                    reg_runoff_monthly_fixed = pickle.load(f)
##                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
##                    watershed_runoff_monthly_fixed = pickle.load(f)
##                # Runoff: precipitation
##                with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
##                    reg_prec_monthly = pickle.load(f)
##                with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'rb') as f:
##                    watershed_prec_monthly= pickle.load(f)
##                # Runoff: off-glacier precipitation
##                with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
##                    reg_offglac_prec_monthly = pickle.load(f)
##                with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'rb') as f:
##                    watershed_offglac_prec_monthly = pickle.load(f)
##                # Runoff: off-glacier melt
##                with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
##                    reg_offglac_melt_monthly = pickle.load(f)
##                with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'rb') as f:
##                    watershed_offglac_melt_monthly = pickle.load(f)
##                # Runoff: off-glacier refreeze
##                with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
##                    reg_offglac_refreeze_monthly = pickle.load(f)
##                with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'rb') as f:
##                    watershed_offglac_refreeze_monthly = pickle.load(f)
##                # ELA
##                with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
##                    reg_ela_annual = pickle.load(f)
##                # AAR
##                with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
##                    reg_aar_annual = pickle.load(f) 
#                    
#                # ===== NO CALVING =====
#                pickle_fp_reg_nocalving =  pickle_fp_nocalving + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
#                
#                # Filenames
#                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
##                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
##                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
##                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
##                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
##                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
##                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
##                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
##                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
##                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
##                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
##                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
##                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
##                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
##                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
##                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
##                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
##                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
##                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
##                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
##                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
##                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
##                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
##                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
##                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
##                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
##                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
##                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
##                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
##                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
#                
#                 # Volume
#                with open(pickle_fp_reg_nocalving + fn_reg_vol_annual, 'rb') as f:
#                    reg_vol_annual_nocalving = pickle.load(f)
#                    
#                
#                print(reg, gcm_name, rcp, 'difference [%]', np.round((reg_vol_annual_nocalving[50] - reg_vol_annual[50])/ reg_vol_annual[0] * 100,2))
#                
#                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
#                reg_vol_all_nocalving[reg][rcp][gcm_name] = reg_vol_annual_nocalving
#                
#    
#    # MULTI-GCM STATISTICS
#    ds_multigcm_vol = {}
#    ds_multigcm_vol_nocalving = {}
##    ds_multigcm_vol_bsl = {}
##    ds_multigcm_area = {}
#    for reg in regions:
#        ds_multigcm_vol[reg] = {}
#        ds_multigcm_vol_nocalving[reg] = {}
##        ds_multigcm_vol_bsl[reg] = {}
##        ds_multigcm_area[reg] = {}
#        for rcp in rcps: 
#                
#            for ngcm, gcm_name in enumerate(gcm_names):
#                
#                print(rcp, gcm_name)
#    
#                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
#                reg_vol_gcm_nocalving = reg_vol_all_nocalving[reg][rcp][gcm_name]
##                reg_vol_bsl_gcm = reg_vol_all_bwl[reg][rcp][gcm_name]
##                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
#    
#                if ngcm == 0:
#                    reg_vol_gcm_all = reg_vol_gcm   
#                    reg_vol_gcm_all_nocalving = reg_vol_gcm_nocalving
##                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm   
##                    reg_area_gcm_all = reg_area_gcm    
#                else:
#                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
#                    reg_vol_gcm_all_nocalving = np.vstack((reg_vol_gcm_all_nocalving, reg_vol_gcm_nocalving))
##                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
##                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
#            
#            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
#            ds_multigcm_vol_nocalving[reg][rcp] = reg_vol_gcm_all_nocalving
##            ds_multigcm_vol_bsl[reg][rcp] = reg_vol_bsl_gcm_all
##            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
#                
#                
#    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
#    normyear_idx = np.where(years == normyear)[0][0]
#    
#    for reg in regions:
#        fig = plt.figure()    
#        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0)
#        ax1 = fig.add_subplot(gs[0:2,0:2])
#        ax2 = fig.add_subplot(gs[2:3,0:2])
#        ax3 = fig.add_subplot(gs[3:4,0:2])
#
#        for rcp in rcps:
#            
#            # Median and absolute median deviation
#            reg_vol = ds_multigcm_vol[reg][rcp]
#            reg_vol_mean = np.median(reg_vol, axis=0)
#            reg_vol_std = median_abs_deviation(reg_vol, axis=0)
#
#            reg_vol_nocalving = ds_multigcm_vol_nocalving[reg][rcp]
#            reg_vol_mean_nocalving = np.median(reg_vol_nocalving, axis=0)
#            reg_vol_std_nocalving = median_abs_deviation(reg_vol_nocalving, axis=0)
#            
#            # Normalized
#            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
#            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
#            
#            reg_vol_mean_norm_nocalving = reg_vol_mean_nocalving / reg_vol_mean_nocalving[normyear_idx]
#            reg_vol_std_norm_nocalving = reg_vol_std_nocalving / reg_vol_mean_nocalving[normyear_idx]
#            
#            # Delay in timing
#            reg_vol_delay = np.zeros(reg_vol_mean_norm_nocalving.shape)
#            for nyear, year in enumerate(years):
##                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nocalving[nyear])
#                year_idx = np.where(reg_vol_mean_norm_nocalving[nyear] < reg_vol_mean_norm)[0]
#                if len(year_idx) > 0:
#                    reg_vol_delay[nyear] = years[year_idx[-1]] - year
#
#            # Plot
#            ax1.plot(years, reg_vol_mean_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
#                    linewidth=1, zorder=4, label=rcp)
#            ax1.plot(years, reg_vol_mean_norm_nocalving, color=rcp_colordict[rcp], linestyle=':', 
#                    linewidth=1, zorder=3)
#            ax2.plot(years, (reg_vol_mean_norm - reg_vol_mean_norm_nocalving)*100, color=rcp_colordict[rcp], 
#                     linewidth=1, zorder=4)
#            ax3.plot(years, reg_vol_delay, color=rcp_colordict[rcp], linewidth=1, zorder=4)
##            if rcp in rcps_plot_mad:
##                ax.fill_between(years, 
##                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
##                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
##                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
#        
#        ax1.set_ylabel('Mass (rel. to 2015)')
#        ax1.set_xlim(startyear, endyear)
#        ax1.xaxis.set_major_locator(MultipleLocator(40))
#        ax1.xaxis.set_minor_locator(MultipleLocator(10))
#        ax1.set_ylim(0,1.1)
#        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
#        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
#        ax1.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax1.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        ax2.yaxis.set_label_position("right")
#        ax2.set_ylabel('Difference (%)')
#        ax2.set_xlim(startyear, endyear)
#        ax2.xaxis.set_major_locator(MultipleLocator(40))
#        ax2.xaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [1,2,11]:
#            ax2.set_ylim(0,7)
#        elif reg in [18]:
#            ax2.set_ylim(0,14)
#        ax2.yaxis.set_major_locator(MultipleLocator(5))
#        ax2.yaxis.set_minor_locator(MultipleLocator(1))
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        ax3.set_ylabel('Difference (yrs)')
#        ax3.set_xlim(startyear, endyear)
#        ax3.xaxis.set_major_locator(MultipleLocator(40))
#        ax3.xaxis.set_minor_locator(MultipleLocator(10))
##        if reg in [1,2,11]:
##            ax2.set_ylim(0,7)
##        elif reg in [18]:
##            ax2.set_ylim(0,14)
##        ax2.yaxis.set_major_locator(MultipleLocator(5))
##        ax2.yaxis.set_minor_locator(MultipleLocator(1))
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        ax1.text(1, 1.16, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                 verticalalignment='top', transform=ax1.transAxes)
#        ax1.axes.xaxis.set_ticklabels([])
#            
#        # Save figure
#        fig_fn = (str(reg).zfill(2) + '_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'calvingcompare.png')
#        fig.set_size_inches(4,4)
#        fig.savefig(fig_fp_calvingcompare + fig_fn, bbox_inches='tight', dpi=300)
#        
#    #%%
#    fig = plt.figure()
#    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.4)
#    ax1 = fig.add_subplot(gs[0,0])
#    ax2 = fig.add_subplot(gs[0,1])
#    ax3 = fig.add_subplot(gs[0,2])
#    ax4 = fig.add_subplot(gs[1,0])
#    ax5 = fig.add_subplot(gs[1,1])
#    ax6 = fig.add_subplot(gs[1,2])
#    ax7 = fig.add_subplot(gs[2,0])
#    ax8 = fig.add_subplot(gs[2,1])
#    
#    regions_ordered = [1,3,4,5,7,9,17,19]
#    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
#        
#        reg = regions_ordered[nax]  
#    
#        for rcp in rcps:
#            
#            # Median and absolute median deviation
#            reg_vol = ds_multigcm_vol[reg][rcp]
#            reg_vol_mean = np.median(reg_vol, axis=0)
#            reg_vol_std = median_abs_deviation(reg_vol, axis=0)
#
#            reg_vol_nocalving = ds_multigcm_vol_nocalving[reg][rcp]
#            reg_vol_mean_nocalving = np.median(reg_vol_nocalving, axis=0)
#            reg_vol_std_nocalving = median_abs_deviation(reg_vol_nocalving, axis=0)
#            
#            # Normalized
#            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
#            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
#            
#            reg_vol_mean_norm_nocalving = reg_vol_mean_nocalving / reg_vol_mean_nocalving[normyear_idx]
#            reg_vol_std_norm_nocalving = reg_vol_std_nocalving / reg_vol_mean_nocalving[normyear_idx]
#            
#            # Delay in timing
#            reg_vol_delay = np.zeros(reg_vol_mean_norm_nocalving.shape)
#            for nyear, year in enumerate(years):
##                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nocalving[nyear])
#                year_idx = np.where(reg_vol_mean_norm_nocalving[nyear] < reg_vol_mean_norm)[0]
#                if len(year_idx) > 0:
#                    reg_vol_delay[nyear] = years[year_idx[-1]] - year
#
#            # Plot
#            ax.plot(years, reg_vol_mean_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
#                    linewidth=1, zorder=4, label=rcp)
#            ax.plot(years, reg_vol_mean_norm_nocalving, color=rcp_colordict[rcp], linestyle=':', 
#                    linewidth=1, zorder=3, label=None)
##            ax2.plot(years, (reg_vol_mean_norm - reg_vol_mean_norm_nocalving)*100, color=rcp_colordict[rcp], 
##                     linewidth=1, zorder=4)
##            ax3.plot(years, reg_vol_delay, color=rcp_colordict[rcp], linewidth=1, zorder=4)
##            if rcp in rcps_plot_mad:
##                ax.fill_between(years, 
##                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
##                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
##                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
#            
#            if ax in [ax1, ax4, ax7]:
#                ax.set_ylabel('Mass (rel. to 2015)')
#            ax.set_xlim(startyear, endyear)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
#            ax.xaxis.set_minor_locator(MultipleLocator(10))
#            ax.set_ylim(0,1.1)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#            ax.tick_params(axis='both', which='major', direction='inout', right=True)
#            ax.tick_params(axis='both', which='minor', direction='in', right=True)
#            
#            # Legend
#            if ax == ax8:            
#                rcp_dict = {'ssp119': 'SSP1-1.9',
#                            'ssp126': 'SSP1-2.6',
#                            'ssp245': 'SSP2-4.5',
#                            'ssp370': 'SSP3-7.0',
#                            'ssp585': 'SSP5-8.5'}
#                rcp_lines = []
#                rcp_labels = []
#                line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
#                rcp_lines.append(line)
#                rcp_labels.append('Calving')
#                line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=1)
#                rcp_lines.append(line)
#                rcp_labels.append('No calving')
#                line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
#                rcp_lines.append(line)
#                rcp_labels.append(' ')
#                line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
#                rcp_lines.append(line)
#                rcp_labels.append(' ')
#                
#                for rcp in rcps:
#                    line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=1)
#                    rcp_lines.append(line)
#                    rcp_labels.append(rcp_dict[rcp])
#                
#                ax.legend(rcp_lines, rcp_labels, loc=(1.13,0.45), fontsize=8, labelspacing=0.2, handlelength=1, 
#                          handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.3, frameon=False)
#            
#            
#    #        ax2.yaxis.set_label_position("right")
#    #        ax2.set_ylabel('Difference (%)')
#    #        ax2.set_xlim(startyear, endyear)
#    #        ax2.xaxis.set_major_locator(MultipleLocator(40))
#    #        ax2.xaxis.set_minor_locator(MultipleLocator(10))
#    #        if reg in [1,2,11]:
#    #            ax2.set_ylim(0,7)
#    #        elif reg in [18]:
#    #            ax2.set_ylim(0,14)
#    #        ax2.yaxis.set_major_locator(MultipleLocator(5))
#    #        ax2.yaxis.set_minor_locator(MultipleLocator(1))
#    #        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#    #        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#    #        
#    #        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#    #        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#    
#            ax.text(1, 1.14, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                     verticalalignment='top', transform=ax.transAxes)
#
#    # Save figure
#    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'calvingcompare.png')
#    fig.set_size_inches(6.5,5.5)
#    fig.savefig(fig_fp_calvingcompare + fig_fn, bbox_inches='tight', dpi=300)
    
#%% ===== TEMPERATURE INCREASE FOR THE VARIOUS CLIMATE SCENARIOS =====     
if option_policy_temp_figs:
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']

    temp_dev_fn = 'Global_mean_temp_deviation_2081_2100_rel_1850_1900.csv'
    
    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    
    if os.path.exists(csv_fp + temp_dev_fn):
        temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
        
        print('check that all are in there')
    else:
        dates_table = modelsetup.datesmodelrun(startyear=1986, endyear=2100, spinupyears=0)
        temp_dev_cns = ['Scenario', 'GCM', 'global_mean_deviation_degC']
        temp_dev_df = pd.DataFrame(np.zeros((len(gcm_names_ssps)*4+len(gcm_names_rcps)*3,len(temp_dev_cns))), columns=temp_dev_cns)
        ncount = 0
        for scenario in rcps[0:1]:
            
            if 'rcp' in scenario:
                gcm_names = gcm_names_rcps
            elif 'ssp' in scenario:
                gcm_names = gcm_names_ssps
            
            for gcm_name in gcm_names[0:1]:
                print(scenario, gcm_name)
    
#                if scenario.startswith('ssp'):
#                    # Variable names
#                    temp_vn = 'tas'
#                    prec_vn = 'pr'
#                    elev_vn = 'orog'
#                    lat_vn = 'lat'
#                    lon_vn = 'lon'
#                    time_vn = 'time'
#                    # Variable filenames
#                    temp_fn = gcm_name + '_' + scenario + '_r1i1p1f1_' + temp_vn + '.nc'
#                    prec_fn = gcm_name + '_' + scenario + '_r1i1p1f1_' + prec_vn + '.nc'
#                    elev_fn = gcm_name + '_' + elev_vn + '.nc'
#                    # Variable filepaths
#                    var_fp = pygem_prms.cmip6_fp_prefix + gcm_name + '/'
#                    fx_fp = pygem_prms.cmip6_fp_prefix + gcm_name + '/'
#                    # Extra information
#                    timestep = pygem_prms.timestep
#                        
#                elif scenario.startswith('rcp'):
#                    # Variable names
#                    temp_vn = 'tas'
#                    prec_vn = 'pr'
#                    elev_vn = 'orog'
#                    lat_vn = 'lat'
#                    lon_vn = 'lon'
#                    time_vn = 'time'
#                    # Variable filenames
#                    temp_fn = temp_vn + '_mon_' + gcm_name + '_' + scenario + '_r1i1p1_native.nc'
#                    prec_fn = prec_vn + '_mon_' + gcm_name + '_' + scenario + '_r1i1p1_native.nc'
#                    elev_fn = elev_vn + '_fx_' + gcm_name + '_' + scenario + '_r0i0p0.nc'
#                    # Variable filepaths
#                    var_fp = pygem_prms.cmip5_fp_var_prefix + scenario + pygem_prms.cmip5_fp_var_ending
#                    fx_fp = pygem_prms.cmip5_fp_fx_prefix + scenario + pygem_prms.cmip5_fp_fx_ending
#                    # Extra information
#                    timestep = pygem_prms.timestep
#                        
#                ds = xr.open_dataset(var_fp + temp_fn)
#                
#                start_idx = (np.where(pd.Series(ds[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
#                                      dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
#                end_idx = (np.where(pd.Series(ds[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
#                                    dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]              
#        
#                time_series = pd.Series(ds[time_vn][start_idx:end_idx+1])
#    
#                if 'expver' in ds.keys():
#                    expver_idx = 0
#                    temp_all = ds[temp_vn][start_idx:end_idx+1, expver_idx, :, :].values
#                else:
#                    temp_all = ds[temp_vn][start_idx:end_idx+1, :, :].values
#                    
#                years = np.array([x.year for x in time_series][::12])
#                    
#                temp_global_mean_monthly = temp_all.mean(2).mean(1)
#                temp_global_mean_annual = temp_global_mean_monthly.reshape(-1,12).mean(axis=1)
#                
#                temp_global_mean_preindustrial = temp_global_mean_annual[0:20].mean()
#                
#                # SROCC Summary Policy Makers: future global mean surface air temperature relative to 1986-2005 
#                #   + 0.63 to account fo changes from 1850-1986
#                temp_global_mean_deviation = 0.63 + temp_global_mean_annual - temp_global_mean_preindustrial
#                
#                global_mean_deviation_degC = temp_global_mean_deviation[-20:].mean()
#                
#                print(gcm_name, scenario, global_mean_deviation_degC)
#                
#                temp_dev_df.loc[ncount,'GCM'] = gcm_name
#                temp_dev_df.loc[ncount,'Scenario'] = scenario
#                temp_dev_df.loc[ncount,'global_mean_deviation_degC'] = global_mean_deviation_degC
#                
#                ncount += 1
#        temp_dev_df.to_csv(csv_fp + temp_dev_fn, index=False)
    
#    #%% ----- FIGURE SHOWING DISTRIBUTION OF TEMPERATURES FOR VARIOUS RCP/SSP SCENARIOS -----
#    # Set up processing
#    reg_vol_all = {}
#    reg_vol_all_bwl = {}
#    reg_area_all = {} 
#    reg_runoff_all = {}
#    
#    # Set up Global region
#    reg_vol_all['all'] = {}
#    reg_vol_all_bwl['all'] = {}
#    reg_area_all['all'] = {}
#    reg_runoff_all['all'] = {}
#    for rcp in rcps:
#        reg_vol_all['all'][rcp] = {}
#        reg_vol_all_bwl['all'][rcp] = {}
#        reg_area_all['all'][rcp] = {}
#        reg_runoff_all['all'][rcp] = {}
#        if 'rcp' in rcp:
#            gcm_names = gcm_names_rcps
#        elif 'ssp' in rcp:
#            gcm_names = gcm_names_ssps
#        for gcm_name in gcm_names:
#            reg_vol_all['all'][rcp][gcm_name] = None
#            reg_vol_all_bwl['all'][rcp][gcm_name] = None
#            reg_area_all['all'][rcp][gcm_name] = None
#            reg_runoff_all['all'][rcp][gcm_name] = None
#            
#    for reg in regions:
#    
#        reg_vol_all[reg] = {}
#        reg_vol_all_bwl[reg] = {}
#        reg_area_all[reg] = {}
#        reg_runoff_all[reg] = {}
#        
#        for rcp in rcps:
#            reg_vol_all[reg][rcp] = {}
#            reg_vol_all_bwl[reg][rcp] = {}
#            reg_area_all[reg][rcp] = {}
#            reg_runoff_all[reg][rcp] = {}
#            
#            if 'rcp' in rcp:
#                gcm_names = gcm_names_rcps
#            elif 'ssp' in rcp:
#                gcm_names = gcm_names_ssps
#                
#            for gcm_name in gcm_names:
#                
#                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
#                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
#                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
#                # Region string prefix
#                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
#                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
#                
#                # Filenames
#                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
#                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
#                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
#                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
#                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
#                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
#                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
#                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
#                
#                 # Volume
#                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
#                    reg_vol_annual = pickle.load(f)
#                # Volume below sea level
#                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
#                    reg_vol_annual_bwl = pickle.load(f)
#                # Area 
#                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
#                    reg_area_annual = pickle.load(f)
#                # Runoff: moving-gauged
#                with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
#                    reg_runoff_monthly_moving = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
#                    watershed_runoff_monthly_moving = pickle.load(f)
#                # Runoff: fixed-gauged
#                with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
#                    reg_runoff_monthly_fixed = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
#                    watershed_runoff_monthly_fixed = pickle.load(f)                   
#                    
#                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
#                if reg_vol_annual_bwl is None:
#                    reg_vol_all_bwl[reg][rcp][gcm_name] = np.zeros(reg_vol_annual.shape)
#                else:
#                    reg_vol_all_bwl[reg][rcp][gcm_name] = reg_vol_annual_bwl
#                reg_area_all[reg][rcp][gcm_name] = reg_area_annual  
#                reg_runoff_all[reg][rcp][gcm_name] = reg_runoff_monthly_fixed
#                
#                if reg_vol_all['all'][rcp][gcm_name] is None:
#                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
#                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
#                    reg_area_all['all'][rcp][gcm_name] = reg_area_all[reg][rcp][gcm_name]
#                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all[reg][rcp][gcm_name]
#                else:
#                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all['all'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
#                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl['all'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
#                    reg_area_all['all'][rcp][gcm_name] = reg_area_all['all'][rcp][gcm_name] + reg_area_all[reg][rcp][gcm_name]
#                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all['all'][rcp][gcm_name] + reg_runoff_all[reg][rcp][gcm_name]
#
#    regions.append('all')
#    
#    #%% SLR for each region, gcm, and scenario
#    regions_ordered = ['all',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#    years = np.arange(2000,2101+1)
#    normyear_idx = np.where(years == normyear)[0][0]
#    if not 'SLR_mmSLE-all' in temp_dev_df.columns:
#        for reg in regions_ordered:
#            temp_dev_df['SLR_mmSLE-' + str(reg)] = np.nan
#            temp_dev_df['SLR_mmSLE_max-' + str(reg)] = np.nan
#        for scenario in rcps:
#            if 'rcp' in scenario:
#                gcm_names = gcm_names_rcps
#            elif 'ssp' in scenario:
#                gcm_names = gcm_names_ssps
#            
#            for gcm_name in gcm_names:
#                
#                ncount = temp_dev_df.loc[(temp_dev_df.Scenario==scenario) & (temp_dev_df.GCM==gcm_name)].index.values[0]
#                
#                for reg in regions_ordered:
#                    reg_vol = reg_vol_all[reg][scenario][gcm_name]
#                    reg_vol_bsl = reg_vol_all_bwl[reg][scenario][gcm_name]
#                    
#                    # Cumulative Sea-level change [mm SLE]
#                    #  - accounts for water from glaciers replacing the ice that is below sea level as well
#                    #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
#                    reg_slr = (-1*(((reg_vol[1:] - reg_vol[0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
#                               (reg_vol_bsl[1:] - reg_vol_bsl[0:-1])) / pygem_prms.area_ocean * 1000))
#                    reg_slr_cum_raw = np.cumsum(reg_slr)
#                    reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[normyear_idx]
#                    
#                    reg_slr_max = reg_vol[normyear_idx] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000
#                    
#                    temp_dev_df.loc[ncount,'SLR_mmSLE-' + str(reg)] = reg_slr_cum[-1]
#                    temp_dev_df.loc[ncount,'SLR_mmSLE_max-' + str(reg)] = reg_slr_max
#
#        temp_dev_df.to_csv(csv_fp + temp_dev_fn, index=False)
#                    
#    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE -----
#    fig = plt.figure()
#    ax = fig.add_axes([0,0,1,1])
#    ax.patch.set_facecolor('none')
#        
#    for scenario in rcps:
#        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
#        
#        if scenario.startswith('ssp'):
#            marker = 'o'
#        else:
#            marker = 'd'
#        ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-all'], 
#                linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
#    
#    ax.set_xlabel('Global mean temperature change (-)')
#    ax.set_ylabel('Sea level rise (mm SLE)', size=12)
##    ax.set_xlim(startyear, endyear)
#    ax.xaxis.set_major_locator(MultipleLocator(1))
#    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
##    ax.set_ylim(0,1.1)
#    ax.yaxis.set_major_locator(MultipleLocator(25))
#    ax.yaxis.set_minor_locator(MultipleLocator(5))
#    ax.tick_params(axis='both', which='major', direction='inout', right=True)
#    ax.tick_params(axis='both', which='minor', direction='in', right=True)
#    
#    ax.legend(labels=['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
#              loc=(0.02,0.75), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,
#              borderpad=0.1, ncol=2, columnspacing=0.5, frameon=True)
#    
#    # Save figure
#    fig_fn = 'Temp_vs_SLR-global.png'
#    fig.set_size_inches(4,3)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
#
#    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE REGIONAL -----
#    fig = plt.figure()
#    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
#    ax1 = fig.add_subplot(gs[0:2,0:2])
#    ax2 = fig.add_subplot(gs[0,3])
#    ax3 = fig.add_subplot(gs[1,2])
#    ax4 = fig.add_subplot(gs[1,3])
#    ax5 = fig.add_subplot(gs[2,0])
#    ax6 = fig.add_subplot(gs[2,1])
#    ax7 = fig.add_subplot(gs[2,2])
#    ax8 = fig.add_subplot(gs[2,3])
#    ax9 = fig.add_subplot(gs[3,0])
#    ax10 = fig.add_subplot(gs[3,1])
#    ax11 = fig.add_subplot(gs[3,2])
#    ax12 = fig.add_subplot(gs[3,3])
#    ax13 = fig.add_subplot(gs[4,0])
#    ax14 = fig.add_subplot(gs[4,1])
#    ax15 = fig.add_subplot(gs[4,2])
#    ax16 = fig.add_subplot(gs[4,3])
#    ax17 = fig.add_subplot(gs[5,0])
#    ax18 = fig.add_subplot(gs[5,1])
#    ax19 = fig.add_subplot(gs[5,2])
#    ax20 = fig.add_subplot(gs[5,3])
#    
#    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
#    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        
#        reg = regions_ordered[nax]            
#    
#        slr_max = 0
#        for scenario in rcps:
#            temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
#            if scenario.startswith('ssp'):
#                marker = 'o'
#            else:
#                marker = 'd'
#            ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-' + str(reg)], 
#                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
#            
#            if temp_dev_df_subset['SLR_mmSLE_max-' + str(reg)].mean() > slr_max:
#                slr_max = temp_dev_df_subset['SLR_mmSLE_max-' + str(reg)].mean()
#        
#        ax.hlines(slr_max, 0, 7, color='k', linewidth=0.5)
#            
#        
#        ax.set_xlim(0,7)
#        ax.xaxis.set_major_locator(MultipleLocator(1))
#        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#        ax.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#            
#        if reg in ['all']:
#            ax.set_ylim(0,270)
#            ax.yaxis.set_major_locator(MultipleLocator(50))
#            ax.yaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [19, 3, 1]:
#            ax.set_ylim(0,52)
#            ax.yaxis.set_major_locator(MultipleLocator(10))
#            ax.yaxis.set_minor_locator(MultipleLocator(5))    
#        elif reg in [5, 9]:
#            ax.set_ylim(0,33)
#            ax.yaxis.set_major_locator(MultipleLocator(10))
#            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
#        elif reg in [4, 7]:
#            ax.set_ylim(0,23)
#            ax.yaxis.set_major_locator(MultipleLocator(10))
#            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
#        elif reg in [17, 13, 6, 14]:
#            ax.set_ylim(0,13)
#            ax.yaxis.set_major_locator(MultipleLocator(5))
#            ax.yaxis.set_minor_locator(MultipleLocator(1))
#        elif reg in [15, 2]:
#            ax.set_ylim(0,2.7)
#            ax.yaxis.set_major_locator(MultipleLocator(1))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
#        elif reg in [8]:
#            ax.set_ylim(0,0.8)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
#        elif reg in [10, 11, 16]:
#            ax.set_ylim(0,0.32)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
#        elif reg in [12, 18]:
#            ax.set_ylim(0,0.22)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
#        
#        if nax == 0:
#            label_height=1.06
#        else:
#            label_height=1.14
#        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                verticalalignment='top', transform=ax.transAxes)
#        ax.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        
#        if nax == 1:
#            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'rcp26' in rcps and len(rcps) == 3:
#                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'ssp126' in rcps and len(rcps) == 4:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
#                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
#                      )
#    fig.text(0.5,0.08,'Global mean temperature change (-)', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
#    fig.text(0.07,0.5,'Sea level rise (mm SLE)', size=12, horizontalalignment='center', verticalalignment='top', rotation=90)
#    
#    # Save figure
#    fig_fn = 'Temp_vs_SLR-regional.png'
#    fig.set_size_inches(8.5,11)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
#%% ===== MASS BALANCE SENSITIVITY FIGURES =====     
if option_sensitivity_figs:
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']

    mb_sens_fn = 'mb_sensitivity_test.csv'
    
    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    
    fn_reg_temp_all = 'reg_temp_all.pkl'
    fn_reg_temp_all_monthly = 'reg_temp_all_monthly.pkl'
    fn_reg_prec_all = 'reg_prec_all.pkl'
    fn_reg_prec_all_monthly = 'reg_prec_all_monthly.pkl'
            
    
    if os.path.exists(pickle_fp + fn_reg_temp_all):
        
        with open(pickle_fp + fn_reg_temp_all, 'rb') as f:
            reg_temp_all = pickle.load(f)
        with open(pickle_fp + fn_reg_temp_all_monthly, 'rb') as f:
            reg_temp_all_monthly = pickle.load(f)
        with open(pickle_fp + fn_reg_prec_all, 'rb') as f:
            reg_prec_all = pickle.load(f)
        with open(pickle_fp + fn_reg_prec_all_monthly, 'rb') as f:
            reg_prec_all_monthly = pickle.load(f)
        
    else:
    
        reg_temp_all = {}
        reg_temp_all_monthly = {}
        reg_prec_all = {}
        reg_prec_all_monthly = {}
        
        # Set up regions
        for reg in regions:
            reg_temp_all[reg] = {}
            reg_temp_all_monthly[reg] = {}
            reg_prec_all[reg] = {}
            reg_prec_all_monthly[reg] = {}
    
            for rcp in rcps:
                reg_temp_all[reg][rcp] = {}
                reg_temp_all_monthly[reg][rcp] = {}
                reg_prec_all[reg][rcp] = {}
                reg_prec_all_monthly[reg][rcp] = {}
        
        # Set up Global and All regions
        #  - global includes every pixel
        #  - all includes only glacierized pixels
        reg_temp_all['all'] = {}
        reg_temp_all_monthly['all'] = {}
        reg_prec_all['all'] = {}
        reg_prec_all_monthly['all'] = {}
        reg_temp_all['global'] = {}
        reg_temp_all_monthly['global'] = {}
        reg_prec_all['global'] = {}
        reg_prec_all_monthly['global'] = {}
    
        for rcp in rcps:
            reg_temp_all['all'][rcp] = {}
            reg_temp_all_monthly['all'][rcp] = {}
            reg_prec_all['all'][rcp] = {}
            reg_prec_all_monthly['all'][rcp] = {}
            reg_temp_all['global'][rcp] = {}
            reg_temp_all_monthly['global'][rcp] = {}
            reg_prec_all['global'][rcp] = {}
            reg_prec_all_monthly['global'][rcp] = {}
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
        
            dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0)
            
            scenario = rcp
            
            if 'rcp' in scenario:
                gcm_names = gcm_names_rcps
            elif 'ssp' in scenario:
                gcm_names = gcm_names_ssps
            
            for gcm_name in gcm_names:
    
                if scenario.startswith('ssp'):
                    # Variable names
                    temp_vn = 'tas'
                    prec_vn = 'pr'
                    elev_vn = 'orog'
                    lat_vn = 'lat'
                    lon_vn = 'lon'
                    time_vn = 'time'
                    # Variable filenames
                    temp_fn = gcm_name + '_' + scenario + '_r1i1p1f1_' + temp_vn + '.nc'
                    prec_fn = gcm_name + '_' + scenario + '_r1i1p1f1_' + prec_vn + '.nc'
                    elev_fn = gcm_name + '_' + elev_vn + '.nc'
                    # Variable filepaths
                    var_fp = pygem_prms.cmip6_fp_prefix + gcm_name + '/'
                    fx_fp = pygem_prms.cmip6_fp_prefix + gcm_name + '/'
                    # Extra information
                    timestep = pygem_prms.timestep
                        
                elif scenario.startswith('rcp'):
                    # Variable names
                    temp_vn = 'tas'
                    prec_vn = 'pr'
                    elev_vn = 'orog'
                    lat_vn = 'lat'
                    lon_vn = 'lon'
                    time_vn = 'time'
                    # Variable filenames
                    temp_fn = temp_vn + '_mon_' + gcm_name + '_' + scenario + '_r1i1p1_native.nc'
                    prec_fn = prec_vn + '_mon_' + gcm_name + '_' + scenario + '_r1i1p1_native.nc'
                    elev_fn = elev_vn + '_fx_' + gcm_name + '_' + scenario + '_r0i0p0.nc'
                    # Variable filepaths
                    var_fp = pygem_prms.cmip5_fp_var_prefix + scenario + pygem_prms.cmip5_fp_var_ending
                    fx_fp = pygem_prms.cmip5_fp_fx_prefix + scenario + pygem_prms.cmip5_fp_fx_ending
                    # Extra information
                    timestep = pygem_prms.timestep
                        
                ds_temp = xr.open_dataset(var_fp + temp_fn)
                ds_prec = xr.open_dataset(var_fp + prec_fn)
                
                start_idx = (np.where(pd.Series(ds_temp[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                                      dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
                end_idx = (np.where(pd.Series(ds_temp[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                                    dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]              
        
                time_series = pd.Series(ds_temp[time_vn][start_idx:end_idx+1])
                years = np.array([x.year for x in time_series][::12])
    
                # Global statistics
                if 'expver' in ds_temp.keys():
                    expver_idx = 0
                    temp_all = ds_temp[temp_vn][start_idx:end_idx+1, expver_idx, :, :].values
                    prec_all = ds_prec[prec_vn][start_idx:end_idx+1, expver_idx, :, :].values
                else:
                    temp_all = ds_temp[temp_vn][start_idx:end_idx+1, :, :].values
                    prec_all = ds_prec[prec_vn][start_idx:end_idx+1, :, :].values
                    
                # Correct precipitaiton to monthly
                if 'units' in ds_prec[prec_vn].attrs and ds_prec[prec_vn].attrs['units'] == 'kg m-2 s-1':  
                    # Convert from kg m-2 s-1 to m day-1
                    prec_all = prec_all/1000*3600*24
                    #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
                # Else check the variables units
                else:
                    print('Check units of precipitation from GCM is meters per day.')
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    prec_all = prec_all * dates_table['daysinmonth'].values[:,np.newaxis,np.newaxis]
                    
                temp_global_mean_monthly = temp_all.mean(2).mean(1)
                temp_global_mean_annual = temp_global_mean_monthly.reshape(-1,12).mean(axis=1)
                
                prec_global_mean_monthly = prec_all.mean(2).mean(1)
                prec_global_sum_annual = prec_global_mean_monthly.reshape(-1,12).sum(axis=1)
                
                reg_temp_all['global'][rcp][gcm_name] = temp_global_mean_annual
                reg_temp_all_monthly['global'][rcp][gcm_name] = temp_global_mean_monthly
                reg_prec_all['global'][rcp][gcm_name] = prec_global_sum_annual
                reg_prec_all_monthly['global'][rcp][gcm_name] = prec_global_mean_monthly
                
                # Regional statistics
                for nreg, reg in enumerate(regions):
                    # Glaciers
                    fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
                    with open(pickle_fp + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'rb') as f:
                        glacno_list = pickle.load(f)
                        
                    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
                        
                    #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
                    #  extract the position's value as opposed to having an array
                    lat_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lat_colname].values[:,np.newaxis] - 
                                          ds_temp.variables[lat_vn][:].values).argmin(axis=1))
                    lon_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lon_colname].values[:,np.newaxis] - 
                                          ds_temp.variables[lon_vn][:].values).argmin(axis=1))
                    # Find unique latitude/longitudes
                    latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
                    latlon_nearidx_unique = list(set(latlon_nearidx))
                    # Create dictionary of time series for each unique latitude/longitude
                    temp_reg_latlon = {}
                    prec_reg_latlon = {}
                    for latlon in latlon_nearidx_unique:                
                        if 'expver' in ds_temp.keys():
                            expver_idx = 0
                            temp_reg_latlon[latlon] = ds_temp[temp_vn][start_idx:end_idx+1, expver_idx, latlon[0], latlon[1]].values
                            prec_reg_latlon[latlon] = ds_prec[prec_vn][start_idx:end_idx+1, expver_idx, latlon[0], latlon[1]].values
                        else:
                            temp_reg_latlon[latlon] = ds_temp[temp_vn][start_idx:end_idx+1, latlon[0], latlon[1]].values
                            prec_reg_latlon[latlon] = ds_prec[prec_vn][start_idx:end_idx+1, latlon[0], latlon[1]].values
    
                    # Convert to regional mean
                    temp_reg_all = np.array([temp_reg_latlon[x] for x in latlon_nearidx_unique])
                    temp_reg_mean_monthly = temp_reg_all.mean(axis=0)
                    temp_reg_mean_annual = temp_reg_mean_monthly.reshape(-1,12).mean(axis=1)
                    
                    prec_reg_all = np.array([prec_reg_latlon[x] for x in latlon_nearidx_unique])
                    prec_reg_mean_monthly = prec_reg_all.mean(axis=0)
                    # Correct precipitation to monthly
                    if 'units' in ds_prec[prec_vn].attrs and ds_prec[prec_vn].attrs['units'] == 'kg m-2 s-1':  
                        # Convert from kg m-2 s-1 to m day-1
                        prec_reg_mean_monthly = prec_reg_mean_monthly/1000*3600*24
                        #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
                    else:
                        print('Check units of precipitation from GCM is meters per day.')
                    # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                    if 'daysinmonth' in dates_table.columns:
                        prec_reg_mean_monthly = prec_reg_mean_monthly * dates_table['daysinmonth'].values
                    prec_reg_sum_annual = prec_reg_mean_monthly.reshape(-1,12).sum(axis=1)
                    
                    # Record data
                    reg_temp_all[reg][rcp][gcm_name] = temp_reg_mean_annual
                    reg_temp_all_monthly[reg][rcp][gcm_name] = temp_reg_mean_monthly
                    reg_prec_all[reg][rcp][gcm_name] = prec_reg_sum_annual
                    reg_prec_all_monthly[reg][rcp][gcm_name] = prec_reg_mean_monthly
                    
                    
                    if nreg == 0:
                        temp_reg_all_raw = temp_reg_all
                        prec_reg_all_raw = prec_reg_all                    
                    else:
                        temp_reg_all_raw = np.concatenate((temp_reg_all_raw, temp_reg_all), axis=0)
                        prec_reg_all_raw = np.concatenate((prec_reg_all_raw, prec_reg_all), axis=0)
                    
                # All glacierized statistics
                temp_reg_all = temp_reg_all_raw
                temp_reg_mean_monthly = temp_reg_all.mean(axis=0)
                temp_reg_mean_annual = temp_reg_mean_monthly.reshape(-1,12).mean(axis=1)
                
                prec_reg_all = prec_reg_all_raw
                prec_reg_mean_monthly = prec_reg_all.mean(axis=0)
                # Correct precipitation to monthly
                if 'units' in ds_prec[prec_vn].attrs and ds_prec[prec_vn].attrs['units'] == 'kg m-2 s-1':  
                    # Convert from kg m-2 s-1 to m day-1
                    prec_reg_mean_monthly = prec_reg_mean_monthly/1000*3600*24
                    #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
                else:
                    print('Check units of precipitation from GCM is meters per day.')
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    prec_reg_mean_monthly = prec_reg_mean_monthly * dates_table['daysinmonth'].values
                prec_reg_sum_annual = prec_reg_mean_monthly.reshape(-1,12).sum(axis=1)
                
                # Record data
                reg_temp_all['all'][rcp][gcm_name] = temp_reg_mean_annual
                reg_temp_all_monthly['all'][rcp][gcm_name] = temp_reg_mean_monthly
                reg_prec_all['all'][rcp][gcm_name] = prec_reg_sum_annual
                reg_prec_all_monthly['all'][rcp][gcm_name] = prec_reg_mean_monthly
        
        
        with open(pickle_fp + fn_reg_temp_all, 'wb') as f:
            pickle.dump(reg_temp_all, f)
        with open(pickle_fp + fn_reg_temp_all_monthly, 'wb') as f:
            pickle.dump(reg_temp_all_monthly, f)
        with open(pickle_fp + fn_reg_prec_all, 'wb') as f:
            pickle.dump(reg_prec_all, f)
        with open(pickle_fp + fn_reg_prec_all_monthly, 'wb') as f:
            pickle.dump(reg_prec_all_monthly, f)

    #%% ----- FIGURE SHOWING DISTRIBUTION OF TEMPERATURES FOR VARIOUS RCP/SSP SCENARIOS -----
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_bwl = {}
    reg_area_all = {} 
    reg_runoff_all = {}
    reg_mb_all_monthly = {}
    
    # Set up Global region
    reg_vol_all['all'] = {}
    reg_vol_all_bwl['all'] = {}
    reg_area_all['all'] = {}
    reg_runoff_all['all'] = {}
    reg_mb_all_monthly['all'] = {}
    for rcp in rcps:
        reg_vol_all['all'][rcp] = {}
        reg_vol_all_bwl['all'][rcp] = {}
        reg_area_all['all'][rcp] = {}
        reg_runoff_all['all'][rcp] = {}
        reg_mb_all_monthly['all'][rcp] = {}
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            reg_vol_all['all'][rcp][gcm_name] = None
            reg_vol_all_bwl['all'][rcp][gcm_name] = None
            reg_area_all['all'][rcp][gcm_name] = None
            reg_runoff_all['all'][rcp][gcm_name] = None
            reg_mb_all_monthly['all'][rcp][gcm_name] = None
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_bwl[reg] = {}
        reg_area_all[reg] = {}
        reg_runoff_all[reg] = {}
        reg_mb_all_monthly[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_bwl[reg][rcp] = {}
            reg_area_all[reg][rcp] = {}
            reg_runoff_all[reg][rcp] = {}
            reg_mb_all_monthly[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_reg_mb_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                
                 # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                # Volume below sea level
                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                    reg_vol_annual_bwl = pickle.load(f)
                # Area 
                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                    reg_area_annual = pickle.load(f)
                # Mass balance monthly
                with open(pickle_fp_reg + fn_reg_mb_monthly, 'rb') as f:
                    reg_mb_monthly = pickle.load(f)
                # Runoff: moving-gauged
                with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                    reg_runoff_monthly_moving = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                    watershed_runoff_monthly_moving = pickle.load(f)
                # Runoff: fixed-gauged
                with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                    reg_runoff_monthly_fixed = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                    watershed_runoff_monthly_fixed = pickle.load(f)                   
                    
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                if reg_vol_annual_bwl is None:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = np.zeros(reg_vol_annual.shape)
                else:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = reg_vol_annual_bwl
                reg_area_all[reg][rcp][gcm_name] = reg_area_annual  
                reg_mb_all_monthly[reg][rcp][gcm_name] = reg_mb_monthly  
                reg_runoff_all[reg][rcp][gcm_name] = reg_runoff_monthly_fixed
                
                if reg_vol_all['all'][rcp][gcm_name] is None:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all[reg][rcp][gcm_name]
                    reg_mb_all_monthly['all'][rcp][gcm_name] = reg_mb_all_monthly[reg][rcp][gcm_name]
                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all[reg][rcp][gcm_name]
                else:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all['all'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl['all'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all['all'][rcp][gcm_name] + reg_area_all[reg][rcp][gcm_name]
                    reg_mb_all_monthly['all'][rcp][gcm_name] = reg_mb_all_monthly['all'][rcp][gcm_name] + reg_mb_all_monthly[reg][rcp][gcm_name]
                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all['all'][rcp][gcm_name] + reg_runoff_all[reg][rcp][gcm_name]

    regions.append('all')
    
    # Add the annual mass balance (separately to get global value properly)
    # Mass balance monthly is m3we, while mass balance below is in mm w.e. yr-1
    reg_mb_all = {}
    for reg in regions:
        reg_mb_all[reg] = {}
        
        for rcp in rcps:
            reg_mb_all[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                # Specific mass balance
                reg_vol = reg_vol_all[reg][rcp][gcm_name]
                reg_area = reg_area_all[reg][rcp][gcm_name]
                reg_mass = reg_vol * pygem_prms.density_ice
                reg_mb = (reg_mass[1:] - reg_mass[0:-1]) / reg_area[0:-1]
                
                reg_mb_all[reg][rcp][gcm_name] = reg_mb
                
                # Convert monthly mass balance to same units (note will be slightly off due to averaging time scales)
                # kg m-2 yr-1, which equals mm w.e. yr-1
                reg_mb_monthly = reg_mb_all_monthly[reg][rcp][gcm_name]
                reg_mb_all_monthly[reg][rcp][gcm_name] = reg_mb_monthly / np.repeat(reg_area[0:-1],12) * 1000
                
    #%%  ===== MASS BALANCE SENSITIVITY =====
    for reg in regions:
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                years_present = [2000,2020]
                years_future = [2080,2100]
                idx_present = [np.where(years==years_present[0])[0][0],
                               np.where(years==years_present[1])[0][0]]
                idx_future = [np.where(years==years_future[0])[0][0],
                              np.where(years==years_future[1])[0][0]]
                
                # Mass balance sensitivity
                reg_mb = reg_mb_all[reg][rcp][gcm_name]
                reg_temp = reg_temp_all[reg][rcp][gcm_name]
                
                reg_mb_sens = ((np.mean(reg_mb[idx_future[0]:idx_future[1]]) - np.mean(reg_mb[idx_present[0]:idx_present[1]])) / 
                               (np.mean(reg_temp[idx_future[0]:idx_future[1]]) - np.mean(reg_temp[idx_present[0]:idx_present[1]])))
                
                print(reg, rcp, gcm_name, np.round(reg_mb_sens))
                
                
    print('pickle datasets!')
                
    #%%
#    fig = plt.figure()
#    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
#    ax1 = fig.add_subplot(gs[0:2,0:2])
#    ax2 = fig.add_subplot(gs[0,3])
#    ax3 = fig.add_subplot(gs[1,2])
#    ax4 = fig.add_subplot(gs[1,3])
#    ax5 = fig.add_subplot(gs[2,0])
#    ax6 = fig.add_subplot(gs[2,1])
#    ax7 = fig.add_subplot(gs[2,2])
#    ax8 = fig.add_subplot(gs[2,3])
#    ax9 = fig.add_subplot(gs[3,0])
#    ax10 = fig.add_subplot(gs[3,1])
#    ax11 = fig.add_subplot(gs[3,2])
#    ax12 = fig.add_subplot(gs[3,3])
#    ax13 = fig.add_subplot(gs[4,0])
#    ax14 = fig.add_subplot(gs[4,1])
#    ax15 = fig.add_subplot(gs[4,2])
#    ax16 = fig.add_subplot(gs[4,3])
#    ax17 = fig.add_subplot(gs[5,0])
#    ax18 = fig.add_subplot(gs[5,1])
#    ax19 = fig.add_subplot(gs[5,2])
#    ax20 = fig.add_subplot(gs[5,3])
#    
#    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
#    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        
#        reg = regions_ordered[nax]
#
#        for rcp in rcps:
#            
#            # Median and absolute median deviation
#            reg_vol = ds_multigcm_vol[reg][rcp]
#            reg_mass = reg_vol * pygem_prms.density_ice
#            reg_area = ds_multigcm_area[reg][rcp]
#
#            # Specific mass change rate
#            reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
#            reg_mb_med = np.median(reg_mb, axis=0)
#            reg_mb_mad = median_abs_deviation(reg_mb, axis=0)
#            
#            ax.plot(years[0:-1], reg_mb_med / 1000, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
#                    linewidth=1, zorder=4, label=rcp)
#            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_mb_med + 1.96*reg_mb_mad) / 1000, 
#                                (reg_mb_med - 1.96*reg_mb_mad) / 1000, 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
#        
#        if ax in [ax1,ax5,ax9,ax13,ax17]:
##            ax.set_ylabel('$\Delta$M/$\Delta$t (m w.e. yr$^{-1}$)')
#            ax.set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)')
#        ax.set_xlim(startyear, endyear)
#        ax.xaxis.set_major_locator(MultipleLocator(50))
#        ax.xaxis.set_minor_locator(MultipleLocator(10))
#        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
#        ax.set_ylim(-5.5,0.5)
#        ax.yaxis.set_major_locator(MultipleLocator(1))
#        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
#        if nax == 0:
#            label_height=1.06
#        else:
#            label_height=1.14
#        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                verticalalignment='top', transform=ax.transAxes)
#        ax.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        if nax == 1:
#            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'rcp26' in rcps and len(rcps) == 3:
#                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'ssp126' in rcps and len(rcps) == 4:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
#                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
#                      )
#    # Save figure
#    if 'rcp26' in rcps and 'ssp126' in rcps:
#        scenario_str = 'rcps_ssps'
#    elif 'rcp26' in rcps:
#        scenario_str = 'rcps'
#    elif 'ssp126' in rcps:
#        scenario_str = 'ssps'
#    fig_fn = ('allregions_mb_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
#              '-' + scenario_str + '.png')
#    fig.set_size_inches(8.5,11)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
    #%% SLR for each region, gcm, and scenario
#    regions_ordered = ['all',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#    years = np.arange(2000,2101+1)
#    normyear_idx = np.where(years == normyear)[0][0]
#    if not 'SLR_mmSLE-all' in temp_dev_df.columns:
#        for reg in regions_ordered:
#            temp_dev_df['SLR_mmSLE-' + str(reg)] = np.nan
#            temp_dev_df['SLR_mmSLE_max-' + str(reg)] = np.nan
#        for scenario in rcps:
#            if 'rcp' in scenario:
#                gcm_names = gcm_names_rcps
#            elif 'ssp' in scenario:
#                gcm_names = gcm_names_ssps
#            
#            for gcm_name in gcm_names:
#                
#                ncount = temp_dev_df.loc[(temp_dev_df.Scenario==scenario) & (temp_dev_df.GCM==gcm_name)].index.values[0]
#                
#                for reg in regions_ordered:
#                    reg_vol = reg_vol_all[reg][scenario][gcm_name]
#                    reg_vol_bsl = reg_vol_all_bwl[reg][scenario][gcm_name]
#                    
#                    # Cumulative Sea-level change [mm SLE]
#                    #  - accounts for water from glaciers replacing the ice that is below sea level as well
#                    #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
#                    reg_slr = (-1*(((reg_vol[1:] - reg_vol[0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
#                               (reg_vol_bsl[1:] - reg_vol_bsl[0:-1])) / pygem_prms.area_ocean * 1000))
#                    reg_slr_cum_raw = np.cumsum(reg_slr)
#                    reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[normyear_idx]
#                    
#                    reg_slr_max = reg_vol[normyear_idx] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000
#                    
#                    temp_dev_df.loc[ncount,'SLR_mmSLE-' + str(reg)] = reg_slr_cum[-1]
#                    temp_dev_df.loc[ncount,'SLR_mmSLE_max-' + str(reg)] = reg_slr_max
#
#        temp_dev_df.to_csv(csv_fp + temp_dev_fn, index=False)
#                    
#    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE -----
#    fig = plt.figure()
#    ax = fig.add_axes([0,0,1,1])
#    ax.patch.set_facecolor('none')
#        
#    for scenario in rcps:
#        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
#        
#        if scenario.startswith('ssp'):
#            marker = 'o'
#        else:
#            marker = 'd'
#        ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-all'], 
#                linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
#    
#    ax.set_xlabel('Global mean temperature change (-)')
#    ax.set_ylabel('Sea level rise (mm SLE)', size=12)
##    ax.set_xlim(startyear, endyear)
#    ax.xaxis.set_major_locator(MultipleLocator(1))
#    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
##    ax.set_ylim(0,1.1)
#    ax.yaxis.set_major_locator(MultipleLocator(25))
#    ax.yaxis.set_minor_locator(MultipleLocator(5))
#    ax.tick_params(axis='both', which='major', direction='inout', right=True)
#    ax.tick_params(axis='both', which='minor', direction='in', right=True)
#    
#    ax.legend(labels=['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
#              loc=(0.02,0.75), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,
#              borderpad=0.1, ncol=2, columnspacing=0.5, frameon=True)
#    
#    # Save figure
#    fig_fn = 'Temp_vs_SLR-global.png'
#    fig.set_size_inches(4,3)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
#
#    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE REGIONAL -----
#    fig = plt.figure()
#    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
#    ax1 = fig.add_subplot(gs[0:2,0:2])
#    ax2 = fig.add_subplot(gs[0,3])
#    ax3 = fig.add_subplot(gs[1,2])
#    ax4 = fig.add_subplot(gs[1,3])
#    ax5 = fig.add_subplot(gs[2,0])
#    ax6 = fig.add_subplot(gs[2,1])
#    ax7 = fig.add_subplot(gs[2,2])
#    ax8 = fig.add_subplot(gs[2,3])
#    ax9 = fig.add_subplot(gs[3,0])
#    ax10 = fig.add_subplot(gs[3,1])
#    ax11 = fig.add_subplot(gs[3,2])
#    ax12 = fig.add_subplot(gs[3,3])
#    ax13 = fig.add_subplot(gs[4,0])
#    ax14 = fig.add_subplot(gs[4,1])
#    ax15 = fig.add_subplot(gs[4,2])
#    ax16 = fig.add_subplot(gs[4,3])
#    ax17 = fig.add_subplot(gs[5,0])
#    ax18 = fig.add_subplot(gs[5,1])
#    ax19 = fig.add_subplot(gs[5,2])
#    ax20 = fig.add_subplot(gs[5,3])
#    
#    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
#    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        
#        reg = regions_ordered[nax]            
#    
#        slr_max = 0
#        for scenario in rcps:
#            temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
#            if scenario.startswith('ssp'):
#                marker = 'o'
#            else:
#                marker = 'd'
#            ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-' + str(reg)], 
#                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
#            
#            if temp_dev_df_subset['SLR_mmSLE_max-' + str(reg)].mean() > slr_max:
#                slr_max = temp_dev_df_subset['SLR_mmSLE_max-' + str(reg)].mean()
#        
#        ax.hlines(slr_max, 0, 7, color='k', linewidth=0.5)
#            
#        
#        ax.set_xlim(0,7)
#        ax.xaxis.set_major_locator(MultipleLocator(1))
#        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#        ax.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#            
#        if reg in ['all']:
#            ax.set_ylim(0,270)
#            ax.yaxis.set_major_locator(MultipleLocator(50))
#            ax.yaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [19, 3, 1]:
#            ax.set_ylim(0,52)
#            ax.yaxis.set_major_locator(MultipleLocator(10))
#            ax.yaxis.set_minor_locator(MultipleLocator(5))    
#        elif reg in [5, 9]:
#            ax.set_ylim(0,33)
#            ax.yaxis.set_major_locator(MultipleLocator(10))
#            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
#        elif reg in [4, 7]:
#            ax.set_ylim(0,23)
#            ax.yaxis.set_major_locator(MultipleLocator(10))
#            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
#        elif reg in [17, 13, 6, 14]:
#            ax.set_ylim(0,13)
#            ax.yaxis.set_major_locator(MultipleLocator(5))
#            ax.yaxis.set_minor_locator(MultipleLocator(1))
#        elif reg in [15, 2]:
#            ax.set_ylim(0,2.7)
#            ax.yaxis.set_major_locator(MultipleLocator(1))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
#        elif reg in [8]:
#            ax.set_ylim(0,0.8)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
#        elif reg in [10, 11, 16]:
#            ax.set_ylim(0,0.32)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
#        elif reg in [12, 18]:
#            ax.set_ylim(0,0.22)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
#        
#        if nax == 0:
#            label_height=1.06
#        else:
#            label_height=1.14
#        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                verticalalignment='top', transform=ax.transAxes)
#        ax.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#        
#        
#        if nax == 1:
#            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'rcp26' in rcps and len(rcps) == 3:
#                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
#            elif 'ssp126' in rcps and len(rcps) == 4:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
#                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
#                      )
#    fig.text(0.5,0.08,'Global mean temperature change (-)', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
#    fig.text(0.07,0.5,'Sea level rise (mm SLE)', size=12, horizontalalignment='center', verticalalignment='top', rotation=90)
#    
#    # Save figure
#    fig_fn = 'Temp_vs_SLR-regional.png'
#    fig.set_size_inches(8.5,11)
#    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)


#%%
if option_process_data_wcalving:
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_calving/'
    
    pickle_fp_land = '/Users/drounce/Documents/HiMAT/spc_backup/analysis/pickle/'
    csv_fp_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_csv/'
    csv_fp_calving = netcdf_fp_cmip5 + '_csv/'
    if not os.path.exists(csv_fp_calving):
        os.makedirs(csv_fp_calving)
    
    overwrite_pickle = False
    
    grouping = 'all'

    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
    
        
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')

        # ===== EXPORT RESULTS =====
        success_fullfn = csv_fp + 'CMIP5_success.csv'
        success_cns = ['O1Region', 'count_success', 'count', 'count_%', 'reg_area_km2_success', 'reg_area_km2', 'reg_area_%']
        success_df_single = pd.DataFrame(np.zeros((1,len(success_cns))), columns=success_cns)
        success_df_single.loc[0,:] = [reg, main_glac_rgi.shape[0], main_glac_rgi_all.shape[0],
                                      np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,2),
                                      np.round(main_glac_rgi.Area.sum(),2), np.round(main_glac_rgi_all.Area.sum(),2),
                                      np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,2)]
        if os.path.exists(success_fullfn):
            success_df = pd.read_csv(success_fullfn)
            
            # Add or overwrite existing file
            success_idx = np.where((success_df.O1Region == reg))[0]
            if len(success_idx) > 0:
                success_df.loc[success_idx,:] = success_df_single.values
            else:
                success_df = pd.concat([success_df, success_df_single], axis=0)
                
        else:
            success_df = success_df_single
            
        success_df = success_df.sort_values('O1Region', ascending=True)
        success_df.reset_index(inplace=True, drop=True)
        success_df.to_csv(success_fullfn, index=False)       

        #%%         
        # Pickle datasets (LAND GLACIERS)
        # Glacier list
        fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
        with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'rb') as f:
            glacno_list_land = pickle.load(f)
        
        main_glac_rgi_land = modelsetup.selectglaciersrgitable(glac_no=glacno_list_land)
        
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        main_glac_rgi_land['CenLon_round'] = np.floor(main_glac_rgi_land.CenLon.values/degree_size) * degree_size
        main_glac_rgi_land['CenLat_round'] = np.floor(main_glac_rgi_land.CenLat.values/degree_size) * degree_size
        deg_groups = main_glac_rgi_land.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi_land.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi_land.loc[x,'CenLon_round'], main_glac_rgi_land.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi_land))]
        main_glac_rgi_land['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi_land['deg_id'] = main_glac_rgi_land.CenLon_CenLat.map(deg_dict)
        
        # River Basin
        watershed_dict_fn = pygem_prms.main_directory + '/../qgis_datasets/rgi60_watershed_dict.csv'
        watershed_csv = pd.read_csv(watershed_dict_fn)
        watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
        main_glac_rgi_land['watershed'] = main_glac_rgi_land.RGIId.map(watershed_dict)
        if len(np.where(main_glac_rgi_land.watershed.isnull())[0]) > 0:
            main_glac_rgi_land.loc[np.where(main_glac_rgi_land.watershed.isnull())[0],'watershed'] = 'nan'
        
        #%%
        # Unique Groups
        # O2 Regions
        unique_regO2s = np.unique(main_glac_rgi_land['O2Region'])
        
        # Degrees
        if main_glac_rgi_land['deg_id'].isnull().all():
            unique_degids = None
        else:
            unique_degids = np.unique(main_glac_rgi_land['deg_id'])
            
        print('# degids:', len(unique_degids))
        
        # Watersheds
        if main_glac_rgi_land['watershed'].isnull().all():
            unique_watersheds = None
        else:
            unique_watersheds = np.unique(main_glac_rgi_land['watershed'])

        # Elevation bins
        elev_bin_size = 10
        zmax = int(np.ceil(main_glac_rgi_land.Zmax.max() / elev_bin_size) * elev_bin_size) + 500
        elev_bins = np.arange(0,zmax,elev_bin_size)
        elev_bins = np.insert(elev_bins, 0, -1000)
        

#        # O2Region dict
#        fn_unique_regO2s = 'R' + str(reg) + '_unique_regO2s.pkl'
#        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_regO2s, 'wb') as f:
#            pickle.dump(unique_regO2s, f)      
#        # Watershed dict
#        fn_unique_watersheds = 'R' + str(reg) + '_unique_watersheds.pkl'
#        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_watersheds, 'wb') as f:
#            pickle.dump(unique_watersheds, f) 
#        # Degree ID dict
#        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
#        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_degids, 'wb') as f:
#            pickle.dump(unique_degids, f)
#        
#        fn_elev_bins = 'R' + str(reg) + '_elev_bins.pkl'
#        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_elev_bins, 'wb') as f:
#            pickle.dump(elev_bins, f)
        
                
        glacno_list_good = []     
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # Tidewater glaciers
                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                netcdf_fp_binned_land = netcdf_fp_cmip5_land + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats_land = netcdf_fp_cmip5_land + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- Process glacier volume and losses -----
                reg_vol_annual_fn = str(reg).zfill(2) + '_' + gcm_name + '_' + rcp + '_glac_vol_annual.csv'
                # Load data
                reg_glac_vol_annual_df = pd.read_csv(csv_fp_land + reg_vol_annual_fn)
                rgiids_raw = list(reg_glac_vol_annual_df.values[:,0])
                rgiids = [str(reg) + '.' + str(int(np.round((x-reg)*1e5))).zfill(5) for x in rgiids_raw]
                # If doesn't exist, process data
                if not os.path.exists(csv_fp_calving + reg_vol_annual_fn) or overwrite_pickle:
                    reg_glac_vol_annual_gcm_land = reg_glac_vol_annual_df.values[:,1:]
                    
                    # Load calving data
                    netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                    
                    netcdf_fns = []
                    for i in os.listdir(netcdf_fp_stats):
                        if i.endswith('_all.nc'):
                            netcdf_fns.append(i)
                    netcdf_fns = sorted(netcdf_fns)
                    rgiids_calving = [x.split('_')[0] for x in netcdf_fns]
                    
                    reg_glac_vol_annual_gcm_calving = np.copy(reg_glac_vol_annual_gcm_land)
                    for nglac, netcdf_fn in enumerate(netcdf_fns):
                        rgiid = rgiids_calving[nglac]
                        if rgiid in rgiids:
                            ds_stats = xr.open_dataset(netcdf_fp_stats + netcdf_fn)
                            
                            # Upload new data
                            reg_glac_idx = rgiids.index(rgiid)
                            reg_glac_vol_annual_gcm_calving[reg_glac_idx,:] = ds_stats.glac_volume_annual.values[0,:]
                    
                    # Export new file
                    reg_glac_vol_annual_df_calving = pd.DataFrame.copy(reg_glac_vol_annual_df, deep=True)
                    reg_glac_vol_annual_df_calving.iloc[:,1:] = reg_glac_vol_annual_gcm_calving
                    reg_glac_vol_annual_df_calving.to_csv(csv_fp_calving + reg_vol_annual_fn, index=False)
            
                # ----- Make sure only processing glaciers that have land simulations -----
                glacno_list_gcmrcp = rgiids
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list_good) == 0:
                    glacno_list_good = glacno_list_gcmrcp
                else:
                    glacno_list_good = list(set(glacno_list_good).intersection(glacno_list_gcmrcp))
                glacno_list_good = sorted(glacno_list_good)
                   
                
        print('# glaciers:', len(glacno_list_good))
        # Reset land bins based on the rgiids (for RCPs in regions 5, 7, 9; there are slight differences in deg_ids due to missing glaciers)
        main_glac_rgi_land = modelsetup.selectglaciersrgitable(glac_no=glacno_list_good)
        
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        main_glac_rgi_land['CenLon_round'] = np.floor(main_glac_rgi_land.CenLon.values/degree_size) * degree_size
        main_glac_rgi_land['CenLat_round'] = np.floor(main_glac_rgi_land.CenLat.values/degree_size) * degree_size
        deg_groups = main_glac_rgi_land.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi_land.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi_land.loc[x,'CenLon_round'], main_glac_rgi_land.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi_land))]
        main_glac_rgi_land['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi_land['deg_id'] = main_glac_rgi_land.CenLon_CenLat.map(deg_dict)
        
        # Degrees
        if main_glac_rgi_land['deg_id'].isnull().all():
            unique_degids = None
        else:
            unique_degids = np.unique(main_glac_rgi_land['deg_id'])
        
        print('# degids:', len(unique_degids))

    
        years = None 
        bad_glacno_list = {}
        for rcp in rcps:
            bad_glacno_list[rcp] = {}
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                bad_glacno_list[rcp][gcm_name] = []
                
                # Tidewater glaciers
                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                netcdf_fp_binned_land = netcdf_fp_cmip5_land + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats_land = netcdf_fp_cmip5_land + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Next set
                pickle_fp_reg_calving =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                if not os.path.exists(pickle_fp_reg_calving + fn_reg_vol_annual) or overwrite_pickle:
                
                    # ----- Load existing LAND-TERMINATING data -----
                    # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                    pickle_fp_reg =  pickle_fp_land + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                    pickle_fp_regO2 =  pickle_fp_land + str(reg).zfill(2) + '/O2Regions/' + gcm_name + '/' + rcp + '/'
                    pickle_fp_degid =  pickle_fp_land + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                    # Region string prefix
                    reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                    regO2_rcp_gcm_str = 'R' + str(reg) + '_O2Regions_' + rcp + '_' + gcm_name
                    degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name
                    
                    # Volume
                    fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                    fn_regO2_vol_annual = regO2_rcp_gcm_str + '_vol_annual.pkl'
                    fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                    # Volume below sea level 
                    fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                    fn_regO2_vol_annual_bwl = regO2_rcp_gcm_str + '_vol_annual_bwl.pkl'
                    fn_degid_vol_annual_bwl = degid_rcp_gcm_str + '_vol_annual_bwl.pkl'
                    # Volume below debris
                    fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                    fn_regO2_vol_annual_bd = regO2_rcp_gcm_str + '_vol_annual_bd.pkl'
                    fn_degid_vol_annual_bd = degid_rcp_gcm_str + '_vol_annual_bd.pkl'
                    # Area 
                    fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                    fn_regO2_area_annual = regO2_rcp_gcm_str + '_area_annual.pkl'
                    fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                    # Area below debris
                    fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                    fn_regO2_area_annual_bd = regO2_rcp_gcm_str + '_area_annual_bd.pkl'
                    fn_degid_area_annual_bd = degid_rcp_gcm_str + '_area_annual_bd.pkl'
                    # Binned Volume
                    fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                    fn_regO2_vol_annual_binned = regO2_rcp_gcm_str + '_vol_annual_binned.pkl'
                    # Binned Volume below debris
                    fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                    fn_regO2_vol_annual_binned_bd = regO2_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                    # Binned Area
                    fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                    fn_regO2_area_annual_binned = regO2_rcp_gcm_str + '_area_annual_binned.pkl'
                    # Binned Area below debris
                    fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                    fn_regO2_area_annual_binned_bd = regO2_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                    # Mass balance: accumulation
                    fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                    fn_regO2_acc_monthly = regO2_rcp_gcm_str + '_acc_monthly.pkl'
                    # Mass balance: refreeze
                    fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                    fn_regO2_refreeze_monthly = regO2_rcp_gcm_str + '_refreeze_monthly.pkl'
                    # Mass balance: melt
                    fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                    fn_regO2_melt_monthly = regO2_rcp_gcm_str + '_melt_monthly.pkl'
                    # Mass balance: frontal ablation
                    fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                    fn_regO2_frontalablation_monthly = regO2_rcp_gcm_str + '_frontalablation_monthly.pkl'
                    # Mass balance: total mass balance
                    fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                    fn_regO2_massbaltotal_monthly = regO2_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                    fn_degid_massbaltotal_monthly = degid_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                    # Binned Climatic Mass Balance
                    fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                    fn_regO2_mbclim_annual_binned = regO2_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                    # Runoff: moving-gauged
                    fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                    fn_regO2_runoff_monthly_moving = regO2_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                    fn_degid_runoff_monthly_moving = degid_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                    # Runoff: fixed-gauged
                    fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                    fn_regO2_runoff_monthly_fixed = regO2_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                    fn_degid_runoff_monthly_fixed = degid_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                    # Runoff: precipitation
                    fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
                    fn_regO2_prec_monthly = regO2_rcp_gcm_str + '_prec_monthly.pkl'
                    # Runoff: off-glacier precipitation
                    fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
                    fn_regO2_offglac_prec_monthly = regO2_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                    # Runoff: off-glacier melt
                    fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                    fn_regO2_offglac_melt_monthly = regO2_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                    # Runoff: off-glacier refreeze
                    fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                    fn_regO2_offglac_refreeze_monthly = regO2_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                    # ELA
                    fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                    fn_regO2_ela_annual = regO2_rcp_gcm_str + '_ela_annual.pkl'
                    # AAR
                    fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                    fn_regO2_aar_annual = regO2_rcp_gcm_str + '_aar_annual.pkl'
    
                    # ----- Load Data -----
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                        reg_vol_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'rb') as f:
                        regO2_vol_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                        degid_vol_annual = pickle.load(f)
                    # Volume below sea level
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'rb') as f:
                        regO2_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'rb') as f:
                        degid_vol_annual_bwl = pickle.load(f)
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                        reg_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'rb') as f:
                        regO2_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'rb') as f:
                        degid_vol_annual_bd = pickle.load(f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                        reg_area_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'rb') as f:
                        regO2_area_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                        degid_area_annual = pickle.load(f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                        reg_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'rb') as f:
                        regO2_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'rb') as f:
                        degid_area_annual_bd = pickle.load(f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                        reg_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'rb') as f:
                        regO2_vol_annual_binned = pickle.load(f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                        reg_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'rb') as f:
                        regO2_vol_annual_binned_bd = pickle.load(f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                        reg_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'rb') as f:
                        regO2_area_annual_binned = pickle.load(f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                        reg_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'rb') as f:
                        regO2_area_annual_binned_bd = pickle.load(f)
                    # Mass balance: accumulation
                    with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                        reg_acc_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_acc_monthly, 'rb') as f:
                        regO2_acc_monthly = pickle.load(f)
                    # Mass balance: refreeze
                    with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                        reg_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_refreeze_monthly, 'rb') as f:
                        regO2_refreeze_monthly = pickle.load(f)
                    # Mass balance: melt
                    with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                        reg_melt_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_melt_monthly, 'rb') as f:
                        regO2_melt_monthly = pickle.load(f)
                    # Mass balance: frontal ablation
                    with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                        reg_frontalablation_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_frontalablation_monthly, 'rb') as f:
                        regO2_frontalablation_monthly = pickle.load(f)
                    # Mass balance: total mass balance
                    with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
                        reg_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_massbaltotal_monthly, 'rb') as f:
                        regO2_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_massbaltotal_monthly, 'rb') as f:
                        degid_massbaltotal_monthly = pickle.load(f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                        reg_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'rb') as f:
                        regO2_mbclim_annual_binned = pickle.load(f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                        reg_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'rb') as f:
                        regO2_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'rb') as f:
                        degid_runoff_monthly_moving = pickle.load(f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                        reg_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'rb') as f:
                        regO2_runoff_monthly_fixed= pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'rb') as f:
                        degid_runoff_monthly_fixed = pickle.load(f)
                    # Runoff: precipitation
                    with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
                        reg_prec_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_prec_monthly, 'rb') as f:
                        regO2_prec_monthly = pickle.load(f)
                    # Runoff: off-glacier precipitation
                    with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
                        reg_offglac_prec_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_prec_monthly, 'rb') as f:
                        regO2_offglac_prec_monthly = pickle.load(f)
                    # Runoff: off-glacier melt
                    with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
                        reg_offglac_melt_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_melt_monthly, 'rb') as f:
                        regO2_offglac_melt_monthly = pickle.load(f)
                    # Runoff: off-glacier refreeze
                    with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
                        reg_offglac_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_refreeze_monthly, 'rb') as f:
                        regO2_offglac_refreeze_monthly = pickle.load(f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                        reg_ela_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'rb') as f:
                        regO2_ela_annual = pickle.load(f)
    #                # AAR
    #                with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
    #                    reg_aar_annual = pickle.load(f)
    #                with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'rb') as f:
    #                    regO2_aar_annual = pickle.load(f)
    
                    #%%
    #                # Load Master RGIId list that is used to correct Region 17
    #                # Load glacier volume data
    #                vol_annual_fn = str(reg).zfill(2) + '_' + rgiids_master_gcm + '_' + rgiids_master_rcp + '_glac_vol_annual.csv'
    #                reg_glac_vol_annual_df = pd.read_csv(csv_fp + vol_annual_fn)
    #                rgiids_gcm_raw = list(reg_glac_vol_annual_df.values[:,0])
    #                rgiids_master = [str(reg) + '.' + str(int(np.round((x-reg)*1e5))).zfill(5) for x in rgiids_gcm_raw]
    #                reg_rgiids[reg] = rgiids_master.copy()
    #                reg_glac_vol_annual_gcm = reg_glac_vol_annual_df.values[:,1:]
    #                reg_glac_vol_annual_master = reg_glac_vol_annual_gcm.copy()
                    
    
                    #%%
                    # ----- COPY ALL DATASETS ----
                    reg_vol_annual_land = np.copy(reg_vol_annual)
                    regO2_vol_annual_land = np.copy(regO2_vol_annual)
                    degid_vol_annual = np.copy(degid_vol_annual)
                    
                    reg_vol_annual_bwl = np.copy(reg_vol_annual_bwl)
                    regO2_vol_annual_bwl = np.copy(regO2_vol_annual_bwl)
                    degid_vol_annual_bwl = np.copy(degid_vol_annual_bwl)
                    
                    reg_vol_annual_bd = np.copy(reg_vol_annual_bd)
                    regO2_vol_annual_bd = np.copy(regO2_vol_annual_bd)
                    degid_vol_annual_bd = np.copy(degid_vol_annual_bd)
                    
                    reg_area_annual = np.copy(reg_area_annual) 
                    regO2_area_annual = np.copy(regO2_area_annual)
                    degid_area_annual = np.copy(degid_area_annual)
                    
                    reg_area_annual_bd = np.copy(reg_area_annual_bd) 
                    regO2_area_annual_bd = np.copy(regO2_area_annual_bd)
                    degid_area_annual_bd = np.copy(degid_area_annual_bd)
                    
                    reg_vol_annual_binned = np.copy(reg_vol_annual_binned)
                    regO2_vol_annual_binned = np.copy(regO2_vol_annual_binned)
                    
                    reg_vol_annual_binned_bd = np.copy(reg_vol_annual_binned_bd)
                    regO2_vol_annual_binned_bd = np.copy(regO2_vol_annual_binned_bd)
                    
                    reg_area_annual_binned = np.copy(reg_area_annual_binned)
                    regO2_area_annual_binned = np.copy(regO2_area_annual_binned)
                    
                    reg_area_annual_binned_bd = np.copy(reg_area_annual_binned_bd)
                    regO2_area_annual_binned_bd = np.copy(regO2_area_annual_binned_bd)
                    
                    reg_acc_monthly = np.copy(reg_acc_monthly)
                    regO2_acc_monthly = np.copy(regO2_acc_monthly)
                    
                    reg_refreeze_monthly = np.copy(reg_refreeze_monthly)
                    regO2_refreeze_monthly = np.copy(regO2_refreeze_monthly)
                    
                    reg_melt_monthly = np.copy(reg_melt_monthly)
                    regO2_melt_monthly = np.copy(regO2_melt_monthly)
                    
                    reg_frontalablation_monthly = np.copy(reg_frontalablation_monthly)
                    regO2_frontalablation_monthly = np.copy(regO2_frontalablation_monthly)
                    
                    reg_massbaltotal_monthly = np.copy(reg_massbaltotal_monthly)
                    regO2_massbaltotal_monthly = np.copy(regO2_massbaltotal_monthly)
                    degid_massbaltotal_monthly = np.copy(degid_massbaltotal_monthly)
                    
                    reg_mbclim_annual_binned = np.copy(reg_mbclim_annual_binned)
                    regO2_mbclim_annual_binned = np.copy(regO2_mbclim_annual_binned)
                    
                    reg_runoff_monthly_moving = np.copy(reg_runoff_monthly_moving)
                    regO2_runoff_monthly_moving = np.copy(regO2_runoff_monthly_moving)
                    degid_runoff_monthly_moving = np.copy(degid_runoff_monthly_moving)
                    
                    reg_runoff_monthly_fixed = np.copy(reg_runoff_monthly_fixed)
                    regO2_runoff_monthly_fixed = np.copy(regO2_runoff_monthly_fixed)
                    degid_runoff_monthly_fixed = np.copy(degid_runoff_monthly_fixed)
                    
                    reg_prec_monthly = np.copy(reg_prec_monthly)
                    regO2_prec_monthly = np.copy(regO2_prec_monthly) 
                    
                    reg_offglac_prec_monthly = np.copy(reg_offglac_prec_monthly)
                    regO2_offglac_prec_monthly = np.copy(regO2_offglac_prec_monthly)
                    
                    reg_offglac_melt_monthly = np.copy(reg_offglac_melt_monthly)
                    regO2_offglac_melt_monthly = np.copy(regO2_offglac_melt_monthly)
                    
                    reg_offglac_refreeze_monthly = np.copy(reg_offglac_refreeze_monthly)
                    regO2_offglac_refreeze_monthly = np.copy(regO2_offglac_refreeze_monthly)
                    
                    reg_ela_annual = np.copy(reg_ela_annual)
                    regO2_ela_annual = np.copy(regO2_ela_annual)
                    reg_ela_annual_area = np.copy(reg_area_annual)
                    regO2_ela_annual_area = np.copy(regO2_area_annual)
                    
    #                reg_aar_annual = np.copy(reg_aar_annual)
    #                regO2_aar_annual = np.copy(regO2_aar_annual)
    
                    
                    for nglac, glacno in enumerate(glacno_list):
                        if nglac%10 == 0:
                            print(gcm_name, rcp, glacno)
                            
                        # Only run for good glaciers
                        if glacno in glacno_list_good:
                            
                            # Group indices
                            try:
                                glac_idx = np.where(main_glac_rgi_land['glacno'] == glacno)[0][0]
                            except:
                                glac_idx = None
                            
                            if glac_idx is not None:
                                regO2 = main_glac_rgi_land.loc[glac_idx, 'O2Region']
                                regO2_idx = np.where(regO2 == unique_regO2s)[0][0]
                                degid = main_glac_rgi_land.loc[glac_idx, 'deg_id']
                                degid_idx = np.where(degid == unique_degids)[0][0]
                                
                                # Load tidewater glacier ds
                                nsim_strs = ['50', '1', '100', '150', '200', '250']
                                ds_binned = None
                                ds_binned_land = None
                                nset = -1
                                while ds_binned is None and nset < len(nsim_strs)-1:
                                    nset += 1
                                    nsim_str = nsim_strs[nset]
                                    
                                    try:
                                        netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                        netcdf_fn_binned = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
                
                                        netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                        netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                        
                                        # Open files
                                        ds_binned = xr.open_dataset(netcdf_fp_binned + '/' + netcdf_fn_binned)
                                        ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                                    except:
                                        ds_binned = None
                                
                                # Years
                                if years is None:
                                    years = ds_stats.year.values
                                # Load clean ice glacier ds
                                ds_binned_land = None
                                nset = -1
                                while ds_binned_land is None and nset < len(nsim_strs)-1:
                                    nset += 1
                                    nsim_str = nsim_strs[nset]
                                    
                                    try:
                                        netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                        netcdf_fn_binned_land = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
                
                                        netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                        netcdf_fn_stats_land = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                        
                                        # Open files
                                        ds_binned_land = xr.open_dataset(netcdf_fp_binned_land + '/' + netcdf_fn_binned_land)
                                        ds_stats_land = xr.open_dataset(netcdf_fp_stats_land + '/' + netcdf_fn_stats_land)
                                    except:
                                        ds_binned_land = None
            
                                if (ds_binned is not None) and (ds_binned_land is not None):
                                    # ----- 1. Volume (m3) vs. Year ----- 
                                    glac_vol_annual = ds_stats.glac_volume_annual.values[0,:]
                                    glac_vol_annual_land = ds_stats_land.glac_volume_annual.values[0,:]
        
                                    if np.isnan(glac_vol_annual).any() or np.isnan(glac_vol_annual_land).any():
                                        assert True==False, 'Fix this glacier:' + glacno
                                    
                                    # All, O2Region, Watershed, DegId
                                    reg_vol_annual = reg_vol_annual - glac_vol_annual_land + glac_vol_annual
                                    regO2_vol_annual[regO2_idx,:] = regO2_vol_annual[regO2_idx,:] - glac_vol_annual_land + glac_vol_annual
                                    degid_vol_annual[degid_idx,:] = degid_vol_annual[degid_idx,:] - glac_vol_annual_land + glac_vol_annual
                                    
                                    # ----- 2. Volume below-sea-level (m3) vs. Year ----- 
                                    #  - initial elevation is stored
                                    #  - bed elevation is constant in time
                                    #  - assume sea level is at 0 m a.s.l.
                                    #  --> recomputing this way ensures consistency with profiles 
                                    #      instead of using ds_stats.glac_volume_bsl_annual
                                    z_sealevel = 0
                                    bin_z_init = ds_binned.bin_surface_h_initial.values[0,:]
                                    bin_thick_annual = ds_binned.bin_thick_annual.values[0,:,:]
                                    bin_z_bed = bin_z_init - bin_thick_annual[:,0]
                                    # Annual surface height
                                    bin_z_surf_annual = bin_z_bed[:,np.newaxis] + bin_thick_annual
                                    
                                    # Annual volume (m3)
                                    bin_vol_annual = ds_binned.bin_volume_annual.values[0,:,:]
                                    # Annual area (m2)
                                    bin_area_annual = np.zeros(bin_vol_annual.shape)
                                    bin_area_annual[bin_vol_annual > 0] = (
                                            bin_vol_annual[bin_vol_annual > 0] / bin_thick_annual[bin_vol_annual > 0])
                                    
                                    # Processed based on OGGM's _vol_below_level function
                                    bwl = (bin_z_bed[:,np.newaxis] < 0) & (bin_thick_annual > 0)
                                    if bwl.any():
                                        # Annual surface height (max of sea level for calcs)
                                        bin_z_surf_annual_bwl = bin_z_surf_annual.copy()
                                        bin_z_surf_annual_bwl[bin_z_surf_annual_bwl > z_sealevel] = z_sealevel
                                        # Annual thickness below sea level (m)
                                        bin_thick_annual_bwl = bin_thick_annual.copy()
                                        bin_thick_annual_bwl = bin_z_surf_annual_bwl - bin_z_bed[:,np.newaxis]
                                        bin_thick_annual_bwl[~bwl] = 0
                                        # Annual volume below sea level (m3)
                                        bin_vol_annual_bwl = np.zeros(bin_vol_annual.shape)
                                        bin_vol_annual_bwl[bwl] = bin_thick_annual_bwl[bwl] * bin_area_annual[bwl]
                                        glac_vol_annual_bwl = bin_vol_annual_bwl.sum(0)
                                    else:
                                        glac_vol_annual_bwl = np.zeros(glac_vol_annual.shape)
                                        
                                    if glac_vol_annual_bwl[-1] > 0 and glac_vol_annual_bwl[-2] == 0:
                                        bad_glacno_list[rcp][gcm_name].append(glacno)
                                        
                                    # PROCESS LAND-TERMINATING
                                    bin_z_init_land = ds_binned_land.bin_surface_h_initial.values[0,:]
                                    bin_thick_annual_land = ds_binned_land.bin_thick_annual.values[0,:,:]
                                    bin_z_bed_land = bin_z_init_land - bin_thick_annual_land[:,0]
                                    # Annual surface height
                                    bin_z_surf_annual_land = bin_z_bed_land[:,np.newaxis] + bin_thick_annual_land
                                    
                                    # Annual volume (m3)
                                    bin_vol_annual_land = ds_binned_land.bin_volume_annual.values[0,:,:]
                                    # Annual area (m2)
                                    bin_area_annual_land = np.zeros(bin_vol_annual_land.shape)
                                    bin_area_annual_land[bin_vol_annual_land > 0] = (
                                            bin_vol_annual_land[bin_vol_annual_land > 0] / bin_thick_annual_land[bin_vol_annual_land > 0])
                                    
                                    # Processed based on OGGM's _vol_below_level function
                                    bwl_land = (bin_z_bed_land[:,np.newaxis] < 0) & (bin_thick_annual_land > 0)
                                    if bwl_land.any():
                                        # Annual surface height (max of sea level for calcs)
                                        bin_z_surf_annual_bwl_land = bin_z_surf_annual_land.copy()
                                        bin_z_surf_annual_bwl_land[bin_z_surf_annual_bwl_land > z_sealevel] = z_sealevel
                                        # Annual thickness below sea level (m)
                                        bin_thick_annual_bwl_land = bin_thick_annual_land.copy()
                                        bin_thick_annual_bwl_land = bin_z_surf_annual_bwl_land - bin_z_bed_land[:,np.newaxis]
                                        bin_thick_annual_bwl_land[~bwl_land] = 0
                                        # Annual volume below sea level (m3)
                                        bin_vol_annual_bwl_land = np.zeros(bin_vol_annual_land.shape)
                                        bin_vol_annual_bwl_land[bwl_land] = bin_thick_annual_bwl_land[bwl_land] * bin_area_annual_land[bwl_land]
                                        glac_vol_annual_bwl_land = bin_vol_annual_bwl_land.sum(0)
                                    else:
                                        glac_vol_annual_bwl_land = np.zeros(glac_vol_annual.shape)
                                    
                                        
                                    # All, O2Region, Watershed, DegId
                                    reg_vol_annual_bwl = reg_vol_annual_bwl - glac_vol_annual_bwl_land + glac_vol_annual_bwl
                                    regO2_vol_annual_bwl[regO2_idx,:] = regO2_vol_annual_bwl[regO2_idx,:] - glac_vol_annual_bwl_land + glac_vol_annual_bwl
                                    degid_vol_annual_bwl[degid_idx,:] = degid_vol_annual_bwl[degid_idx,:] - glac_vol_annual_bwl_land + glac_vol_annual_bwl
                                    
                                    
                                    # ----- 3. Volume below-debris vs. Time ----- 
                                    gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
                                    fls = gdir.read_pickle('inversion_flowlines')
                #                    fls = gdir.read_pickle('model_flowlines')
                                    bin_debris_hd = np.zeros(bin_z_init.shape)
                                    bin_debris_ed = np.zeros(bin_z_init.shape) + 1
                                    if 'debris_hd' in dir(fls[0]):
                                        bin_debris_hd[0:fls[0].debris_hd.shape[0]] = fls[0].debris_hd
                                        bin_debris_ed[0:fls[0].debris_hd.shape[0]] = fls[0].debris_ed
                                    if bin_debris_hd.sum() > 0:
                                        
                                        try: 
                                            bin_vol_annual_bd = np.zeros(bin_vol_annual.shape)
                                            bin_vol_annual_bd[bin_debris_hd > 0, :] = bin_vol_annual[bin_debris_hd > 0, :]
                                            glac_vol_annual_bd = bin_vol_annual_bd.sum(0)
                                            
                                            bin_vol_annual_bd_land = np.zeros(bin_vol_annual.shape)
                                            bin_vol_annual_bd_land[bin_debris_hd > 0, :] = bin_vol_annual_land[bin_debris_hd > 0, :]
                                            glac_vol_annual_bd_land = bin_vol_annual_bd_land.sum(0)
                                            
                                            # All, O2Region, Watershed, DegId
                                            reg_vol_annual_bd = reg_vol_annual_bd - glac_vol_annual_bd_land + glac_vol_annual_bd
                                            regO2_vol_annual_bd[regO2_idx,:] = regO2_vol_annual_bd[regO2_idx,:] - glac_vol_annual_bd_land + glac_vol_annual_bd
                                            degid_vol_annual_bd[degid_idx,:] = degid_vol_annual_bd[degid_idx,:] - glac_vol_annual_bd_land + glac_vol_annual_bd
                                        except:
                                            pass
            
                                    # ----- 4. Area vs. Time ----- 
                                    glac_area_annual = ds_stats.glac_area_annual.values[0,:]
                                    glac_area_annual_land = ds_stats_land.glac_area_annual.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_area_annual = reg_area_annual - glac_area_annual_land + glac_area_annual
                                    regO2_area_annual[regO2_idx,:] = regO2_area_annual[regO2_idx,:] - glac_area_annual_land + glac_area_annual
                                    degid_area_annual[degid_idx,:] = degid_area_annual[degid_idx,:] - glac_area_annual_land + glac_area_annual
                                        
                                    # ----- 5. Area below-debris vs. Time ----- 
                                    if bin_debris_hd.sum() > 0:
                                        try:
                                            bin_area_annual_bd = np.zeros(bin_area_annual.shape)
                                            bin_area_annual_bd[bin_debris_hd > 0, :] = bin_area_annual[bin_debris_hd > 0, :]
                                            glac_area_annual_bd = bin_area_annual_bd.sum(0)
                                            
                                            bin_area_annual_bd_land = np.zeros(bin_area_annual.shape)
                                            bin_area_annual_bd_land[bin_debris_hd > 0, :] = bin_area_annual_land[bin_debris_hd > 0, :]
                                            glac_area_annual_bd_land = bin_area_annual_bd_land.sum(0)
                                            
                                            # All, O2Region, Watershed, DegId
                                            reg_area_annual_bd = reg_area_annual_bd - glac_area_annual_bd_land + glac_area_annual_bd
                                            regO2_area_annual_bd[regO2_idx,:] = regO2_area_annual_bd[regO2_idx,:] - glac_area_annual_bd_land + glac_area_annual_bd
                                            degid_area_annual_bd[degid_idx,:] = degid_area_annual_bd[degid_idx,:] - glac_area_annual_bd_land + glac_area_annual_bd
                                        except:
                                            pass
                                            
                                    # ----- 6. Binned glacier volume vs. Time ----- 
                                    bin_vol_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                                    bin_vol_annual_10m_land = np.zeros((len(elev_bins)-1, len(years)))
                                    for ncol, year in enumerate(years):
                                        bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                             weights=bin_vol_annual[:,ncol])
                                        bin_vol_annual_10m[:,ncol] = bin_counts
                                        
                                        bin_counts_land, bin_edges = np.histogram(bin_z_surf_annual_land[:,ncol], bins=elev_bins, 
                                                                                  weights=bin_vol_annual_land[:,ncol])
                                        bin_vol_annual_10m_land[:,ncol] = bin_counts_land
                                    
                                    # All, O2Region, Watershed, DegId
                                    reg_vol_annual_binned = reg_vol_annual_binned - bin_vol_annual_10m_land + bin_vol_annual_10m
                                    regO2_vol_annual_binned[regO2_idx,:,:] = regO2_vol_annual_binned[regO2_idx,:,:] - bin_vol_annual_10m_land + bin_vol_annual_10m
                                        
                                
                                    # ----- 7. Binned glacier volume below debris vs. Time ----- 
                                    if bin_debris_hd.sum() > 0:
                                        # Bin debris mask for the given elevation bins
                                        bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                                        bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                                        bin_debris_mask_10m[bin_counts > 0] = 1
                                        bin_vol_annual_10m_bd = bin_vol_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                                        
                                        bin_vol_annual_10m_bd_land = bin_vol_annual_10m_land * bin_debris_mask_10m[:,np.newaxis]
                                        
                                        # All, O2Region, Watershed, DegId
                                        reg_vol_annual_binned_bd = reg_vol_annual_binned_bd - bin_vol_annual_10m_bd_land + bin_vol_annual_10m_bd
                                        regO2_vol_annual_binned_bd[regO2_idx,:,:] = regO2_vol_annual_binned_bd[regO2_idx,:,:] - bin_vol_annual_10m_bd_land + bin_vol_annual_10m_bd
                
                                    # ----- 8. Binned glacier area vs. Time ----- 
                                    bin_area_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                                    bin_area_annual_10m_land = np.zeros((len(elev_bins)-1, len(years)))
                                    for ncol, year in enumerate(years):
                                        bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                             weights=bin_area_annual[:,ncol])
                                        bin_area_annual_10m[:,ncol] = bin_counts
                                        
                                        bin_counts_land, bin_edges = np.histogram(bin_z_surf_annual_land[:,ncol], bins=elev_bins, 
                                                                                  weights=bin_area_annual_land[:,ncol])
                                        bin_area_annual_10m_land[:,ncol] = bin_counts_land
                                    
                                    # All, O2Region, Watershed, DegId
                                    reg_area_annual_binned = reg_area_annual_binned - bin_area_annual_10m_land + bin_area_annual_10m
                                    regO2_area_annual_binned[regO2_idx,:,:] = regO2_area_annual_binned[regO2_idx,:,:] - bin_area_annual_10m_land + bin_area_annual_10m
                                
                                
                                    # ----- 9. Binned glacier area below debris vs. Time ----- 
                                    if bin_debris_hd.sum() > 0:
                                        # Bin debris mask for the given elevation bins
                                        bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                                        bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                                        bin_debris_mask_10m[bin_counts > 0] = 1
                                        bin_area_annual_10m_bd = bin_area_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                                        
                                        bin_area_annual_10m_bd_land = bin_area_annual_10m_land * bin_debris_mask_10m[:,np.newaxis]
                                        
                                        # All, O2Region, Watershed, DegId
                                        reg_area_annual_binned_bd = reg_area_annual_binned_bd - bin_area_annual_10m_bd_land + bin_area_annual_10m_bd
                                        regO2_area_annual_binned_bd[regO2_idx,:,:] = regO2_area_annual_binned_bd[regO2_idx,:,:] - bin_area_annual_10m_bd_land + bin_area_annual_10m_bd
                                    
                
                                    # ----- 10. Mass Balance Components vs. Time -----
                                    # - these are only meant for monthly and/or relative purposes 
                                    #   mass balance from volume change should be used for annual changes
                                    # Accumulation
                                    glac_acc_monthly = ds_stats.glac_acc_monthly.values[0,:]
                                    glac_acc_monthly_land = ds_stats_land.glac_acc_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_acc_monthly = reg_acc_monthly - glac_acc_monthly_land + glac_acc_monthly
                                    regO2_acc_monthly[regO2_idx,:] = regO2_acc_monthly[regO2_idx,:] - glac_acc_monthly_land + glac_acc_monthly
                                    
                                    
                                    # Refreeze
                                    glac_refreeze_monthly = ds_stats.glac_refreeze_monthly.values[0,:]
                                    glac_refreeze_monthly_land = ds_stats_land.glac_refreeze_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_refreeze_monthly = reg_refreeze_monthly - glac_refreeze_monthly_land + glac_refreeze_monthly
                                    regO2_refreeze_monthly[regO2_idx,:] = regO2_refreeze_monthly[regO2_idx,:] - glac_refreeze_monthly_land + glac_refreeze_monthly
                                        
                                    # Melt
                                    glac_melt_monthly = ds_stats.glac_melt_monthly.values[0,:]
                                    glac_melt_monthly_land = ds_stats_land.glac_melt_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_melt_monthly = reg_melt_monthly - glac_melt_monthly_land + glac_melt_monthly
                                    regO2_melt_monthly[regO2_idx,:] = regO2_melt_monthly[regO2_idx,:] - glac_melt_monthly_land + glac_melt_monthly
                                        
                                    # Frontal Ablation
                                    glac_frontalablation_monthly = ds_stats.glac_frontalablation_monthly.values[0,:]
                                    glac_frontalablation_monthly_land = ds_stats_land.glac_frontalablation_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_frontalablation_monthly = reg_frontalablation_monthly - glac_frontalablation_monthly_land + glac_frontalablation_monthly
                                    regO2_frontalablation_monthly[regO2_idx,:] = regO2_frontalablation_monthly[regO2_idx,:] - glac_frontalablation_monthly_land + glac_frontalablation_monthly
                                        
                                    # Total Mass Balance
                                    glac_massbaltotal_monthly = ds_stats.glac_massbaltotal_monthly.values[0,:]
                                    glac_massbaltotal_monthly_land = ds_stats_land.glac_massbaltotal_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_massbaltotal_monthly = reg_massbaltotal_monthly - glac_massbaltotal_monthly_land + glac_massbaltotal_monthly
                                    regO2_massbaltotal_monthly[regO2_idx,:] = regO2_massbaltotal_monthly[regO2_idx,:] - glac_massbaltotal_monthly_land + glac_massbaltotal_monthly
                                    degid_massbaltotal_monthly[degid_idx,:] = degid_massbaltotal_monthly[degid_idx,:] - glac_massbaltotal_monthly_land + glac_massbaltotal_monthly
                                    
                                    
                                    
                                    # ----- 11. Binned Climatic Mass Balance vs. Time -----
                                    # - Various mass balance datasets may have slight mismatch due to averaging
                                    #   ex. mbclim_annual was reported in mwe, so the area average will cause difference
                                    #   ex. mbtotal_monthly was averaged on a monthly basis, so the temporal average will cause difference
                                    bin_mbclim_annual = ds_binned.bin_massbalclim_annual.values[0,:,:]
                                    bin_mbclim_annual_m3we = bin_mbclim_annual * bin_area_annual
                                    bin_mbclim_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                                    for ncol, year in enumerate(years):
                                        bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                             weights=bin_mbclim_annual_m3we[:,ncol])
                                        bin_mbclim_annual_10m[:,ncol] = bin_counts
                                        
                                    bin_mbclim_annual_land = ds_binned_land.bin_massbalclim_annual.values[0,:,:]
                                    bin_mbclim_annual_m3we_land = bin_mbclim_annual_land * bin_area_annual_land
                                    bin_mbclim_annual_10m_land = np.zeros((len(elev_bins)-1, len(years)))
                                    for ncol, year in enumerate(years):
                                        bin_counts_land, bin_edges = np.histogram(bin_z_surf_annual_land[:,ncol], bins=elev_bins, 
                                                                                  weights=bin_mbclim_annual_m3we_land[:,ncol])
                                        bin_mbclim_annual_10m_land[:,ncol] = bin_counts_land
                                        
                                    # All, O2Region, Watershed, DegId
                                    reg_mbclim_annual_binned = reg_mbclim_annual_binned - bin_mbclim_annual_10m_land + bin_mbclim_annual_10m
                                    regO2_mbclim_annual_binned[regO2_idx,:,:] = regO2_mbclim_annual_binned[regO2_idx,:,:] - bin_mbclim_annual_10m_land + bin_mbclim_annual_10m
                
                                    
                                    # ----- 12. Runoff vs. Time -----
                                    glac_runoff_monthly = ds_stats.glac_runoff_monthly.values[0,:]
                                    glac_runoff_monthly_land = ds_stats_land.glac_runoff_monthly.values[0,:]
                                    # Moving-gauge Runoff vs. Time
                                    # All, O2Region, Watershed, DegId
                                    reg_runoff_monthly_moving = reg_runoff_monthly_moving- glac_runoff_monthly_land + glac_runoff_monthly
                                    regO2_runoff_monthly_moving[regO2_idx,:] = regO2_runoff_monthly_moving[regO2_idx,:] - glac_runoff_monthly_land + glac_runoff_monthly
                                    degid_runoff_monthly_moving[degid_idx,:] = degid_runoff_monthly_moving[degid_idx,:] - glac_runoff_monthly_land + glac_runoff_monthly
                                        
                                    # Fixed-gauge Runoff vs. Time
                                    offglac_runoff_monthly = ds_stats.offglac_runoff_monthly.values[0,:]
                                    glac_runoff_monthly_fixed = glac_runoff_monthly + offglac_runoff_monthly
                                    
                                    offglac_runoff_monthly_land = ds_stats_land.offglac_runoff_monthly.values[0,:]
                                    glac_runoff_monthly_fixed_land = glac_runoff_monthly_land + offglac_runoff_monthly_land
                                    
                                    # All, O2Region, Watershed, DegId
                                    reg_runoff_monthly_fixed = reg_runoff_monthly_fixed - glac_runoff_monthly_fixed_land + glac_runoff_monthly_fixed
                                    regO2_runoff_monthly_fixed[regO2_idx,:] = regO2_runoff_monthly_fixed[regO2_idx,:] - glac_runoff_monthly_fixed_land + glac_runoff_monthly_fixed
                                    degid_runoff_monthly_fixed[degid_idx,:] = degid_runoff_monthly_fixed[degid_idx,:] - glac_runoff_monthly_fixed_land + glac_runoff_monthly_fixed
                                    
                                    
                                    # Runoff Components
                                    # Precipitation
                                    glac_prec_monthly = ds_stats.glac_prec_monthly.values[0,:]
                                    glac_prec_monthly_land = ds_stats_land.glac_prec_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_prec_monthly = reg_prec_monthly - glac_prec_monthly_land + glac_prec_monthly
                                    regO2_prec_monthly[regO2_idx,:] = regO2_prec_monthly[regO2_idx,:] - glac_prec_monthly_land + glac_prec_monthly
                                        
                                    # Off-glacier Precipitation
                                    offglac_prec_monthly = ds_stats.offglac_prec_monthly.values[0,:]
                                    offglac_prec_monthly_land = ds_stats_land.offglac_prec_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_offglac_prec_monthly = reg_offglac_prec_monthly - offglac_prec_monthly_land + offglac_prec_monthly
                                    regO2_offglac_prec_monthly[regO2_idx,:] = regO2_offglac_prec_monthly[regO2_idx,:] - offglac_prec_monthly_land + offglac_prec_monthly
                                        
                                    # Off-glacier Melt
                                    offglac_melt_monthly = ds_stats.offglac_melt_monthly.values[0,:]
                                    offglac_melt_monthly_land = ds_stats_land.offglac_melt_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_offglac_melt_monthly = reg_offglac_melt_monthly - offglac_melt_monthly_land + offglac_melt_monthly
                                    regO2_offglac_melt_monthly[regO2_idx,:] = regO2_offglac_melt_monthly[regO2_idx,:]  - offglac_melt_monthly_land + offglac_melt_monthly
          
                                    # Off-glacier Refreeze
                                    offglac_refreeze_monthly = ds_stats.offglac_refreeze_monthly.values[0,:]
                                    offglac_refreeze_monthly_land = ds_stats_land.offglac_refreeze_monthly.values[0,:]
                                    # All, O2Region, Watershed, DegId
                                    reg_offglac_refreeze_monthly = reg_offglac_refreeze_monthly - offglac_refreeze_monthly_land + offglac_refreeze_monthly
                                    regO2_offglac_refreeze_monthly[regO2_idx,:] = regO2_offglac_refreeze_monthly[regO2_idx,:] - offglac_refreeze_monthly_land + offglac_refreeze_monthly
                
                                    # ----- 13. ELA vs. Time -----
                                    glac_ela_annual = ds_stats.glac_ELA_annual.values[0,:]
                                    glac_ela_annual_land = ds_stats_land.glac_ELA_annual.values[0,:]
                                    if np.isnan(glac_ela_annual).any():
                                        # Quality control nan values 
                                        #  - replace with max elev because occur when entire glacier has neg mb
                                        bin_z_surf_annual_glaconly = bin_z_surf_annual.copy()
                                        bin_z_surf_annual_glaconly[bin_thick_annual == 0] = np.nan
                                        zmax_annual = np.nanmax(bin_z_surf_annual_glaconly, axis=0)
                                        glac_ela_annual[np.isnan(glac_ela_annual)] = zmax_annual[np.isnan(glac_ela_annual)]
                                    
                                    if np.isnan(glac_ela_annual_land).any():
                                        # Quality control nan values 
                                        #  - replace with max elev because occur when entire glacier has neg mb
                                        bin_z_surf_annual_glaconly_land = bin_z_surf_annual_land.copy()
                                        bin_z_surf_annual_glaconly_land[bin_thick_annual_land == 0] = np.nan
                                        zmax_annual_land = np.nanmax(bin_z_surf_annual_glaconly_land, axis=0)
                                        glac_ela_annual_land[np.isnan(glac_ela_annual_land)] = zmax_annual_land[np.isnan(glac_ela_annual_land)]
                
                                    # Area-weighted ELA
                                    # All, O2Region, Watershed, DegId
                                    # Use index to avoid dividing by 0 when glacier completely melts       
                                    ela_idx = np.where(reg_ela_annual_area + glac_area_annual > 0)[0]
                                    reg_ela_annual[ela_idx] = (
                                            (reg_ela_annual[ela_idx] * reg_ela_annual_area[ela_idx] -
                                             glac_ela_annual_land[ela_idx] * glac_area_annual_land[ela_idx] + 
                                             glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                            (reg_ela_annual_area[ela_idx] - glac_area_annual_land[ela_idx] + glac_area_annual[ela_idx]))
                                    reg_ela_annual_area = reg_ela_annual_area - glac_area_annual_land + glac_area_annual
                                    
                                    # O2Region
                                    ela_idx = np.where(regO2_ela_annual_area[regO2_idx,:] + glac_area_annual > 0)[0]
                                    regO2_ela_annual[regO2_idx,ela_idx] = (
                                            (regO2_ela_annual[regO2_idx,ela_idx] * regO2_ela_annual_area[regO2_idx,ela_idx] -
                                             glac_ela_annual_land[ela_idx] * glac_area_annual_land[ela_idx] + 
                                             glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                             (regO2_ela_annual_area[regO2_idx,ela_idx] - glac_area_annual_land[ela_idx] + glac_area_annual[ela_idx]))
                                    regO2_ela_annual_area[regO2_idx,:] = regO2_ela_annual_area[regO2_idx,:] - glac_area_annual_land + glac_area_annual
        
            
                                    # ----- 14. AAR vs. Time -----
                                    #  - averaging issue with bin_area_annual.sum(0) != glac_area_annual
                                    #  - hence only use these 
                                    # NOT UPDATED BECAUSE DON'T HAVE THE BINNED AREAS OF THE OTHERS; NEED TO REPROCESS IF GOING TO USE
                    
                    print(rcp, gcm_name, 'w calving:', np.round(reg_vol_annual[-1]), 'vs', np.round(reg_vol_annual_land[-1]))
                    
                    # Volume
                    pickle_fp_reg_calving =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                    if not os.path.exists(pickle_fp_reg_calving):
                        os.makedirs(pickle_fp_reg_calving)
                    pickle_fp_regO2_calving =  pickle_fp + str(reg).zfill(2) + '/O2Regions/' + gcm_name + '/' + rcp + '/'
                    if not os.path.exists(pickle_fp_regO2_calving):
                        os.makedirs(pickle_fp_regO2_calving)
                    pickle_fp_degid_calving =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                    if not os.path.exists(pickle_fp_degid_calving):
                        os.makedirs(pickle_fp_degid_calving)
                    
                    # Volume
                    with open(pickle_fp_reg_calving + fn_reg_vol_annual, 'wb') as f:
                        pickle.dump(reg_vol_annual, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_vol_annual, 'wb') as f:
                        pickle.dump(regO2_vol_annual, f)
                    with open(pickle_fp_degid_calving + fn_degid_vol_annual, 'wb') as f:
                        pickle.dump(degid_vol_annual, f)
                    # Volume below sea level 
                    with open(pickle_fp_reg_calving + fn_reg_vol_annual_bwl, 'wb') as f:
                        pickle.dump(reg_vol_annual_bwl, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_vol_annual_bwl, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bwl, f)
                    with open(pickle_fp_degid_calving + fn_degid_vol_annual_bwl, 'wb') as f:
                        pickle.dump(degid_vol_annual_bwl, f) 
                    # Volume below debris
                    with open(pickle_fp_reg_calving + fn_reg_vol_annual_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_bd, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_vol_annual_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bd, f)
                    with open(pickle_fp_degid_calving + fn_degid_vol_annual_bd, 'wb') as f:
                        pickle.dump(degid_vol_annual_bd, f)
                    # Area 
                    with open(pickle_fp_reg_calving + fn_reg_area_annual, 'wb') as f:
                        pickle.dump(reg_area_annual, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_area_annual, 'wb') as f:
                        pickle.dump(regO2_area_annual, f)
                    with open(pickle_fp_degid_calving + fn_degid_area_annual, 'wb') as f:
                        pickle.dump(degid_area_annual, f)
                    # Area below debris
                    with open(pickle_fp_reg_calving + fn_reg_area_annual_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_bd, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_area_annual_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_bd, f)
                    with open(pickle_fp_degid_calving + fn_degid_area_annual_bd, 'wb') as f:
                        pickle.dump(degid_area_annual_bd, f)
                    # Binned Volume
                    with open(pickle_fp_reg_calving + fn_reg_vol_annual_binned, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_vol_annual_binned, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned, f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg_calving + fn_reg_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned_bd, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned_bd, f)
                    # Binned Area
                    with open(pickle_fp_reg_calving + fn_reg_area_annual_binned, 'wb') as f:
                        pickle.dump(reg_area_annual_binned, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_area_annual_binned, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned, f)
                    # Binned Area below debris
                    with open(pickle_fp_reg_calving + fn_reg_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_binned_bd, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned_bd, f)
                    # Mass balance: accumulation
                    with open(pickle_fp_reg_calving + fn_reg_acc_monthly, 'wb') as f:
                        pickle.dump(reg_acc_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_acc_monthly, 'wb') as f:
                        pickle.dump(regO2_acc_monthly, f)
                    # Mass balance: refreeze
                    with open(pickle_fp_reg_calving + fn_reg_refreeze_monthly, 'wb') as f:
                        pickle.dump(reg_refreeze_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_refreeze_monthly, 'wb') as f:
                        pickle.dump(regO2_refreeze_monthly, f)
                    # Mass balance: melt
                    with open(pickle_fp_reg_calving + fn_reg_melt_monthly, 'wb') as f:
                        pickle.dump(reg_melt_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_melt_monthly, 'wb') as f:
                        pickle.dump(regO2_melt_monthly, f)
                    # Mass balance: frontal ablation
                    with open(pickle_fp_reg_calving + fn_reg_frontalablation_monthly, 'wb') as f:
                        pickle.dump(reg_frontalablation_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_frontalablation_monthly, 'wb') as f:
                        pickle.dump(regO2_frontalablation_monthly, f)
                    # Mass balance: total mass balance
                    with open(pickle_fp_reg_calving + fn_reg_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(reg_massbaltotal_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(regO2_massbaltotal_monthly, f)
                    with open(pickle_fp_degid_calving + fn_degid_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(degid_massbaltotal_monthly, f)  
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg_calving + fn_reg_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(reg_mbclim_annual_binned, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(regO2_mbclim_annual_binned, f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg_calving + fn_reg_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_moving, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_moving, f)
                    with open(pickle_fp_degid_calving + fn_degid_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_moving, f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg_calving + fn_reg_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_fixed, f)     
                    with open(pickle_fp_regO2_calving + fn_regO2_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_fixed, f)
                    with open(pickle_fp_degid_calving + fn_degid_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_fixed, f)  
                    # Runoff: precipitation
                    with open(pickle_fp_reg_calving + fn_reg_prec_monthly, 'wb') as f:
                        pickle.dump(reg_prec_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_prec_monthly, 'wb') as f:
                        pickle.dump(regO2_prec_monthly, f)
                    # Runoff: off-glacier precipitation
                    with open(pickle_fp_reg_calving + fn_reg_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_prec_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_prec_monthly, f)
                    # Runoff: off-glacier melt
                    with open(pickle_fp_reg_calving + fn_reg_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_melt_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_melt_monthly, f)
                    # Runoff: off-glacier refreeze
                    with open(pickle_fp_reg_calving + fn_reg_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_refreeze_monthly, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_refreeze_monthly, f)
                    # ELA
                    with open(pickle_fp_reg_calving + fn_reg_ela_annual, 'wb') as f:
                        pickle.dump(reg_ela_annual, f)
                    with open(pickle_fp_regO2_calving + fn_regO2_ela_annual, 'wb') as f:
                        pickle.dump(regO2_ela_annual, f)
                    
                        
                    #%%
                    if args.option_plot:
                        # ===== REGIONAL PLOTS =====
                        fig_fp_reg = fig_fp + str(reg).zfill(2) + '/'
                        if not os.path.exists(fig_fp_reg):
                            os.makedirs(fig_fp_reg)
                            
                        # ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
                        fig, ax = plt.subplots(3, 4, squeeze=False, sharex=False, sharey=False, 
                                               gridspec_kw = {'wspace':0.7, 'hspace':0.5})
                        label= gcm_name + ' ' + rcp
                        
                        # VOLUME CHANGE
                        ax[0,0].plot(years, reg_vol_annual/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                        if not reg_vol_annual_bwl is None:
                            ax[0,0].plot(years, reg_vol_annual_bwl/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle='--', zorder=4, label='bwl')
                        if not reg_vol_annual_bd is None and not (reg_vol_annual_bd == np.array(None)).all():
                            ax[0,0].plot(years, reg_vol_annual_bd/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                            nodebris = False
                        else:
                            nodebris = True
                        ax[0,0].set_ylabel('Volume (km$^{3}$)')
                        ax[0,0].set_xlim(years.min(), years.max())
                        ax[0,0].xaxis.set_major_locator(MultipleLocator(50))
                        ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
                        ax[0,0].set_ylim(0,reg_vol_annual.max()*1.05/1e9)
                        ax[0,0].tick_params(direction='inout', right=True)
                        ax[0,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        #                               loc=(0.05,0.05),
                                       )        
                        
        
                        # AREA CHANGE
                        ax[0,1].plot(years, reg_area_annual/1e6, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                        if not reg_area_annual_bd is None and not nodebris:
                            ax[0,1].plot(years, reg_area_annual_bd/1e6, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                        ax[0,1].set_ylabel('Area (km$^{2}$)')
                        ax[0,1].set_xlim(years.min(), years.max())
                        ax[0,1].xaxis.set_major_locator(MultipleLocator(50))
                        ax[0,1].xaxis.set_minor_locator(MultipleLocator(10))
                        ax[0,1].set_ylim(0,reg_area_annual.max()*1.05/1e6)
                        ax[0,1].tick_params(direction='inout', right=True)
                        ax[0,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        #                               loc=(0.05,0.05),
                                       )    
                        
                        
                        # MASS BALANCE
                        reg_mbmwea_annual = ((reg_vol_annual[1:] - reg_vol_annual[:-1]) / reg_area_annual[:-1] * 
                                             pygem_prms.density_ice / pygem_prms.density_water)
                        ax[0,2].plot(years[0:-1], reg_mbmwea_annual, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                        ax[0,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                        ax[0,2].set_xlim(years.min(), years[0:-1].max())
                        ax[0,2].xaxis.set_major_locator(MultipleLocator(50))
                        ax[0,2].xaxis.set_minor_locator(MultipleLocator(10))
                        ax[0,2].tick_params(direction='inout', right=True)
                        
                        
                        # RUNOFF CHANGE 
                        reg_runoff_annual_fixed = reg_runoff_monthly_fixed.reshape(-1,12).sum(axis=1)
                        reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
                        ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
                        ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
                        ax[0,3].set_ylabel('Runoff (km$^{3}$)')
                        ax[0,3].set_xlim(years.min(), years[0:-1].max())
                        ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
                        ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
                        ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
                        ax[0,3].tick_params(direction='inout', right=True)
                        
                        
        
                        
                        # BINNED VOLUME
                        elev_bin_major = 1000
                        elev_bin_minor = 250
                        ymin = np.floor(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][0]] / elev_bin_major) * elev_bin_major
                        ymax = np.ceil(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][-1]] / elev_bin_major) * elev_bin_major
                        ax[1,0].plot(reg_vol_annual_binned[:,0]/1e9, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                        ax[1,0].plot(reg_vol_annual_binned[:,-1]/1e9, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                        if not reg_vol_annual_bd is None and not nodebris:
                            ax[1,0].plot(reg_vol_annual_binned_bd[:,0]/1e9, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                            ax[1,0].plot(reg_vol_annual_binned_bd[:,-1]/1e9, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,0].set_ylabel('Elevation (m)')
                        ax[1,0].set_xlabel('Volume (km$^{3}$)')
                        ax[1,0].set_xlim(0, reg_vol_annual_binned.max()/1e9)
                        ax[1,0].set_ylim(ymin, ymax)
                        ax[1,0].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                        ax[1,0].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                        ax[1,0].tick_params(direction='inout', right=True)
                        ax[1,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        #                               loc=(0.05,0.05),
                                       ) 
        
                        # BINNED AREA
                        ax[1,1].plot(reg_area_annual_binned[:,0]/1e6, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                        ax[1,1].plot(reg_area_annual_binned[:,-1]/1e6, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                        if not reg_area_annual_binned_bd is None and not nodebris:
                            ax[1,1].plot(reg_area_annual_binned_bd[:,0]/1e6, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                            ax[1,1].plot(reg_area_annual_binned_bd[:,-1]/1e6, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,1].set_ylabel('Elevation (m)')
                        ax[1,1].set_xlabel('Area (km$^{2}$)')
                        ax[1,1].set_xlim(0, reg_area_annual_binned.max()/1e6)
                        ax[1,1].set_ylim(ymin, ymax)
                        ax[1,1].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                        ax[1,1].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                        ax[1,1].tick_params(direction='inout', right=True)
                        ax[1,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        #                               loc=(0.05,0.05),
                                       ) 
        
                        # CLIMATIC MASS BALANCE GRADIENT
                        reg_mbclim_annual_binned_mwea = reg_mbclim_annual_binned / reg_area_annual_binned
                        ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,0], elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                        ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,-2], elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years.min()))
                        ax[1,2].set_ylabel('Elevation (m)')
                        ax[1,2].set_xlabel('$b_{clim}$ (m w.e. yr$^{-1}$)')
                        ax[1,2].set_ylim(ymin, ymax)
                        ax[1,2].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                        ax[1,2].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                        ax[1,2].tick_params(direction='inout', right=True)           
                        ax[1,2].axvline(0, color='k', linewidth=0.25)
                        
                        
                        # RUNOFF COMPONENTS
        #                reg_offglac_melt_annual = reg_offglac_melt_monthly.reshape(-1,12).sum(axis=1)
        #                reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
        #                ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
        #                ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
        #                ax[0,3].set_ylabel('Runoff (km$^{3}$)')
        #                ax[0,3].set_xlim(years.min(), years[0:-1].max())
        #                ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
        #                ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
        #                ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
        #                ax[0,3].tick_params(direction='inout', right=True)
                        
                        
                        # ELA
                        ela_min = np.floor(np.min(reg_ela_annual[0:-1]) / 100) * 100
                        ela_max = np.ceil(np.max(reg_ela_annual[0:-1]) / 100) * 100
                        ax[2,0].plot(years[0:-1], reg_ela_annual[0:-1], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                        ax[2,0].set_ylabel('ELA (m)')
                        ax[2,0].set_xlim(years.min(), years[0:-1].max())
        #                ax[2,0].set_ylim(ela_min, ela_max)
                        ax[2,0].tick_params(direction='inout', right=True)
                        
                        
    #                    # AAR
    #                    ax[2,1].plot(years, reg_aar_annual, color='k', linewidth=0.5, zorder=4, label=str(years.min()))
    #                    ax[2,1].set_ylabel('AAR (-)')
    #                    ax[2,1].set_ylim(0,1)
    #                    ax[2,1].set_xlim(years.min(), years[0:-1].max())
    #                    ax[2,1].tick_params(direction='inout', right=True)
                        
                        
                        # MASS BALANCE COMPONENTS
                        # - these are only meant for monthly and/or relative purposes 
                        #   mass balance from volume change should be used for annual changes
                        reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
                        # Refreeze
                        reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
                        # Melt
                        reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
                        # Frontal Ablation
                        reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
                        # Periods
                        if reg_acc_annual.shape[0] == 101:
                            period_yrs = 20
                            periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
                            reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
                            reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
                            reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
                            reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
                            reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
                            
                            # Convert to mwea
                            reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
                            reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
                            reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
                            reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
                            reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
                            reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
                        else:
                            assert True==False, 'Set up for different time periods'
        
                        # Plot
                        ax[2,2].bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='refreeze', zorder=2)
                        ax[2,2].bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='acc', zorder=3)
                        if not reg_frontalablation_periods_mwea.sum() == 0:
                            ax[2,2].bar(periods, -reg_frontalablation_periods_mwea, color='red', width=period_yrs/2-1, label='frontal ablation', zorder=3)
                        ax[2,2].bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='melt', zorder=2)
                        ax[2,2].bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='total', zorder=1)
                        ax[2,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                        ax[2,2].set_xlim(years.min(), years[0:-1].max())
                        ax[2,2].xaxis.set_major_locator(MultipleLocator(100))
                        ax[2,2].xaxis.set_minor_locator(MultipleLocator(20))
                        ax[2,2].yaxis.set_major_locator(MultipleLocator(1))
                        ax[2,2].yaxis.set_minor_locator(MultipleLocator(0.25))
                        ax[2,2].tick_params(direction='inout', right=True)
                        ax[2,2].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                                       loc=(1.2,0.25)) 
                        
                        
                        # Remove plot in lower right
                        fig.delaxes(ax[2,3])
                        
                        
                        # Title
                        fig.text(0.5, 0.95, rgi_reg_dict[reg] + ' (' + gcm_name + ' ' + rcp + ')', size=12, ha='center', va='top',)
                        
                        # Save figure
                        fig_fn = str(reg) + '_allplots_' + str(years.min()) + '-' + str(years.max()) + '_' + gcm_name + '_' + rcp + '.png'
                        fig.set_size_inches(8,6)
                        fig.savefig(fig_fp_reg + fig_fn, bbox_inches='tight', dpi=300)
        #%%
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                if len(bad_glacno_list[rcp][gcm_name]) > 0:
                    print('#',gcm_name, rcp)
                    print('glac_no=', bad_glacno_list[rcp][gcm_name])
            
#%%    

print('Total processing time:', time.time()-time_start, 's')



#%% ----- FRONTAL ABLATION ERROR ANALYSIS -----
if option_process_fa_err:


    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
    
    # Set up file
    fa_err_annual_gt_reg_dict = {}
    for reg in regions:
        fa_err_annual_gt_reg_dict[reg] = {}
    
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')

        years = None
        for rcp in rcps:
            
            fa_err_annual_gt_reg_dict[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:

#                # Add Order 2 regions
#                regO2_list = list(np.unique(main_glac_rgi.O2Region))
#                for regO2 in regO2_list:
#                    fa_err_annual_gt_reg_dict[reg][rcp][gcm_name][regO2] = None
                
                # Tidewater glaciers
                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                reg_fa_var_annual = None
                
                for nglac, glacno in enumerate(glacno_list):
                    if nglac%100 == 0:
                        print(gcm_name, rcp, glacno)
                            
                    # Load tidewater glacier ds
                    netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                    netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                    ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                    
                    # Years
                    if years is None:
                        years = ds_stats.year.values
                        normyear_idx = list(years).index(normyear)
                        
                    # Glacier frontal ablation error
                    #  Multiply by 1.4826 to account for nmad
                    glac_fa_nmad_annual = 1.4826*ds_stats.glac_frontalablation_monthly_mad.values[0,11::12]
                    glac_fa_var_annual = glac_fa_nmad_annual**2
                    
                    if reg_fa_var_annual is None:
                        reg_fa_var_annual = glac_fa_var_annual
                    else:
                        reg_fa_var_annual += glac_fa_var_annual
                    
                
                # Report the mean annual frontal ablation from 2015-2100 assuming perfect correlation in each region
                reg_fa_std_annual_gt = reg_fa_var_annual**0.5/1e9
                
                fa_err_annual_gt_reg_dict[reg][rcp][gcm_name] = reg_fa_std_annual_gt
                
                print('\n',gcm_name, rcp, 'mean fa gta:', np.round(np.mean(reg_fa_std_annual_gt),4),'\n')
                
        
    #%% Multi-GCM aggregation
    ds_multigcm_fa_std = {}
    for reg in regions:
        ds_multigcm_fa_std[reg] = {}
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            # Multi-GCM mean annual frontal ablation based on model parameter uncertainty
            reg_fa_gta_std_multigcm = None
#            reg_fa_gta_2000_2020_std_multigcm_list = []
#            reg_fa_gta_2015_2100_std_multigcm_list = []
            for gcm_name in gcm_names:
            
                reg_fa_std_annual_gt = fa_err_annual_gt_reg_dict[reg][rcp][gcm_name]
                
                if reg_fa_gta_std_multigcm is None:
                    reg_fa_gta_std_multigcm = reg_fa_std_annual_gt[np.newaxis,:]
                else:
                    reg_fa_gta_std_multigcm = np.vstack((reg_fa_gta_std_multigcm, reg_fa_std_annual_gt[np.newaxis,:]))
                
#                reg_fa_gta_2000_2020_std_multigcm_list.append(reg_fa_std_annual_gt[0:20])
#                reg_fa_gta_2015_2100_std_multigcm_list.append(reg_fa_std_annual_gt[normyear_idx:])
#                
#            reg_fa_gta_2000_2020_std_multigcm_list = []
#            reg_fa_gta_2015_2100_std_multigcm_list = []
            
            # Median of the multi-GCM annual frontal ablation standard deviations
            ds_multigcm_fa_std[reg][rcp] = reg_fa_gta_std_multigcm
            
            
    #%% Export data
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms',
                          'fa_gta_2000-2020_std', 'fa_gta_2015-2100_std']
    
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(stats_overview_cns))), columns=stats_overview_cns)
    
    ncount = 0
    for nreg, reg in enumerate(regions):
        for rcp in rcps:
            
            reg_fa_gta_std_multigcm = ds_multigcm_fa_std[reg][rcp]
            reg_fa_gta_std_multigcm_med = np.median(reg_fa_gta_std_multigcm, axis=0)

            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = rcp
            stats_overview_df.loc[ncount,'n_gcms'] = reg_fa_gta_std_multigcm.shape[0]
            stats_overview_df.loc[ncount,'fa_gta_2000-2020_std'] = np.mean(reg_fa_gta_std_multigcm_med[0:20])
            stats_overview_df.loc[ncount,'fa_gta_2015-2100_std'] = np.mean(reg_fa_gta_std_multigcm_med[normyear_idx:])
            
            ncount += 1

    stats_overview_df.to_csv(csv_fp + 'fa_annual_std_stats.csv', index=False)
                    
    assert 1==0, 'add global stats'

print('Total processing time:', time.time()-time_start, 's')

      
        #%%
                    
# ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
#fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, 
#                       gridspec_kw = {'wspace':0.7, 'hspace':0.5})
#                                
## RUNOFF CHANGE 
#years = np.arange(0,102)
#reg_runoff_annual_fixed = B_annual + A_annual
#reg_runoff_annual_moving = A_annual
#ax[0,0].plot(years[0:-1], reg_runoff_annual_fixed, color='b', linewidth=1, zorder=4, label='Fixed')
#ax[0,0].plot(years[0:-1], reg_runoff_annual_moving, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
#ax[0,0].set_ylabel('Runoff (km$^{3}$)')
#plt.show
        
#%%
#import zipfile
#zip_fp = '/Users/drounce/Documents/HiMAT/climate_data/cmip6/'
#zip_fn = 'BCC-CSM2-MR.zip'
#with zipfile.ZipFile(zip_fp + zip_fn, 'r') as zip_ref:
#    zip_ref.extractall(zip_fp)
        
#%% ----- MISSING DIFFERENT RCPS/GCMS -----
# Need to run script twice and comment out the processing (cheap shortcut)
#missing_rcp26 = glacno_list_missing.copy()
#missing_rcp45 = glacno_list_missing.copy()
#A = np.setdiff1d(missing_rcp45, missing_rcp26).tolist()
#print(A)
        
#%% ----- MOVE FILES -----
if option_move_files:
    regions = [19]
    for reg in regions:
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
            
                print('moving', gcm_name, rcp)
        
                # Filepath where glaciers are stored
                netcdf_fp_binned = '/Users/drounce/Documents/HiMAT/Output/simulations/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = '/Users/drounce/Documents/HiMAT/Output/simulations/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
#                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '-trial1/' + gcm_name + '/' + rcp + '/binned/'
#                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '-trial1/' + gcm_name + '/' + rcp + '/stats/'
                
                move_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                move_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                
                if os.path.exists(netcdf_fp_binned):
                    binned_fns_2move = []
                    for i in os.listdir(netcdf_fp_binned):
                        if i.endswith('.nc'):
                            binned_fns_2move.append(i)
                    binned_fns_2move = sorted(binned_fns_2move)
            
                    if len(binned_fns_2move) > 0:
                        for i in binned_fns_2move:
                            shutil.move(netcdf_fp_binned + i, move_binned + i)
                    
                    stats_fns_2move = []
                    for i in os.listdir(netcdf_fp_stats):
                        if i.endswith('.nc'):
                            stats_fns_2move.append(i)
                    stats_fns_2move = sorted(stats_fns_2move)
            
                    if len(stats_fns_2move) > 0:
                        for i in stats_fns_2move:
                            shutil.move(netcdf_fp_stats + i, move_stats + i)
        
        
#%%
if option_swap_calving_sims:
    
#    regions = [1, 3, 4, 5, 7, 9, 17, 19]
    regions = [2]
    
    option_nps_sims = True      # Option to pull out NPS simulations
    if option_nps_sims:
#        regions = [1]
#        nps_fp = '/Users/drounce/Documents/HiMAT/NPS_AK/simulations/'
#        nps_df_fn = '/Users/drounce/Documents/HiMAT/NPS_AK/rgiids_nps.csv'
        regions = [2]
        nps_fp = '/Users/drounce/Documents/HiMAT/NPS_AK/simulations/'
        nps_df_fn = '/Users/drounce/Documents/HiMAT/Menounos/rgiids_list.csv'
        nps_df = pd.read_csv(nps_df_fn)
        nps_rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    zipped_fp_cmip5 = '/Volumes/LaCie/globalsims_backup/simulations-cmip5/_zipped/'
    zipped_fp_cmip6 = '/Volumes/LaCie/globalsims_backup/simulations-cmip6/_zipped/'
    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
    rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['rcp26', 'rcp45', 'rcp85']
#    rcps = ['rcp26', 'rcp45', 'rcp85', 'ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']

    for reg in regions:
        # Load glaciers
        glacno_list = []
        
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
        
            for gcm_name in gcm_names:
                print(reg, rcp, gcm_name)
                
                # Filename
                if rcp in ['rcp26','rcp45','rcp85']:
                    zipped_fp = zipped_fp_cmip5
                elif rcp in ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']:
                    zipped_fp = zipped_fp_cmip6
                zipped_stats_fp = zipped_fp + str(reg).zfill(2) + '/stats/'
                zipped_stats_fn = gcm_name + '_' + rcp + '_stats.zip'
                zipped_binned_fp = zipped_fp + str(reg).zfill(2) + '/binned/'
                zipped_binned_fn = gcm_name + '_' + rcp + '_binned.zip'
                
                # Copy file
                copy_fp = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/'
                
                if not os.path.exists(copy_fp):
                    os.makedirs(copy_fp, exist_ok=True)                
                shutil.copy(zipped_stats_fp + zipped_stats_fn, copy_fp)
                shutil.copy(zipped_binned_fp + zipped_binned_fn, copy_fp)
                
                # Unzip filepath
                unzip_stats_fp = copy_fp + gcm_name + '/' + rcp + '/stats/'
                if not os.path.exists(unzip_stats_fp):
                    os.makedirs(unzip_stats_fp)
                with zipfile.ZipFile(copy_fp + zipped_stats_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_stats_fp)
                
                unzip_binned_fp = copy_fp + gcm_name + '/' + rcp + '/binned/'
                if not os.path.exists(unzip_binned_fp):
                    os.makedirs(unzip_binned_fp)
                with zipfile.ZipFile(copy_fp + zipped_binned_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_binned_fp)
                
                # Remove zipped file
                os.remove(copy_fp + zipped_stats_fn)
                os.remove(copy_fp + zipped_binned_fn)
                

                # Swap in calving runs
                calving_stats_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                calving_binned_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                
                if os.path.exists(calving_stats_fp):
                    for i in os.listdir(calving_stats_fp):
                        if i.endswith('all.nc'):
                            shutil.copy(calving_stats_fp + i, unzip_stats_fp)
                    
                    for i in os.listdir(calving_binned_fp):
                        if i.endswith('binned.nc'):
                            shutil.copy(calving_binned_fp + i, unzip_binned_fp)
                   
                
                # ----- NPS COPY FILES OF INTEREST -----
                # Pull out glaciers of interest in AK
                if option_nps_sims and rcp in nps_rcps:
                    if not os.path.exists(nps_fp):
                        os.makedirs(nps_fp, exist_ok=True) 
                    
                    nps_fp_stats = nps_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                    nps_fp_binned = nps_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                    
                    if not os.path.exists(nps_fp_stats):
                        os.makedirs(nps_fp_stats, exist_ok=True) 
                    if not os.path.exists(nps_fp_binned):
                        os.makedirs(nps_fp_binned, exist_ok=True) 
                    
                    # List of filenames available
                    copy_stats_fns = []
                    for i in os.listdir(unzip_stats_fp):
                        if i.endswith('_all.nc'):
                            copy_stats_fns.append(i)
                    copy_stats_fns = sorted(copy_stats_fns)
                    
                    copy_binned_fns = []
                    for i in os.listdir(unzip_binned_fp):
                        if i.endswith('_binned.nc'):
                            copy_binned_fns.append(i)
                    copy_binned_fns = sorted(copy_binned_fns)
                    
                    # RGIIds to copy
                    rgiids = list(nps_df.RGIId)
                    glacno_list = sorted([str(int(x.split('-')[1].split('.')[0])) + '.' + x.split('-')[1].split('.')[1] for x in rgiids])
                    for glacno in glacno_list:
                        
                        # Find the binned and stats filenames to copy
                        glac_copy_binned_fn_list = [x for x in copy_binned_fns if x.startswith(glacno)]
                        if len(glac_copy_binned_fn_list) > 0:
                            glac_copy_binned_fn = glac_copy_binned_fn_list[0]
                            # Copy file
                            shutil.copy(unzip_binned_fp + glac_copy_binned_fn, nps_fp_binned)
                            
                        glac_copy_stats_fn_list = [x for x in copy_stats_fns if x.startswith(glacno)]
                        if len(glac_copy_stats_fn_list) > 0:
                            glac_copy_stats_fn = glac_copy_stats_fn_list[0]
                            # Copy file
                            shutil.copy(unzip_stats_fp + glac_copy_stats_fn, nps_fp_stats)
                # ----- END NPS COPYING -----
                
                
                
                # Zip directory
                shutil.make_archive(copy_fp + zipped_stats_fn.replace('.zip',''), 'zip', unzip_stats_fp)
                shutil.make_archive(copy_fp + zipped_binned_fn.replace('.zip',''), 'zip', unzip_binned_fp)
                

                # Remove unzipped netcdf files
                shutil.rmtree(copy_fp + gcm_name)
                

                # Remove zipped file on external hard drive
                os.remove(zipped_stats_fp + zipped_stats_fn)
                os.remove(zipped_binned_fp + zipped_binned_fn)
                
                
                # Copy updated files to external hard drive
                shutil.copy(copy_fp + zipped_stats_fn, zipped_stats_fp)
                shutil.copy(copy_fp + zipped_binned_fn, zipped_binned_fp)
                
                
                # Delete zipped files on local computer
                os.remove(copy_fp + zipped_stats_fn)
                os.remove(copy_fp + zipped_binned_fn)
                



if option_extract_sims:
    
#    regions = [1, 3, 4, 5, 7, 9, 17, 19]
    regions = [2]
    
    nps_fp = '/Users/drounce/Documents/HiMAT/Menounos/simulations/'
    nps_df_fn = '/Users/drounce/Documents/HiMAT/Menounos/rgiids_list.csv'
    nps_df = pd.read_csv(nps_df_fn)
    nps_rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    zipped_fp_cmip5 = '/Volumes/LaCie/globalsims_backup/simulations-cmip5/_zipped/'
    zipped_fp_cmip6 = '/Volumes/LaCie/globalsims_backup/simulations-cmip6/_zipped/'
    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
    rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['rcp26', 'rcp45', 'rcp85']
#    rcps = ['rcp26', 'rcp45', 'rcp85', 'ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    for reg in regions:
        # Load glaciers
        glacno_list = []
        
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
        
            for gcm_name in gcm_names:
                print(reg, rcp, gcm_name)
                
                # Filename
                if rcp in ['rcp26','rcp45','rcp85']:
                    zipped_fp = zipped_fp_cmip5
                elif rcp in ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']:
                    zipped_fp = zipped_fp_cmip6
                zipped_stats_fp = zipped_fp + str(reg).zfill(2) + '/stats/'
                zipped_stats_fn = gcm_name + '_' + rcp + '_stats.zip'
                zipped_binned_fp = zipped_fp + str(reg).zfill(2) + '/binned/'
                zipped_binned_fn = gcm_name + '_' + rcp + '_binned.zip'
                
                # Copy file path
                copy_fp = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/'
                
                if not os.path.exists(copy_fp):
                    os.makedirs(copy_fp, exist_ok=True)                
                shutil.copy(zipped_stats_fp + zipped_stats_fn, copy_fp)
                shutil.copy(zipped_binned_fp + zipped_binned_fn, copy_fp)
                
                # Unzip filepath
                unzip_stats_fp = copy_fp + gcm_name + '/' + rcp + '/stats/'
                if not os.path.exists(unzip_stats_fp):
                    os.makedirs(unzip_stats_fp)
                with zipfile.ZipFile(copy_fp + zipped_stats_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_stats_fp)
                
                unzip_binned_fp = copy_fp + gcm_name + '/' + rcp + '/binned/'
                if not os.path.exists(unzip_binned_fp):
                    os.makedirs(unzip_binned_fp)
                with zipfile.ZipFile(copy_fp + zipped_binned_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_binned_fp)
                
                # Remove zipped file
                os.remove(copy_fp + zipped_stats_fn)
                os.remove(copy_fp + zipped_binned_fn)
                

                # ----- NPS COPY FILES OF INTEREST -----
                if not os.path.exists(nps_fp):
                    os.makedirs(nps_fp, exist_ok=True) 
                
                nps_fp_stats = nps_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                nps_fp_binned = nps_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                
                if not os.path.exists(nps_fp_stats):
                    os.makedirs(nps_fp_stats, exist_ok=True) 
                if not os.path.exists(nps_fp_binned):
                    os.makedirs(nps_fp_binned, exist_ok=True) 
                
                # List of filenames available
                copy_stats_fns = []
                for i in os.listdir(unzip_stats_fp):
                    if i.endswith('_all.nc'):
                        copy_stats_fns.append(i)
                copy_stats_fns = sorted(copy_stats_fns)
                
                copy_binned_fns = []
                for i in os.listdir(unzip_binned_fp):
                    if i.endswith('_binned.nc'):
                        copy_binned_fns.append(i)
                copy_binned_fns = sorted(copy_binned_fns)
                
                # RGIIds to copy
                rgiids = list(nps_df.RGIId)
                glacno_list = sorted([str(int(x.split('-')[1].split('.')[0])) + '.' + x.split('-')[1].split('.')[1] for x in rgiids])
                for glacno in glacno_list:
                    
                    # Find the binned and stats filenames to copy
                    glac_copy_binned_fn_list = [x for x in copy_binned_fns if x.startswith(glacno)]
                    if len(glac_copy_binned_fn_list) > 0:
                        glac_copy_binned_fn = glac_copy_binned_fn_list[0]
                        # Copy file
                        shutil.copy(unzip_binned_fp + glac_copy_binned_fn, nps_fp_binned)
                        
                    glac_copy_stats_fn_list = [x for x in copy_stats_fns if x.startswith(glacno)]
                    if len(glac_copy_stats_fn_list) > 0:
                        glac_copy_stats_fn = glac_copy_stats_fn_list[0]
                        # Copy file
                        shutil.copy(unzip_stats_fp + glac_copy_stats_fn, nps_fp_stats)
                # ----- END NPS COPYING -----
                
                # Remove unzipped netcdf files
                shutil.rmtree(copy_fp + gcm_name)

                
#%% ----- EXPORT TIME SERIES OF DATA TO SUPPORT REVIEW PRIOR TO UPLOADING AT NSIDC -----
if option_export_timeseries:
    
    netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
    export_fp = netcdf_fp_cmip5 + '_review_data/'
    
    
    if not os.path.exists(export_fp):
        os.makedirs(export_fp)
    
    for reg in regions:
        
        for rcp in rcps:
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                load_bwl = True
                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
                    pickle_fp_reg =  (netcdf_fp_cmip5_land + '../analysis/pickle/' + str(reg).zfill(2) + 
                                      '/O1Regions/' + gcm_name + '/' + rcp + '/')
                    load_bwl = False
                else:
                    pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                
                # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                # Volume below sea level
                if load_bwl:
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl = pickle.load(f)
                else:
                    reg_vol_annual_bwl = np.zeros(reg_vol_annual.shape)
                # Area 
                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                    reg_area_annual = pickle.load(f)
                # Mass balance: accumulation
                with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                    reg_acc_monthly = pickle.load(f)
                # Mass balance: refreeze
                with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                    reg_refreeze_monthly = pickle.load(f)
                # Mass balance: melt
                with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                    reg_melt_monthly = pickle.load(f)
                # Mass balance: frontal ablation
                with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                    reg_frontalablation_monthly = pickle.load(f)
                
                # ----- EXPORT THE DATA -----
                # CSV Filenames
                csv_fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.csv' 
                csv_fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.csv'
                csv_fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.csv'
                csv_fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.csv'
                csv_fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.csv'
                csv_fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.csv'
                csv_fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.csv'
                
                years = np.arange(2000,2102,1)
                years_4months = years.repeat(12)
                months_2000_2100 = np.tile(np.arange(1,13,1),int(reg_acc_monthly.shape[0]/12))
                yearmonths = [str(years_4months[x]) + '-' + str(months_2000_2100[x]) for x in np.arange(0,reg_acc_monthly.shape[0])]
                
                def export_array(reg_annual, columns_list, index_list, fp, fn):
                    """Export np.array to csv"""
                    if not os.path.exists(fp):
                        os.makedirs(fp)
                    reg_annual_df = pd.DataFrame(reg_annual.reshape(1,len(columns_list)), index=index_list, columns=columns_list)
                    reg_annual_df.to_csv(fp + fn)
                    
                export_array(reg_vol_annual, list(years), [reg], export_fp + 'reg_volume_annual/', csv_fn_reg_vol_annual)
                export_array(reg_vol_annual_bwl, list(years), [reg], export_fp + 'reg_volume_bwl_annual/', csv_fn_reg_vol_annual_bwl)
                export_array(reg_area_annual, list(years), [reg], export_fp + 'reg_area_annual/', csv_fn_reg_area_annual)
                export_array(reg_acc_monthly, yearmonths, [reg], export_fp + 'reg_acc_monthly/', csv_fn_reg_acc_monthly)
                export_array(reg_refreeze_monthly, yearmonths, [reg], export_fp + 'reg_refreeze_monthly/', csv_fn_reg_refreeze_monthly)
                export_array(reg_melt_monthly, yearmonths, [reg], export_fp + 'reg_melt_monthly/', csv_fn_reg_melt_monthly)
                export_array(reg_frontalablation_monthly, yearmonths, [reg], export_fp + 'reg_frontalablation_monthly/', csv_fn_reg_frontalablation_monthly)



#%%
if option_extract_area:
    
#    regions = [1, 3, 4, 5, 7, 9, 17, 19]
    regions = [8]
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/sims_08/'
    zipped_fp_cmip6 = '/Volumes/LaCie/globalsims_backup/simulations-cmip6/_zipped/'
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp585']
    
    for reg in regions:
        # Load glaciers
        glacno_list = []
        
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
        
            for gcm_name in gcm_names:
                print(reg, rcp, gcm_name)
                
                # Filename
                if rcp in ['rcp26','rcp45','rcp85']:
                    zipped_fp = zipped_fp_cmip5
                elif rcp in ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']:
                    zipped_fp = zipped_fp_cmip6
                zipped_stats_fp = zipped_fp + str(reg).zfill(2) + '/stats/'
                zipped_stats_fn = gcm_name + '_' + rcp + '_stats.zip'
                zipped_binned_fp = zipped_fp + str(reg).zfill(2) + '/binned/'
                zipped_binned_fn = gcm_name + '_' + rcp + '_binned.zip'
                
                # Copy file path
                copy_fp = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/'
                
                if not os.path.exists(copy_fp):
                    os.makedirs(copy_fp, exist_ok=True)                
                shutil.copy(zipped_stats_fp + zipped_stats_fn, copy_fp)
                shutil.copy(zipped_binned_fp + zipped_binned_fn, copy_fp)
                
                # Unzip filepath
                unzip_stats_fp = copy_fp + gcm_name + '/' + rcp + '/stats/'
                if not os.path.exists(unzip_stats_fp):
                    os.makedirs(unzip_stats_fp)
                with zipfile.ZipFile(copy_fp + zipped_stats_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_stats_fp)
                
                unzip_binned_fp = copy_fp + gcm_name + '/' + rcp + '/binned/'
                if not os.path.exists(unzip_binned_fp):
                    os.makedirs(unzip_binned_fp)
                with zipfile.ZipFile(copy_fp + zipped_binned_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_binned_fp)
                
                # Remove zipped file
                os.remove(copy_fp + zipped_stats_fn)
                os.remove(copy_fp + zipped_binned_fn)
        
        #%%
        # Glaciers
        # Load glaciers
        # All glaciers for fraction and missing
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        glacno_list = []
        glacno_list_gcmrcp_missing = {}
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                netcdf_fp_binned = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Check other file too
                glacno_binned_count = 0
                for i in os.listdir(netcdf_fp_binned):
                    if i.endswith('.nc'):
                        glacno_binned_count += 1
                print('  count of stats  files:', len(glacno_list_gcmrcp))
                print('  count of binned files:', glacno_binned_count)
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
                
                # Missing glaciers by gcm/rcp
                glacno_list_gcmrcp_missing[gcm_name + '-' + rcp] = (
                        sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list_gcmrcp).tolist()))
        #%%
        # CSV for each gcm/rcp
        for rcp in rcps:
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            
            reg_area_all = None
            for gcm_name in gcm_names:
                
                 # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                netcdf_fp_binned = netcdf_fp_cmip5 + '_copy/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                
                # Process data
                glac_stat_fns = []
                rgiid_list = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glac_stat_fns.append(i)
                        rgiid_list.append(i.split('_')[0])
                glac_stat_fns = sorted(glac_stat_fns)
                rgiid_list = sorted(rgiid_list)
                
                years = np.arange(2000,2102)
                reg_area_annual = pd.DataFrame(np.zeros((len(rgiid_list),years.shape[0])), index=rgiid_list, columns=years)
                for nglac, glac_stat_fn in enumerate(glac_stat_fns):
                    if nglac%1000==0:
                        print(reg, rcp, gcm_name, glac_stat_fn.split('_')[0])
                    ds = xr.open_dataset(netcdf_fp + glac_stat_fn)
                    reg_area_annual.iloc[nglac,:] = ds.glac_area_annual.values
                        
                reg_area_annual_fn = str(reg).zfill(2) + '_' + rcp + '_' + gcm_name + '_glac_area_annual.csv'
                csv_fp = netcdf_fp_cmip5 + '/_area/'
                if not os.path.exists(csv_fp):
                    os.makedirs(csv_fp)
                reg_area_annual.to_csv(csv_fp + reg_area_annual_fn)
                
                
        #%%
        # Multi-GCM mean
        for rcp in rcps:
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            
            reg_area_all = None
            for gcm_name in gcm_names:
                
                reg_area_annual_fn = str(reg).zfill(2) + '_' + rcp + '_' + gcm_name + '_glac_area_annual.csv'
                csv_fp = netcdf_fp_cmip5 + '/_area/'
                area_df = pd.read_csv(csv_fp + reg_area_annual_fn, index_col=0)
                
                if reg_area_all is None:
                    reg_area_all = area_df.values[np.newaxis,:,:]
                else:
                    reg_area_all = np.concatenate((reg_area_all, area_df.values[np.newaxis,:,:]), axis=0)
                
            # Mean
            reg_area_mean = reg_area_all.mean(0)
            reg_area_df = pd.DataFrame(reg_area_mean, index=glacno_list, columns=years)
            
            reg_area_fn = str(reg).zfill(2) + '_multigcm_mean_glac_area_annual_' + rcp + '.csv'
            reg_area_df.to_csv(csv_fp + reg_area_fn)
              