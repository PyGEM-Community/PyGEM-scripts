"""Run a model simulation."""
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import argparse
import copy
from datetime import date
import inspect
import multiprocessing
import os
import sys
import time
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# Local libraries
try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
    
import pygem.gcmbiasadj as gcmbiasadj
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.oggm_compat import single_flowline_glacier_directory_with_calving
from pygem.shop import debris 
from pygem import class_climate

import oggm
oggm_version = float(oggm.__version__[0:3])
from oggm import cfg, graphics, tasks, utils
if oggm_version > 1.301:
    from oggm.core.massbalance import apparent_mb_from_any_mb # Newer Version of OGGM
    from glaciermip3.oggm_flowline_wstop import FluxBasedModel
else:
    from oggm.core.climate import apparent_mb_from_any_mb # Older Version of OGGM
    from glaciermip3.oggm_v1p3_flowline_wstop import FluxBasedModel

from oggm.core.inversion import find_inversion_calving_from_any_mb

cfg.PARAMS['hydro_month_nh']=1
cfg.PARAMS['hydro_month_sh']=1
cfg.PARAMS['trapezoid_lambdas'] = 1

# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    gcm_list_fn (optional) : str
        text file that contains the climate data to be used in the model simulation
    gcm_name (optional) : str
        gcm name
    period (optional) : str
        period 
    glacno (optional) : str
        glacier number to run 
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn (optional) : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    batch_number (optional): int
        batch number used to differentiate output on supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    debug (optional) : int
        Switch for turning debug printing on or off (default = 0 (off))
    debug_spc (optional) : int
        Switch for turning debug printing of spc on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-period', action='store', type=str, default=None,
                        help='period used for model run')
    parser.add_argument('-glacno', action='store', type=str, default=None,
                        help='glacier number used for model run')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms.gcm_startyear,
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms.gcm_endyear,
                        help='start year for the model run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-batch_number', action='store', type=int, default=None,
                        help='Batch number used to differentiate output on supercomputer')
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    parser.add_argument('-debug_spc', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser


def main(list_packed_vars):
    """
    Model simulation
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]
    parser = getparser()
    args = parser.parse_args()
    
    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=1850, endyear=2100, spinupyears=0,
            option_wateryear='calendar')
    
    # ===== LOAD CLIMATE DATA =====
    shuffled_yrs_fullfn = pygem_prms.main_directory + '/../climate_data/glaciermip3/shuffling_order.csv'
    pd_shuffled_yrs = pd.read_csv(shuffled_yrs_fullfn, index_col=0)
    
    # template of shuffled climate values (that will be filled afterwards)
    periods = pd_shuffled_yrs.columns
    # Correct for regions that only need 2000 year sims
#    if not main_glac_rgi.O1Region.values[0] in [1,3,4,5,6,7,9,17,19]:
    if main_glac_rgi.O1Region.values[0] in [2,8,10,11,12,13,14,15,16,18]:
        pd_shuffled_yrs = pd_shuffled_yrs.loc[0:1999]
    
#    print('\n\nDELETE ME!\n\n')
#    pd_shuffled_yrs = pd_shuffled_yrs.loc[0:2000]
        
    simulation_years = pd_shuffled_yrs.index  # from 0 to 4999
    pd_empty_clim_template = pd.DataFrame(np.NaN, columns=periods, index=np.arange(0,len(simulation_years)*12))
    
    # open the right climate file
    if gcm_name in ['gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0']:
        ensemble = 'r1i1p1f1'
    elif gcm_name == 'ukesm1-0-ll':
        ensemble = 'r1i1p1f2'
    
    # Load air temperature data
    folder_output_tas = 'isimip3b_tasAdjust_monthly'
    
    # Here you have to change the path to the isimip3b data folder
    isimip_folder = pygem_prms.main_directory + '/../climate_data/glaciermip3/isimip3b/'

    # historical dataset 
    path_output_tas_hist = f'{isimip_folder}/{folder_output_tas}/{gcm_name}_{ensemble}_w5e5_historical_tasAdjust_global_monthly_1850_2014.nc'
    ds_tas_monthly_hist = xr.open_dataset(path_output_tas_hist)

    # Load precipitation data
    folder_output_pr = 'isimip3b_prAdjust_monthly'
    
    # historical dataset 
    path_output_pr_hist = f'{isimip_folder}/{folder_output_pr}/{gcm_name}_{ensemble}_w5e5_historical_prAdjust_global_monthly_1850_2014.nc'
    ds_pr_monthly_hist = xr.open_dataset(path_output_pr_hist)
    
    # SSP126
    path_output_tas_ssp126 = f'{isimip_folder}/{folder_output_tas}/{gcm_name}_{ensemble}_w5e5_ssp126_tasAdjust_global_monthly_2015_2100.nc'
    ds_tas_monthly_ssp126 = xr.open_dataset(path_output_tas_ssp126)
    path_output_pr_ssp126 = f'{isimip_folder}/{folder_output_pr}/{gcm_name}_{ensemble}_w5e5_ssp126_prAdjust_global_monthly_2015_2100.nc'
    ds_pr_monthly_ssp126 = xr.open_dataset(path_output_pr_ssp126)
    
    # SSP370
    path_output_tas_ssp370 = f'{isimip_folder}/{folder_output_tas}/{gcm_name}_{ensemble}_w5e5_ssp370_tasAdjust_global_monthly_2015_2100.nc'
    ds_tas_monthly_ssp370 = xr.open_dataset(path_output_tas_ssp370)
    path_output_pr_ssp370 = f'{isimip_folder}/{folder_output_pr}/{gcm_name}_{ensemble}_w5e5_ssp370_prAdjust_global_monthly_2015_2100.nc'
    ds_pr_monthly_ssp370 = xr.open_dataset(path_output_pr_ssp370)
    
    # SSP585
    path_output_tas_ssp585 = f'{isimip_folder}/{folder_output_tas}/{gcm_name}_{ensemble}_w5e5_ssp585_tasAdjust_global_monthly_2015_2100.nc'
    ds_tas_monthly_ssp585 = xr.open_dataset(path_output_tas_ssp585)
    path_output_pr_ssp585 = f'{isimip_folder}/{folder_output_pr}/{gcm_name}_{ensemble}_w5e5_ssp585_prAdjust_global_monthly_2015_2100.nc'
    ds_pr_monthly_ssp585 = xr.open_dataset(path_output_pr_ssp585)
    
    #%%
    # ----- REFERENCE CLIMATE DATA -----
    # Climate class
    # Reference GCM
    ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    ref_startyear = 2000
    ref_endyear = 2019
    dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear,
                                                spinupyears=pygem_prms.ref_spinupyears,
                                                option_wateryear='calendar')
    
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    ref_temp_all, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn,
                                                                          main_glac_rgi, dates_table_ref)
    ref_prec_all, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn,
                                                                          main_glac_rgi, dates_table_ref)
    ref_lr_all, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, 
                                                                        main_glac_rgi, dates_table_ref)
    ref_elev_all = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
    
    # ===== RUN MASS BALANCE =====
    # Number of simulations
    sim_iters = 1

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        reg_str = str(glacier_rgi_table.O1Region).zfill(2)
        rgiid = main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId']

        lon = main_glac_rgi.loc[glac,'CenLon']
        lat = main_glac_rgi.loc[glac,'CenLat']
        
        #%%
        ref_temp = ref_temp_all[glac,:][np.newaxis,:]
        ref_prec = ref_prec_all[glac,:][np.newaxis,:]
        ref_lr = ref_lr_all[glac,:][np.newaxis,:]
        ref_elev = np.array([ref_elev_all[glac]])

        # SSP scenarios
        for nssp, ssp in enumerate(['ssp126', 'ssp370' ,'ssp585']):
        # for nssp, ssp in enumerate(['ssp585']):
            
            # ----- TEMPERATURE -----
            # select the nearest grid point and get the annual means
            # (note: no weighting per month duration is performed for the annual mean)
            # Calculate historical data once
            if nssp == 0:
                ds_monthly_hist_tas = ds_tas_monthly_hist.sel(lon=lon, lat=lat, method='nearest').tasAdjust
                ds_monthly_hist_pr = ds_pr_monthly_hist.sel(lon=lon, lat=lat, method='nearest').prAdjust

            if ssp in ['ssp126']:
                ds_monthly_ssp_tas = ds_tas_monthly_ssp126.sel(lon=lon, lat=lat, method='nearest').tasAdjust
                ds_monthly_tas_1850_2100 = xr.concat([ds_monthly_hist_tas, ds_monthly_ssp_tas], dim='time')
                ds_monthly_ssp_pr = ds_pr_monthly_ssp126.sel(lon=lon, lat=lat, method='nearest').prAdjust
                ds_monthly_pr_1850_2100 = xr.concat([ds_monthly_hist_pr, ds_monthly_ssp_pr], dim='time')
            elif ssp in ['ssp370']:
                ds_monthly_ssp_tas = ds_tas_monthly_ssp370.sel(lon=lon, lat=lat, method='nearest').tasAdjust
                ds_monthly_tas_1850_2100 = xr.concat([ds_monthly_hist_tas, ds_monthly_ssp_tas], dim='time')
                ds_monthly_ssp_pr = ds_pr_monthly_ssp370.sel(lon=lon, lat=lat, method='nearest').prAdjust
                ds_monthly_pr_1850_2100 = xr.concat([ds_monthly_hist_pr, ds_monthly_ssp_pr], dim='time')
            elif ssp in ['ssp585']:
                ds_monthly_ssp_tas = ds_tas_monthly_ssp585.sel(lon=lon, lat=lat, method='nearest').tasAdjust
                ds_monthly_tas_1850_2100 = xr.concat([ds_monthly_hist_tas, ds_monthly_ssp_tas], dim='time')
                ds_monthly_ssp_pr = ds_pr_monthly_ssp585.sel(lon=lon, lat=lat, method='nearest').prAdjust
                ds_monthly_pr_1850_2100 = xr.concat([ds_monthly_hist_pr, ds_monthly_ssp_pr], dim='time')
              
            # ===== BIAS CORRECTIONS =====
            # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
            # Temperature bias correction
            gcm_temp_raw = ds_monthly_tas_1850_2100.values[np.newaxis,:] - 273.15
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp_raw,
                                                                        dates_table_ref, dates_table)
            # Precipitation bias correction
            # Convert units from kg m-2 s-1 to m d-1 to m per month
            #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
            gcm_prec_raw_mperday = ds_monthly_pr_1850_2100.values[np.newaxis,:]/1000*3600*24
            gcm_prec_raw = gcm_prec_raw_mperday * dates_table['daysinmonth'].values[np.newaxis,:]
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec_raw,
                                                                      dates_table_ref, dates_table)
            
            #%%
            # Shuffle
            pd_shuffle_clim_temp = pd_empty_clim_template.copy()
            pd_shuffle_clim_prec = pd_empty_clim_template.copy()
            # get the shuffled climate data for each experiment (time period)
            
            ncount = 0
            if nssp == 0:
                periods_subset = periods
            else:
                periods_subset = periods[4:]
            
            for p in periods_subset:
                gcm_temp = None
                for yr in pd_shuffled_yrs[p]:
                    idx_list = np.where(dates_table['year'] == yr)[0]

                    if gcm_temp is None:
                        gcm_temp = gcm_temp_adj[0,idx_list]
                        gcm_prec = gcm_prec_adj[0,idx_list]
                    else:
                        gcm_temp = np.concatenate((gcm_temp, gcm_temp_adj[0,idx_list]))
                        gcm_prec = np.concatenate((gcm_prec, gcm_prec_adj[0,idx_list]))
                
                ncount += 1
                        
                pd_shuffle_clim_temp[p] = gcm_temp
                pd_shuffle_clim_prec[p] = gcm_prec
                
            #%%
            # Rename columns to include SSPs
            pd_shuffle_clim_temp = pd_shuffle_clim_temp.rename(columns={'2021-2040':ssp+'_2021-2040',
                                                                        '2041-2060':ssp+'_2041-2060',
                                                                        '2061-2080':ssp+'_2061-2080',
                                                                        '2081-2100':ssp+'_2081-2100'})
            pd_shuffle_clim_prec = pd_shuffle_clim_prec.rename(columns={'2021-2040':ssp+'_2021-2040',
                                                                        '2041-2060':ssp+'_2041-2060',
                                                                        '2061-2080':ssp+'_2061-2080',
                                                                        '2081-2100':ssp+'_2081-2100'})
            if nssp == 0:
                pd_shuffle_clim_temp_all = pd_shuffle_clim_temp.copy()
                pd_shuffle_clim_prec_all = pd_shuffle_clim_prec.copy()
            else:
                columns_2append = [ssp + x for x in ['_2021-2040','_2041-2060','_2061-2080','_2081-2100']]
                pd_suffle_clim_temp_subset = pd_shuffle_clim_temp.loc[:,columns_2append]
                pd_shuffle_clim_temp_all = pd.concat([pd_shuffle_clim_temp_all,pd_suffle_clim_temp_subset], axis=1)
                
                pd_suffle_clim_prec_subset = pd_shuffle_clim_prec.loc[:,columns_2append]
                pd_shuffle_clim_prec_all = pd.concat([pd_shuffle_clim_prec_all,pd_suffle_clim_prec_subset], axis=1)

        # ----- LOOP THROUGH SYNTHETIC SIMULATIONS -----
        # Loop through each of the 5000 year synthetic datasets (16 for each glacier)
        if debug:
            print('periods:', pd_shuffle_clim_prec_all.columns)
        
        if args.period is None:
            periods_2run = list(pd_shuffle_clim_prec_all.columns)
        else:
            periods_2run = [args.period]
        
        if debug:
            print('periods to run:', periods_2run)
        
        for period in periods_2run:
        # for period in ['ssp126_2081-2100']:
        # for period in ['ssp585_2081-2100']:
#        for period in pd_shuffle_clim_prec_all.columns:    
            if debug:
                print('\n\nperiod:', period)
                
            # Run info
            contributor = 'Rounce'
            rgi_reg = 'rgi' + reg_str
            agg_level = 'glaciers'
            gcm = gcm_name
            if 'ssp' in period:
                ssp = period.split('_')[0]
                period_name = period.split('_')[1]
            else:
                ssp = 'hist'
                period_name = period
            
            filename = f'{contributor}_{rgi_reg}_{agg_level}_{period_name}_{gcm}_{ssp}_{glacier_str}.nc'    
            sim_fp = pygem_prms.output_filepath + 'simulations/' + reg_str + '/' + gcm_name + '/' + period + '/'
            
            if not os.path.exists(sim_fp + filename):
                # Load climate data
                gcm_temp = pd_shuffle_clim_temp_all[period].values[np.newaxis,:]
                gcm_prec = pd_shuffle_clim_prec_all[period].values[np.newaxis,:]
                gcm_elev = gcm_elev_adj
                gcm_tempstd = np.zeros(gcm_temp.shape)
                
                # Lapse rate (monthly average from reference climate data)
                gcm_lr_monthly_all = gcmbiasadj.monthly_avg_array_rolled(ref_lr, dates_table_ref, dates_table)
                gcm_lr = np.tile(gcm_lr_monthly_all[:,0:12], int(gcm_temp.shape[1]/12))
                
                # Number of years for model simulations
                nyears = int(gcm_temp.shape[1]/12)
                nyears_ref = int(dates_table_ref.shape[0]/12)
                if debug:
                    print('nyears:', nyears)
                
                #%%
                try:
                # for batman in [0]:
    
                    # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
                    if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                        gdir = single_flowline_glacier_directory(glacier_str, logging_level='CRITICAL')
                        gdir.is_tidewater = False
                        calving_k = None
                    else:
                        gdir = single_flowline_glacier_directory_with_calving(glacier_str, logging_level='CRITICAL')
                        gdir.is_tidewater = True
                        cfg.PARAMS['use_kcalving_for_inversion'] = True
                        cfg.PARAMS['use_kcalving_for_run'] = True
        
                    # Flowlines
                    fls = gdir.read_pickle('inversion_flowlines')
                    
                    # Reference gdir for ice thickness inversion
                    gdir_ref = copy.deepcopy(gdir)
                    ref_tempstd = gcm_tempstd[:,0:dates_table_ref.shape[0]]
                    gdir_ref.historical_climate = {'elev': ref_elev[0],
                                                'temp': ref_temp[0,:],
                                                'tempstd': ref_tempstd[0,:],
                                                'prec': ref_prec[0,:],
                                                'lr': ref_lr[0,:]}
                    gdir_ref.dates_table = dates_table_ref
                    
                    # Add climate data to glacier directory
                    gdir.historical_climate = {'elev': gcm_elev[0],
                                                'temp': gcm_temp[0,:],
                                                'tempstd': gcm_tempstd[0,:],
                                                'prec': gcm_prec[0,:],
                                                'lr': gcm_lr[0,:]}
        
                    # Synthetic ("syn") dates table
                    dates_table_syn = pd.DataFrame(np.zeros((gcm_temp.shape[1],len(dates_table.columns))), columns=dates_table.columns)
                    dates_table_syn['date'] = np.nan
                    dates_table_syn['year'] = np.repeat(np.arange(0,gcm_temp.shape[1]/12), 12)
                    dates_table_syn['month'] = np.tile(np.arange(0,12)+1, int(gcm_temp.shape[1]/12))
                    dates_table_syn['daysinmonth'] = np.tile(dates_table.loc[0:11,'daysinmonth'].values, int(gcm_temp.shape[1]/12))
                    dates_table_syn['wateryear'] = (np.repeat(np.arange(0,gcm_temp.shape[1]/12+1), 12))[3:3+gcm_temp.shape[1]]
                    dates_table_syn['season'] = np.tile(dates_table.loc[0:11,'season'].values, int(gcm_temp.shape[1]/12))
                    
                    gdir.dates_table = dates_table_syn
            
                    glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
                    
                    if debug:
                        print('glacier area [km2]:', glacier_area_km2.sum())
                    
                    if (fls is not None) and (glacier_area_km2.sum() > 0):
                        # Load model parameters
                        if pygem_prms.use_calibrated_modelparams:
                            
                            modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                            modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                            + '/')
                            modelprms_fullfn = modelprms_fp + modelprms_fn
            
                            assert os.path.exists(modelprms_fullfn), 'Calibrated parameters do not exist.'
                            with open(modelprms_fullfn, 'rb') as f:
                                modelprms_dict = pickle.load(f)
            
                            assert pygem_prms.option_calibration in modelprms_dict, ('Error: ' + pygem_prms.option_calibration +
                                                                                      ' not in modelprms_dict')
                            modelprms_all = modelprms_dict[pygem_prms.option_calibration]
                            # MCMC needs model parameters to be selected
                            if pygem_prms.option_calibration == 'MCMC':
                                if sim_iters == 1:
                                    modelprms_all = {'kp': [np.median(modelprms_all['kp']['chain_0'])],
                                                      'tbias': [np.median(modelprms_all['tbias']['chain_0'])],
                                                      'ddfsnow': [np.median(modelprms_all['ddfsnow']['chain_0'])],
                                                      'ddfice': [np.median(modelprms_all['ddfice']['chain_0'])],
                                                      'tsnow_threshold': modelprms_all['tsnow_threshold'],
                                                      'precgrad': modelprms_all['precgrad']}
                                
                            # Calving parameter
                            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                                calving_k = None
                            else:
                                # Load quality controlled frontal ablation data 
                                assert os.path.exists(pygem_prms.calving_fp + pygem_prms.calving_fn), 'Calibrated calving dataset does not exist'
                                calving_df = pd.read_csv(pygem_prms.calving_fp + pygem_prms.calving_fn)
                                calving_rgiids = list(calving_df.RGIId)
                                
                                # Use calibrated value if individual data available
                                if rgiid in calving_rgiids:
                                    calving_idx = calving_rgiids.index(rgiid)
                                    calving_k = calving_df.loc[calving_idx, 'calving_k']
                                    calving_k_nmad = calving_df.loc[calving_idx, 'calving_k_nmad']
                                # Otherwise, use region's median value
                                else:
                                    calving_df['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in calving_df.RGIId.values]
                                    calving_df_reg = calving_df.loc[calving_df['O1Region'] == int(reg_str), :]
                                    calving_k = np.median(calving_df_reg.calving_k)
                                    calving_k_nmad = 0
                                
                                if sim_iters == 1:
                                    calving_k_values = np.array([calving_k])
                                
                        if debug and gdir.is_tidewater:
                            print('calving_k:', calving_k)
                            
        
                        # Load OGGM glacier dynamics parameters (if necessary)
                        if pygem_prms.option_dynamics in ['OGGM', 'MassRedistributionCurves']:
        
                            # CFL number (may use different values for calving to prevent errors)
                            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                                cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number
                            else:
                                cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number_calving

                            if debug:
                                print('cfl number:', cfg.PARAMS['cfl_number'])
                                
                            if pygem_prms.use_reg_glena:
                                glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)                    
                                glena_O1regions = [int(x) for x in glena_df.O1Region.values]
                                assert glacier_rgi_table.O1Region in glena_O1regions, glacier_str + ' O1 region not in glena_df'
                                glena_idx = np.where(glena_O1regions == glacier_rgi_table.O1Region)[0][0]
                                glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                                fs = glena_df.loc[glena_idx,'fs']
            
                        # Loop through model parameters
                        count_exceed_boundary_errors = 0
                        for n_iter in range(sim_iters):
        
                            if debug:                    
                                print('n_iter:', n_iter)
                            
                            if not calving_k is None:
                                calving_k = calving_k_values[n_iter]
                                cfg.PARAMS['calving_k'] = calving_k
                                cfg.PARAMS['inversion_calving_k'] = calving_k
                            
        
                            # successful_run used to continue runs when catching specific errors
                            successful_run = True
                            
                            modelprms = {'kp': modelprms_all['kp'][n_iter],
                                          'tbias': modelprms_all['tbias'][n_iter],
                                          'ddfsnow': modelprms_all['ddfsnow'][n_iter],
                                          'ddfice': modelprms_all['ddfice'][n_iter],
                                          'tsnow_threshold': modelprms_all['tsnow_threshold'][n_iter],
                                          'precgrad': modelprms_all['precgrad'][n_iter]}
            
                            if debug:
                                print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                                      ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                                      ' tbias: ' + str(np.round(modelprms['tbias'],2)))
        
        
                            # ----- ICE THICKNESS INVERSION using OGGM -----
                            if not pygem_prms.option_dynamics is None:
                                # Apply inversion_filter on mass balance with debris to avoid negative flux
                                if pygem_prms.include_debris:
                                    inversion_filter = True
                                else:
                                    inversion_filter = False
                                    
                                # Perform inversion based on PyGEM MB using reference directory
                                mbmod_inv = PyGEMMassBalance(gdir_ref, modelprms, glacier_rgi_table,
                                                             hindcast=pygem_prms.hindcast,
                                                             debug=pygem_prms.debug_mb,
                                                             debug_refreeze=pygem_prms.debug_refreeze,
                                                             fls=fls, option_areaconstant=True,
                                                             inversion_filter=inversion_filter)
        #                        if debug:
        #                            h, w = gdir.get_inversion_flowline_hw()
        #                            mb_t0 = (mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls) * cfg.SEC_IN_YEAR * 
        #                                     pygem_prms.density_ice / pygem_prms.density_water) 
        #                            plt.plot(mb_t0, h, '.')
        #                            plt.ylabel('Elevation')
        #                            plt.xlabel('Mass balance (mwea)')
        #                            plt.show()

                                # Non-tidewater glaciers
                                if not gdir.is_tidewater or not pygem_prms.include_calving:
                                    # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
                                    apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears_ref))
                                    tasks.prepare_for_inversion(gdir)
                                    tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)

                                # Tidewater glaciers
                                else:
                                    out_calving = find_inversion_calving_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears_ref),
                                                                                      glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
                                        

                                # ----- INDENTED TO BE JUST WITH DYNAMICS -----
                                tasks.init_present_time_glacier(gdir) # adds bins below
                                debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
                
                                try:
                                    nfls = gdir.read_pickle('model_flowlines')
                                except FileNotFoundError as e:
                                    if 'model_flowlines.pkl' in str(e):
                                        tasks.compute_downstream_line(gdir)
                                        tasks.compute_downstream_bedshape(gdir)
                                        tasks.init_present_time_glacier(gdir) # adds bins below
                                        nfls = gdir.read_pickle('model_flowlines')
                                    else:
                                        raise
                            
                            # No ice dynamics options
                            else:
                                nfls = fls
                            
                            # Water Level
                            # Check that water level is within given bounds
                            cls = gdir.read_pickle('inversion_input')[-1]
                            th = cls['hgt'][-1]
                            vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
                            water_level = utils.clip_scalar(0, th - vmax, th - vmin) 
                            
                            # ------ MODEL WITH EVOLVING AREA ------
                            # Mass balance model
                            mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                                      hindcast=pygem_prms.hindcast,
                                                      debug=pygem_prms.debug_mb,
                                                      debug_refreeze=pygem_prms.debug_refreeze,
                                                      fls=nfls, option_areaconstant=False)
        
                            # Glacier dynamics model
                            if pygem_prms.option_dynamics == 'OGGM':
                                if debug:
                                    print('OGGM GLACIER DYNAMICS!')
                                    
                                ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, 
                                                          glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                          is_tidewater=gdir.is_tidewater,
                                                          water_level=water_level
                                                          )
                                
                                if debug:
                                    graphics.plot_modeloutput_section(ev_model)
                                    plt.show()    
                                    print('nyears:', nyears)
        
        
                                try:
#                                for batman in [0]:
                                    if oggm_version > 1.301:
                                        diag = ev_model.run_until_and_store(nyears, stop_criterion=True)
                                    else:
                                        _, diag = ev_model.run_until_and_store(nyears, stop_criterion=True)
                                    ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                                    ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
                                    
                                    # Record frontal ablation for tidewater glaciers and update total mass balance
                                    if gdir.is_tidewater:
                                        # Glacier-wide frontal ablation (m3 w.e.)
                                        # - note: diag.calving_m3 is cumulative calving
                                        if debug:
                                            print('\n\ndiag.calving_m3:', diag.calving_m3.values)
                                            print('calving_m3_since_y0:', ev_model.calving_m3_since_y0)
                                        calving_m3_annual = ((diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1]) * 
                                                              pygem_prms.density_ice / pygem_prms.density_water)
                                        for n in np.arange(calving_m3_annual.shape[0]):
                                            ev_model.mb_model.glac_wide_frontalablation[12*n+11] = calving_m3_annual[n]
        
                                        # Glacier-wide total mass balance (m3 w.e.)
                                        ev_model.mb_model.glac_wide_massbaltotal = (
                                                ev_model.mb_model.glac_wide_massbaltotal  - ev_model.mb_model.glac_wide_frontalablation)
                                        
                                        if debug:
                                            print('avg calving_m3:', calving_m3_annual.sum() / nyears)
                                            print('avg frontal ablation [Gta]:', 
                                                  np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                                            print('avg frontal ablation [Gta]:', 
                                                  np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                                    
                                    if debug:
                                        print('successful oggm dynamical run')
                                    
                                except RuntimeError as e:
                                    print('except runtime error')
                                    if 'Glacier exceeds domain boundaries' in repr(e):
                                        count_exceed_boundary_errors += 1
                                        successful_run = False
                                        
                                        # LOG FAILURE
                                        fail_domain_fp = (pygem_prms.output_sim_fp + 'fail-exceed_domain/' + reg_str + '/' 
                                                          + gcm_name + '/')
                                        if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                                            if 'ssp' in period:
                                                ssp = period.split('_')[0]
                                            else:
                                                ssp = 'hist'
                                            fail_domain_fp += ssp + '/'
                                        if not os.path.exists(fail_domain_fp):
                                            os.makedirs(fail_domain_fp, exist_ok=True)
                                        txt_fn_fail = glacier_str + "-sim_failed.txt"
                                        with open(fail_domain_fp + txt_fn_fail, "w") as text_file:
                                            text_file.write(glacier_str + ' failed to complete ' + 
                                                            str(count_exceed_boundary_errors) + ' simulations')
#                                    elif gdir.is_tidewater:
                                    try:
                                        if debug:
                                            print('OGGM dynamics failed, using mass redistribution curves')
                                        # Mass redistribution curves glacier dynamics model
                                        ev_model = MassRedistributionCurveModel(
                                                        nfls, mb_model=mbmod, y0=0,
                                                        glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                        is_tidewater=gdir.is_tidewater,
                                                        water_level=water_level
                                                        )
                                        if oggm_version > 1.301:
                                            diag = ev_model.run_until_and_store(nyears, stop_criterion=True)
                                        else:
                                            _, diag = ev_model.run_until_and_store(nyears, stop_criterion=True)
                                        ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                                        ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values
                        
                                        # Record frontal ablation for tidewater glaciers and update total mass balance
                                        # Update glacier-wide frontal ablation (m3 w.e.)
                                        ev_model.mb_model.glac_wide_frontalablation = ev_model.mb_model.glac_bin_frontalablation.sum(0)
                                        # Update glacier-wide total mass balance (m3 w.e.)
                                        ev_model.mb_model.glac_wide_massbaltotal = (
                                                ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation)
        
                                        if debug:
                                            print('avg frontal ablation [Gta]:', 
                                                  np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                                            print('avg frontal ablation [Gta]:', 
                                                  np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                                        
                                    except RuntimeError as e:
                                        if 'Glacier exceeds domain boundaries' in repr(e):
                                            count_exceed_boundary_errors += 1
                                            successful_run = False
                                            
                                            # LOG FAILURE
                                            fail_domain_fp = (pygem_prms.output_sim_fp + 'fail-exceed_domain/' + reg_str + '/' 
                                                              + gcm_name + '/')
                                            if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                                                if 'ssp' in period:
                                                    ssp = period.split('_')[0]
                                                else:
                                                    ssp = 'hist'
                                                fail_domain_fp += ssp + '/'
                                            if not os.path.exists(fail_domain_fp):
                                                os.makedirs(fail_domain_fp, exist_ok=True)
                                            txt_fn_fail = glacier_str + "-sim_failed.txt"
                                            with open(fail_domain_fp + txt_fn_fail, "w") as text_file:
                                                text_file.write(glacier_str + ' failed to complete ' + 
                                                                str(count_exceed_boundary_errors) + ' simulations')
                                        
                                        else:
                                            raise
        
                                except:
                                    if gdir.is_tidewater:
                                        if debug:
                                            print('OGGM dynamics failed, using mass redistribution curves')
                                                                        # Mass redistribution curves glacier dynamics model
                                        ev_model = MassRedistributionCurveModel(
                                                        nfls, mb_model=mbmod, y0=0,
                                                        glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                        is_tidewater=gdir.is_tidewater,
                                                        water_level=water_level
                                                        )
                                        if oggm_version > 1.301:
                                            diag = ev_model.run_until_and_store(nyears, stop_criterion=True)
                                        else:
                                            _, diag = ev_model.run_until_and_store(nyears, stop_criterion=True)
                                        ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                                        ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values
                        
                                        # Record frontal ablation for tidewater glaciers and update total mass balance
                                        # Update glacier-wide frontal ablation (m3 w.e.)
                                        ev_model.mb_model.glac_wide_frontalablation = ev_model.mb_model.glac_bin_frontalablation.sum(0)
                                        # Update glacier-wide total mass balance (m3 w.e.)
                                        ev_model.mb_model.glac_wide_massbaltotal = (
                                                ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation)
        
                                        if debug:
                                            print('avg frontal ablation [Gta]:', 
                                                  np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                                            print('avg frontal ablation [Gta]:', 
                                                  np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                        
                                    else:
                                        raise
        
        
                            # Record output for successful runs
                            if successful_run:
                                
                                if not pygem_prms.option_dynamics is None:
                                    if debug:
                                        graphics.plot_modeloutput_section(ev_model)
                    #                    graphics.plot_modeloutput_map(gdir, model=ev_model)
                                        plt.figure()
                                        diag.volume_m3.plot()
                                        plt.figure()
                                        plt.show()
                    
                                    # Post-process data to ensure mass is conserved and update accordingly for ignored mass losses
                                    #  ignored mass losses occur because mass balance model does not know ice thickness and flux divergence
                                    area_initial = mbmod.glac_bin_area_annual[:,0].sum()
                                    mb_mwea_diag = ((diag.volume_m3.values[-1] - diag.volume_m3.values[0]) 
                                                    / area_initial / nyears * pygem_prms.density_ice / pygem_prms.density_water)
                                    mb_mwea_mbmod = mbmod.glac_wide_massbaltotal.sum() / area_initial / nyears
                                   
                                    if debug:
                                        vol_change_diag = diag.volume_m3.values[-1] - diag.volume_m3.values[0]
                                        print('  vol init  [Gt]:', np.round(diag.volume_m3.values[0] * 0.9 / 1e9,5))
                                        print('  vol final [Gt]:', np.round(diag.volume_m3.values[-1] * 0.9 / 1e9,5))
                                        print('  vol change[Gt]:', np.round(vol_change_diag * 0.9 / 1e9,5))
                                        print('  mb [mwea]:', np.round(mb_mwea_diag,2))
                                        print('  mb_mbmod [mwea]:', np.round(mb_mwea_mbmod,2))
                                    
                                    
                                    if np.abs(mb_mwea_diag - mb_mwea_mbmod) > 1e-6:
                                        ev_model.mb_model.ensure_mass_conservation(diag)
                                         
                                if debug:
                                    print('mass loss [Gt]:', mbmod.glac_wide_massbaltotal.sum() / 1e9)
                
                
                                # ----- RECORD PARAMETERS TO DATASET ------
                                # Run info
                                contributor = 'Rounce'
                                rgi_reg = 'rgi' + reg_str
                                agg_level = 'glaciers'
                                gcm = gcm_name
                                if 'ssp' in period:
                                    ssp = period.split('_')[0]
                                else:
                                    ssp = 'hist'
                                    
                                filename = f'{contributor}_{rgi_reg}_{agg_level}_{period}_{gcm}_{ssp}_{glacier_str}.nc'
                                
                                years = diag.time.values.astype(int)
                                volume = diag.volume_m3.values
                                area = diag.area_m2.values
    
                                
                                #%%
                                
                                ds = xr.Dataset()
        
                                ds.attrs['contributor'] = contributor
                                ds.attrs['contributor_email'] = 'drounce@cmu.edu'
                                ds.attrs['creation_date'] = date.today().strftime("%d/%m/%Y")
                                ds.attrs['rgi-region'] = rgi_reg
                                ds.attrs['aggregation-level'] = agg_level
                                ds.attrs['period'] = period
                                ds.attrs['gcm'] = gcm
                                ds.attrs['ssp'] = ssp
                                ds.attrs['information'] = 'PyGEM for mass balance and calibration with OGGM for glacier dynamics'
                                ds.attrs['stop_criterion'] = 'Simulations were stopped if volume was 0 for 20 years or 100-yr avg mb was within +/- 10 mm w.e.'
                                
                                ds['simulation_year'] = (('simulation_year'), years)
                                ds['rgi_id'] = (('rgi_id'), [rgiid])
                                
                                varname = 'volume_m3'
                                ds[varname] = (('simulation_year', 'rgi_id'), volume[:,np.newaxis])
                                ds[varname].attrs['units'] = 'm3'
                                ds[varname].attrs['long_name'] = 'Glacier volume at timestamp'
                                
                                varname = 'area_m2'
                                ds[varname] = (('simulation_year', 'rgi_id'), area[:,np.newaxis])
                                ds[varname].attrs['units'] = 'm2'
                                ds[varname].attrs['long_name'] = 'Glacier area at timestamp'
                                
                                # This is the same for all files
                                encoding = {
                                    'simulation_year': {"dtype": "int16"},
                                    'volume_m3': {"dtype": "float32"},
                                    'area_m2': {"dtype": "float32"},
                                }
                                
                                sim_fp = pygem_prms.output_filepath + 'simulations/' + reg_str + '/' + gcm_name + '/' + period + '/'
                                if not os.path.exists(sim_fp):
                                    os.makedirs(sim_fp, exist_ok=True)
                                
                                ds.to_netcdf(sim_fp + filename, encoding=encoding)
                                                    
                except:
                    # LOG FAILURE
                    fail_fp = pygem_prms.output_sim_fp + 'failed/' + reg_str + '/' + gcm_name + '/' + period + '/'
                    if not os.path.exists(fail_fp):
                        os.makedirs(fail_fp, exist_ok=True)
                    txt_fn_fail = glacier_str + "-sim_failed.txt"
                    with open(fail_fp + txt_fn_fail, "w") as text_file:
                        text_file.write(glacier_str + ' failed to complete simulation')

    # Global variables for Spyder development
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False

    if not 'pygem_modelprms' in cfg.BASENAMES:
        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif args.glacno is not None:
        glac_no = [args.glacno]
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)
        
    # Number of cores for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    gcm_name = args.gcm_list_fn
    if args.gcm_name is not None:
        gcm_list = [args.gcm_name]
    elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
        gcm_list = [pygem_prms.ref_gcm_name]
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            print('Found %d gcms to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        print('Processing:', gcm_name)
        # Pack variables for multiprocessing
        list_packed_vars = []
        for count, glac_no_lst in enumerate(glac_no_lsts):
            list_packed_vars.append([count, glac_no_lst, gcm_name])

        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

    print('Total processing time:', time.time()-time_start, 's')
    
#    print('\n\nSWITCH THE PERIOD BACK!!!\n\n')


##%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
#    # Place local variables in variable explorer
#    if args.option_parallels == 0:
#        main_vars_list = list(main_vars.keys())
#        gcm_name = main_vars['gcm_name']
#        main_glac_rgi = main_vars['main_glac_rgi']
#        if pygem_prms.hyps_data in ['Huss', 'Farinotti']:
#            main_glac_hyps = main_vars['main_glac_hyps']
#            main_glac_icethickness = main_vars['main_glac_icethickness']
#            main_glac_width = main_vars['main_glac_width']
#        dates_table = main_vars['dates_table']
#        if pygem_prms.option_synthetic_sim == 1:
#            dates_table_synthetic = main_vars['dates_table_synthetic']
#            gcm_temp_tile = main_vars['gcm_temp_tile']
#            gcm_prec_tile = main_vars['gcm_prec_tile']
#            gcm_lr_tile = main_vars['gcm_lr_tile']
#        gcm_temp = main_vars['gcm_temp']
#        gcm_tempstd = main_vars['gcm_tempstd']
#        gcm_prec = main_vars['gcm_prec']
#        gcm_elev = main_vars['gcm_elev']
#        gcm_lr = main_vars['gcm_lr']
#        gcm_temp_adj = main_vars['gcm_temp_adj']
#        gcm_prec_adj = main_vars['gcm_prec_adj']
#        gcm_elev_adj = main_vars['gcm_elev_adj']
#        gcm_temp_lrglac = main_vars['gcm_lr']
#        ds = main_vars['ds']
#        modelprms = main_vars['modelprms']
#        glacier_rgi_table = main_vars['glacier_rgi_table']
#        glacier_str = main_vars['glacier_str']
#        if pygem_prms.hyps_data in ['OGGM']:
#            gdir = main_vars['gdir']
#            fls = main_vars['fls']
#            width_initial = fls[0].widths_m
#            glacier_area_initial = width_initial * fls[0].dx
#            mbmod = main_vars['mbmod']
#            ev_model = main_vars['ev_model']
#            diag = main_vars['diag']
#            if pygem_prms.use_calibrated_modelparams:
#                modelprms_dict = main_vars['modelprms_dict']