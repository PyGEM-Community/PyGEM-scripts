#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:48:41 2022

@author: drounce
"""

# Built-in libraries
import collections
import glob
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
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

# ----- Processing options -----
option_process_nsidc_regional = False           # Process data to produce NSIDC datasets
option_plot_nsidc_regional = False              # Plot figures associated with regional NSIDC dataset
option_process_nsidc_glaciers = False            # Process data to produce NSIDC datasets
option_process_nsidc_metadata = False           # Add metadata to NSIDC datasets consistent with suggestions
option_process_nsidc_metadata_regional = False   # Add metdata to NSIDC regional datasets consistent with suggestions
option_process_nsidc_metadata_runoff_ultee = True # Add metadata to NSIDC runoff datasets for ultee sims
option_update_tw_glaciers_nsidc_data_perglacier = False     # Update data with new simulations for frontal ablation
option_update_tw_glaciers_nsidc_data_globalreg = False       # Update data with new simulations for frontal ablation
option_update_tw_glaciers_zipped = False

# ----- Parameters -----
#regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [17]

normyear = 2015

# GCMs and RCP scenarios
gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
#rcps = ['ssp119']
#rcps = ['rcp26', 'rcp45', 'rcp85']
#rcps = ['rcp26', 'rcp45', 'rcp85', 'ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']

rgi_shp_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_all_simplified2_robinson.shp'
rgi_regions_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_regions_robinson-v2.shp'

rgi_reg_dict = {'all':'Global',
                'all_no519':'Global, excl. GRL and ANT',
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
rcp_namedict = {'rcp26':'RCP2.6',
                'rcp45':'RCP4.5',
                'rcp85':'RCP8.5',
                'ssp119':'SSP1-1.9',
                'ssp126':'SSP1-2.6',
                'ssp245':'SSP2-4.5',
                'ssp370':'SSP3-7.0',
                'ssp585':'SSP5-8.5'}
# Colors list
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
rcp_styledict = {'rcp26':':', 'rcp45':':', 'rcp85':':',
                 'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp585':'-'}

time_start = time.time()
#%% ===== FUNCTIONS =====
def slr_mmSLEyr(reg_vol, reg_vol_bsl, option='oggm'):
    """ Calculate annual SLR accounting for the ice below sea level
    
    Options
    -------
    oggm : accounts for BSL and the differences in density (new)
    farinotti : accounts for BSL but not the differences in density (Farinotti et al. 2019)
    None : provides mass loss in units of mm SLE
    """
    # Farinotti et al. (2019)
#    reg_vol_asl = reg_vol - reg_vol_bsl
#    return (-1*(reg_vol_asl[:,1:] - reg_vol_asl[:,0:-1]) * 
#            pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000)
    if option == 'oggm':
        # OGGM new approach
        return (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                 (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
    elif option == 'farinotti':
        reg_vol_asl = reg_vol - reg_vol_bsl
        return (-1*(reg_vol_asl[:,1:] - reg_vol_asl[:,0:-1]) * 
                pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000)
    elif option == 'None':
        # No correction
        return -1*(reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000


#%% ----- PROCESS REGIONAL DATA FOR NSIDC -----
# Make the data consistent with GlacierMIP2 to support ease-of-use and adoption from community
if option_process_nsidc_regional:
    print('Processing regional datasets...')
    
    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    
    regions_calving = [1,3,4,5,7,9,17,19]
    
    years = np.arange(2000,2102)
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0, option_wateryear='calendar')
    time_values = dates_table.loc[:,'date'].tolist()
    
#    netcdf_fp_normal = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
#    netcdf_fp_wcalving = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    pickle_fp_base_normal = '/Users/drounce/Documents/HiMAT/spc_backup/analysis/pickle/'
    pickle_fp_base_wcalving = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving_v6/pickle/'

    if len(rcps) > 5:
        assert True==False, 'Process RCPs and SSPs independently. Change rcps list.'

    # ----- REGIONAL DATASETS -----
    reg_vol_all = None
    for reg in regions:
        if reg in regions_calving:
#            netcdf_fp_cmip5 = netcdf_fp_wcalving
            pickle_fp_base = pickle_fp_base_wcalving
        else:
#            netcdf_fp_cmip5 = netcdf_fp_normal
            pickle_fp_base = pickle_fp_base_normal
            
        reg_vol_rcps = None
        reg_vol_rcps_bsl = None
        reg_area_rcps = None
        reg_melt_rcps = None
        reg_acc_rcps = None
        reg_refreeze_rcps = None
        reg_frontalablation_rcps = None
        for rcp in rcps:

            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            reg_vol_rcp_gcms = None
            reg_vol_rcp_gcms_bsl = None
            reg_area_rcp_gcms = None
            reg_melt_rcp_gcms = None
            reg_acc_rcp_gcms = None
            reg_refreeze_rcp_gcms = None
            reg_frontalablation_rcp_gcms = None
            for gcm_name in gcm_names:
                print('  ', reg, rcp, gcm_name)
                # Pickle filepath
                pickle_fp = pickle_fp_base + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                
                # Pickle filenames
                pickle_vol_fn = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_vol_annual.pkl'
                pickle_vol_fn_bsl = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_vol_annual_bwl.pkl'
                pickle_area_fn = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_area_annual.pkl'
                pickle_melt_fn = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_melt_monthly.pkl'
                pickle_acc_fn = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_acc_monthly.pkl'
                pickle_refreeze_fn = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_refreeze_monthly.pkl'
                pickle_frontalablation_fn = 'R' + str(reg) + '_' + rcp + '_' + gcm_name + '_frontalablation_monthly.pkl'
                
                if not os.path.exists(pickle_fp + pickle_vol_fn):
                    reg_vol_rcpgcm = np.zeros(years.shape)
                    reg_vol_rcpgcm_bsl = np.zeros(years.shape)
                    reg_area_rcpgcm = np.zeros(years.shape)
                    reg_melt_rcpgcm = np.zeros(len(time_values))
                    reg_acc_rcpgcm = np.zeros(len(time_values))
                    reg_refreeze_rcpgcm = np.zeros(len(time_values))
                    reg_frontalablation_rcpgcm = np.zeros(len(time_values))
                    reg_vol_rcpgcm[:] = np.nan
                    reg_vol_rcpgcm_bsl[:] = np.nan
                    reg_area_rcpgcm[:] = np.nan
                    reg_melt_rcpgcm[:] = np.nan
                    reg_acc_rcpgcm[:] = np.nan
                    reg_refreeze_rcpgcm[:] = np.nan
                    reg_frontalablation_rcpgcm[:] = np.nan
                else:
                    with open(pickle_fp + pickle_vol_fn, 'rb') as f:
                        reg_vol_rcpgcm = pickle.load(f)
                        
                    if os.path.exists(pickle_fp + pickle_vol_fn_bsl):
                        with open(pickle_fp + pickle_vol_fn_bsl, 'rb') as f:
                            reg_vol_rcpgcm_bsl = pickle.load(f)
                    else:
                        reg_vol_rcpgcm_bsl = np.zeros(years.shape)
                    if reg_vol_rcpgcm_bsl is None:
                        reg_vol_rcpgcm_bsl = np.zeros(years.shape)
                    
                    with open(pickle_fp + pickle_area_fn, 'rb') as f:
                        reg_area_rcpgcm = pickle.load(f)
                    
                    with open(pickle_fp + pickle_melt_fn, 'rb') as f:
                        reg_melt_rcpgcm_monthly = pickle.load(f)
                        
                    with open(pickle_fp + pickle_acc_fn, 'rb') as f:
                        reg_acc_rcpgcm_monthly = pickle.load(f)
                    
                    with open(pickle_fp + pickle_refreeze_fn, 'rb') as f:
                        reg_refreeze_rcpgcm_monthly = pickle.load(f)
                    
                    with open(pickle_fp + pickle_frontalablation_fn, 'rb') as f:
                        reg_frontalablation_rcpgcm_monthly = pickle.load(f)
                    
#                    # Monthly to annual mass balance components
#                    reg_melt_rcpgcm = np.zeros(len(time_values))
#                    reg_melt_rcpgcm[:] = np.nan
#                    reg_melt_rcpgcm[:-1] = reg_melt_rcpgcm_monthly.reshape(int(reg_melt_rcpgcm_monthly.shape[0]/12),12).sum(1)
                    reg_melt_rcpgcm = reg_melt_rcpgcm_monthly
                    reg_acc_rcpgcm = reg_acc_rcpgcm_monthly
                    reg_refreeze_rcpgcm = reg_refreeze_rcpgcm_monthly
                    reg_frontalablation_rcpgcm = reg_frontalablation_rcpgcm_monthly
                    
                # Aggregate GCMs
                if reg_vol_rcp_gcms is None:
                    reg_vol_rcp_gcms = reg_vol_rcpgcm.reshape(1,years.shape[0])
                    reg_vol_rcp_gcms_bsl = reg_vol_rcpgcm_bsl.reshape(1,years.shape[0])
                    reg_area_rcp_gcms = reg_area_rcpgcm.reshape(1,years.shape[0])
                    reg_melt_rcp_gcms = reg_melt_rcpgcm.reshape(1,len(time_values))
                    reg_acc_rcp_gcms = reg_acc_rcpgcm.reshape(1,len(time_values))
                    reg_refreeze_rcp_gcms = reg_refreeze_rcpgcm.reshape(1,len(time_values))
                    reg_frontalablation_rcp_gcms = reg_frontalablation_rcpgcm.reshape(1,len(time_values))
                else:
                    reg_vol_rcp_gcms = np.concatenate((reg_vol_rcp_gcms, reg_vol_rcpgcm.reshape(1,years.shape[0])), axis=0)
                    reg_vol_rcp_gcms_bsl = np.concatenate((reg_vol_rcp_gcms_bsl, reg_vol_rcpgcm_bsl.reshape(1,years.shape[0])), axis=0)
                    reg_area_rcp_gcms = np.concatenate((reg_area_rcp_gcms, reg_area_rcpgcm.reshape(1,years.shape[0])), axis=0)
                    reg_melt_rcp_gcms = np.concatenate((reg_melt_rcp_gcms, reg_melt_rcpgcm.reshape(1,len(time_values))), axis=0)
                    reg_acc_rcp_gcms = np.concatenate((reg_acc_rcp_gcms, reg_acc_rcpgcm.reshape(1,len(time_values))), axis=0)
                    reg_refreeze_rcp_gcms = np.concatenate((reg_refreeze_rcp_gcms, reg_refreeze_rcpgcm.reshape(1,len(time_values))), axis=0)
                    reg_frontalablation_rcp_gcms = np.concatenate((reg_frontalablation_rcp_gcms, reg_frontalablation_rcpgcm.reshape(1,len(time_values))), axis=0)
                
            # Aggregate RCPs
            if reg_vol_rcps is None:
                reg_vol_rcps = reg_vol_rcp_gcms[np.newaxis,:,:]
                reg_vol_rcps_bsl = reg_vol_rcp_gcms_bsl[np.newaxis,:,:]
                reg_area_rcps = reg_area_rcp_gcms[np.newaxis,:,:]
                reg_melt_rcps = reg_melt_rcp_gcms[np.newaxis,:,:]
                reg_acc_rcps = reg_acc_rcp_gcms[np.newaxis,:,:]
                reg_refreeze_rcps = reg_refreeze_rcp_gcms[np.newaxis,:,:]
                reg_frontalablation_rcps = reg_frontalablation_rcp_gcms[np.newaxis,:,:]
            else:
                reg_vol_rcps = np.concatenate((reg_vol_rcps, reg_vol_rcp_gcms[np.newaxis,:,:]), axis=0)
                reg_vol_rcps_bsl = np.concatenate((reg_vol_rcps_bsl, reg_vol_rcp_gcms_bsl[np.newaxis,:,:]), axis=0)
                reg_area_rcps = np.concatenate((reg_area_rcps, reg_area_rcp_gcms[np.newaxis,:,:]), axis=0)
                reg_melt_rcps = np.concatenate((reg_melt_rcps, reg_melt_rcp_gcms[np.newaxis,:,:]), axis=0)
                reg_acc_rcps = np.concatenate((reg_acc_rcps, reg_acc_rcp_gcms[np.newaxis,:,:]), axis=0)
                reg_refreeze_rcps = np.concatenate((reg_refreeze_rcps, reg_refreeze_rcp_gcms[np.newaxis,:,:]), axis=0)
                reg_frontalablation_rcps = np.concatenate((reg_frontalablation_rcps, reg_frontalablation_rcp_gcms[np.newaxis,:,:]), axis=0)
        
        # Aggregate regions
        if reg_vol_all is None:
            reg_vol_all = reg_vol_rcps[np.newaxis,:,:,:]
            reg_vol_all_bsl = reg_vol_rcps_bsl[np.newaxis,:,:,:]
            reg_area_all = reg_area_rcps[np.newaxis,:,:,:]
            reg_melt_all = reg_melt_rcps[np.newaxis,:,:,:]
            reg_acc_all = reg_acc_rcps[np.newaxis,:,:,:]
            reg_refreeze_all = reg_refreeze_rcps[np.newaxis,:,:,:]
            reg_frontalablation_all = reg_frontalablation_rcps[np.newaxis,:,:,:]
        else:
            reg_vol_all = np.concatenate((reg_vol_all, reg_vol_rcps[np.newaxis,:,:,:]), axis=0)
            reg_vol_all_bsl = np.concatenate((reg_vol_all_bsl, reg_vol_rcps_bsl[np.newaxis,:,:,:]), axis=0)
            reg_area_all = np.concatenate((reg_area_all, reg_area_rcps[np.newaxis,:,:,:]), axis=0)
            reg_melt_all = np.concatenate((reg_melt_all, reg_melt_rcps[np.newaxis,:,:,:]), axis=0)
            reg_acc_all = np.concatenate((reg_acc_all, reg_acc_rcps[np.newaxis,:,:,:]), axis=0)
            reg_refreeze_all = np.concatenate((reg_refreeze_all, reg_refreeze_rcps[np.newaxis,:,:,:]), axis=0)
            reg_frontalablation_all = np.concatenate((reg_frontalablation_all, reg_frontalablation_rcps[np.newaxis,:,:,:]), axis=0)

    # Convert volume (m3 ice) to mass (kg)
    reg_mass_all = reg_vol_all * pygem_prms.density_ice
    reg_mass_all_bsl = reg_vol_all_bsl * pygem_prms.density_ice
    # Convert volume (m3 water) to mass (kg)
    reg_melt_all = reg_melt_all * pygem_prms.density_water
    reg_acc_all = reg_acc_all * pygem_prms.density_water
    reg_refreeze_all = reg_refreeze_all * pygem_prms.density_water
    reg_frontalablation_all = reg_frontalablation_all * pygem_prms.density_water    
    
    print('Check on SLE and conversions:')
    print('AK [mm SLE]:', np.round(reg_mass_all[0,0,0,0] / 1e12 * 1/361.8,1))
    print('AK [km2]:', np.round(reg_area_all[0,0,0,0] / 1e6,1))
    

    #%% ===== CREATE NETCDF FILE =====
    # Data with variable attributes
    ds = xr.Dataset(
            data_vars=dict(
                    reg_mass_annual=(["Region", "Scenario", "Climate_Model", "year"], reg_mass_all),
                    reg_mass_bsl_annual=(["Region", "Scenario", "Climate_Model", "year"], reg_mass_all_bsl),
                    reg_area_annual=(["Region", "Scenario", "Climate_Model", "year"], reg_area_all),
                    reg_melt_monthly=(["Region", "Scenario", "Climate_Model", "time"], reg_melt_all),
                    reg_acc_monthly=(["Region", "Scenario", "Climate_Model", "time"], reg_acc_all),
                    reg_refreeze_monthly=(["Region", "Scenario", "Climate_Model", "time"], reg_refreeze_all),
                    reg_frontalablation_monthly=(["Region", "Scenario", "Climate_Model", "time"], reg_frontalablation_all),
                    ),
                    coords=dict(
                            Region=regions,
                            Scenario=np.arange(1,len(rcps)+1),
                            Climate_Model=np.arange(1,len(gcm_names)+1),
                            year=years,
                            time=time_values,
                    ),
                    attrs={'source': 'PyGEMv0.1.0',
                           'institution': 'Carnegie Mellon University',
                           'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                           'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                    )
    
    # Time attributes
    ds.time.attrs['long_name'] = 'time'
    ds.time.attrs['year_type'] = 'calendar year'
    ds.time.attrs['comment'] = 'start of the month'
    
    # Year attributes
    ds.year.attrs['long_name'] = 'years'
    ds.year.attrs['year_type'] = 'calendar year'
    ds.year.attrs['range'] = '2000 - 2101'
    ds.year.attrs['comment'] = 'years referring to the start of each year'
    
    # Region attributes
    ds.Region.attrs['long_name'] = 'Randolph Glacier Inventory Order 1 Region Name'
    ds.Region.attrs['comment'] = 'RGIv6.0'
    ds.Region.attrs['1'] = 'Alaska'
    ds.Region.attrs['2'] = 'Western Canada and U.S.'
    ds.Region.attrs['3'] = 'Arctic Canada North'
    ds.Region.attrs['4'] = 'Arctic Canada South'
    ds.Region.attrs['5'] = 'Greenland Periphery'
    ds.Region.attrs['6'] = 'Iceland'
    ds.Region.attrs['7'] = 'Svalbard'
    ds.Region.attrs['8'] = 'Scandinavia'
    ds.Region.attrs['9'] = 'Russian Arctic'
    ds.Region.attrs['10'] = 'North Asia'
    ds.Region.attrs['11'] = 'Central Europe'
    ds.Region.attrs['12'] = 'Caucasus and Middle East'
    ds.Region.attrs['13'] = 'Central Asia'
    ds.Region.attrs['14'] = 'South Asia West'
    ds.Region.attrs['15'] = 'South Asia East'
    ds.Region.attrs['16'] = 'Low Latitudes'
    ds.Region.attrs['17'] = 'Southern Andes'
    ds.Region.attrs['18'] = 'New Zealand'
    ds.Region.attrs['19'] = 'Antarctic and Subantarctic'
    
    # Scenario and Climate Model attributes
    if 'rcp' in rcp:
        cmip_no = '5'
        climate_model_dict = {
                'long_name': 'General Circulation Model Name',
                'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                '1':'CanESM2',
                '2':'CCSM4',
                '3':'CNRM-CM5',
                '4':'CSIRO-Mk3-6-0', 
                '5':'GFDL-CM3',
                '6':'GFDL-ESM2M', 
                '7':'GISS-E2-R', 
                '8':'IPSL-CM5A-LR', 
                '9':'MPI-ESM-LR', 
                '10':'NorESM1-M'}
        scenario_dict = {
                'long_name': 'Representative Concentration Pathway',
                '1':'RCP2.6',
                '2':'RCP4.5',
                '3':'RCP8.5'}
        scenario_fn = 'rcps'
    elif 'ssp' in rcp:
        cmip_no = '6'
        climate_model_dict = {
                'long_name': 'General Circulation Model Name',
                'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                '1':'BCC-CSM2-MR',
                '2':'CESM2',
                '3':'CESM2-WACCM',
                '4':'EC-Earth3', 
                '5':'EC-Earth3-Veg',
                '6':'FGOALS-f3-L', 
                '7':'GFDL-ESM4', 
                '8':'INM-CM4-8', 
                '9':'INM-CM5-0', 
                '10':'MPI-ESM1-2-HR',
                '11':'MRI-ESM2-0',
                '12':'NorESM2-MM'}
        scenario_dict = {
                'long_name': 'Representative Concentration Pathway',
                'comment': 'Only a subset of climate models had SSP1-1.9',
                '1':'SSP1-1.9',
                '2':'SSP1-2.6',
                '3':'SSP2-4.5',
                '4':'SSP3-7.0',
                '5':'SSP5-8.5'}
        scenario_fn = 'ssps'
    
    for atr_name in climate_model_dict.keys():
        ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
    
    for atr_name in scenario_dict.keys():
        ds.Scenario.attrs[atr_name] = scenario_dict[atr_name]

    # Mass attributes
    ds.reg_mass_annual.attrs['long_Name'] = 'Glacier mass'
    ds.reg_mass_annual.attrs['temporal_resolution'] = 'annual'
    ds.reg_mass_annual.attrs['unit'] = 'kg'
    ds.reg_mass_annual.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year and density of ice of 900 kg/m3'
    # Mass below sea level attributes
    ds.reg_mass_bsl_annual.attrs['long_name'] = 'Glacier mass below sea level'
    ds.reg_mass_bsl_annual.attrs['temporal_resolution'] = 'annual'
    ds.reg_mass_bsl_annual.attrs['unit'] = 'kg'
    ds.reg_mass_bsl_annual.attrs['comment'] = 'mass of ice below sea level based on area and ice thickness at start of the year, density of ice of 900 kg/m3, and sea level of 0 m a.s.l.'
    # Area attributes
    ds.reg_area_annual.attrs['long_name'] = 'Glacier area'
    ds.reg_area_annual.attrs['temporal_resolution'] = 'annual'
    ds.reg_area_annual.attrs['unit'] = 'm2'
    ds.reg_area_annual.attrs['comment'] = 'area at start of the year'
    # Melt attributes
    ds.reg_melt_monthly.attrs['long_name'] = 'Glacier melt'
    ds.reg_melt_monthly.attrs['units'] = 'kg'
    ds.reg_melt_monthly.attrs['temporal_resolution'] = 'monthly'
    # Accumulation attributes
    ds.reg_acc_monthly.attrs['long_name'] = 'Glacier accumulation'
    ds.reg_acc_monthly.attrs['units'] = 'kg'
    ds.reg_acc_monthly.attrs['temporal_resolution'] = 'monthly'
    ds.reg_acc_monthly.attrs['comment'] = 'only the solid precipitation'
    # Refreeze attributes
    ds.reg_refreeze_monthly.attrs['long_name'] = 'Glacier refreeze'
    ds.reg_refreeze_monthly.attrs['units'] = 'kg'
    ds.reg_refreeze_monthly.attrs['temporal_resolution'] = 'monthly'
    # Frontal ablation attributes
    ds.reg_frontalablation_monthly.attrs['long_name'] = 'Glacier frontal ablation'
    ds.reg_frontalablation_monthly.attrs['units'] = 'kg'
    ds.reg_frontalablation_monthly.attrs['temporal_resolution'] = 'monthly'
    ds.reg_frontalablation_monthly.attrs['comment'] = (
            'mass losses from calving, subaerial frontal melting, sublimation above the ' +
            'waterline and subaqueous frontal melting below the waterline; positive values indicate mass lost like melt;' +
            ' frontal ablation calculated on annual time scale but shown as monthly to be consistent with mass balance comopnents')
    
    if not os.path.exists(nsidc_fp):
        os.makedirs(nsidc_fp, exist_ok=True)
    
    ds_fn = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-' + scenario_fn + '.nc'
    
    ds.to_netcdf(nsidc_fp + ds_fn)
            
    # Close datasets
#    ds.close()
    

#%% ---- PLOT REGIONAL DATASETS -----
if option_plot_nsidc_regional:
    print('Plot regional datasets...')

    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    
    # Add global region
    regions.append('all')
    regions_overview = regions[-1:] + regions[0:-1]
    # Remove ssp119 for stats
    rcps_all = rcps.copy()
    if 'ssp119' in rcps:
        rcps.remove('ssp119')
        
    years = np.arange(2000,2102)
    startyear = 2015
    endyear = 2100
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0, option_wateryear='calendar')
    time_values = dates_table.loc[:,'date'].tolist()
    
    # Load data
    if len(rcps) > 5:
        assert True==False, 'Process RCPs and SSPs independently. Change rcps list.'
    
    if 'ssp245' in rcps:
        scenario_fn = 'ssps'
        rcps_plot_mad = ['ssp126', 'ssp585']
    elif 'rcp45' in rcps:
        scenario_fn = 'rcps'
        rcps_plot_mad = ['rcp26', 'rcp85']
        
    ds_fn = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-' + scenario_fn + '.nc'
    ds = xr.open_dataset(nsidc_fp + ds_fn)
    
    #%% ----- MULTI-GCM STATISTICS ----- 
    normyear_idx = np.where(years == normyear)[0][0]
    
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms',
                          'Marzeion_slr_mmsle_mean', 'Edwards_slr_mmsle_mean',
                          'slr_mmSLE_med', 'slr_mmSLE_95', 'slr_mmSLE_mean', 'slr_mmSLE_std', 'slr_mmSLE_mad',
                          'mb_mmSLE_med', 'mb_mmSLE_95', 'mb_mmSLE_mean', 'mb_mmSLE_std', 'mb_mmSLE_mad', 
                          'slr_correction_mmSLE',
                          'slr_2090-2100_mmSLEyr_med', 'slr_2090-2100_mmSLEyr_mean', 'slr_2090-2100_mmSLEyr_std', 'slr_2090-2100_mmSLEyr_mad', 
                          'slr_max_mmSLEyr_med', 'slr_max_mmSLEyr_mean', 'slr_max_mmSLEyr_std', 'slr_max_mmSLEyr_mad', 
                          'yr_max_slr_med', 'yr_max_slr_mean', 'yr_max_slr_std', 'yr_max_slr_mad',
                          'vol_lost_%_med', 'vol_lost_%_95', 'vol_lost_%_mean', 'vol_lost_%_std', 'vol_lost_%_mad',
                          'Marzeion_vol_lost_%_mean',
                          'area_lost_%_med', 'area_lost_%_mean', 'area_lost_%_std', 'area_lost_%_mad',
                          'mb_2090-2100_mmwea_med', 'mb_2090-2100_mmwea_mean', 'mb_2090-2100_mmwea_std', 'mb_2090-2100_mmwea_mad',
                          'mb_max_mmwea_med', 'mb_max_mmwea_mean', 'mb_max_mmwea_std', 'mb_max_mmwea_mad']
    
    
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(stats_overview_cns))), columns=stats_overview_cns)
    
    if 'ssp126' in rcps:
        stats_overview_df.loc[0,'Edwards_slr_mmsle_mean'] = 80
        stats_overview_df.loc[1,'Edwards_slr_mmsle_mean'] = 115
        stats_overview_df.loc[3,'Edwards_slr_mmsle_mean'] = 170
    
    if 'rcp26' in rcps:
        stats_overview_df.loc[0,'Marzeion_slr_mmsle_mean'] = 79
        stats_overview_df.loc[1,'Marzeion_slr_mmsle_mean'] = 119
        stats_overview_df.loc[2,'Marzeion_slr_mmsle_mean'] = 159
        
        stats_overview_df.loc[0,'Marzeion_vol_lost_%_mean'] = 18
        stats_overview_df.loc[1,'Marzeion_vol_lost_%_mean'] = 27
        stats_overview_df.loc[2,'Marzeion_vol_lost_%_mean'] = 36
    
    
    ncount = 0
    for nreg, reg in enumerate(regions_overview):
        for rcp in rcps:
            nrcp = rcps_all.index(rcp)
            
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0)
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values
            
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
    nsidc_csv_fp = nsidc_fp + 'csv/'
    if not os.path.exists(nsidc_csv_fp):
        os.makedirs(nsidc_csv_fp, exist_ok=True)
    stats_overview_df.to_csv(nsidc_csv_fp + 'stats_overview' + '-' + scenario_fn + '.csv', index=False)
    
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
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
        
        for rcp in rcps:
            
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
            
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
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
                ncols=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols = 1
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=ncols, columnspacing=0.5, labelspacing=0.25, 
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
    nsidc_fig_fp = nsidc_fp + 'figures/'
    if not os.path.exists(nsidc_fig_fp):
        os.makedirs(nsidc_fig_fp, exist_ok=True)
    fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    
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
            
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0)
            else:
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values
            
            # Median and absolute median deviation
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
                ncols=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols = 1
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=ncols, columnspacing=0.5, labelspacing=0.25, 
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
    fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
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
            
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0) 
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values
            
            # Median and absolute median deviation
            reg_mass = reg_vol * pygem_prms.density_ice

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
                ncols=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols = 1
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=ncols, columnspacing=0.5, labelspacing=0.25, 
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
    fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    mb_mwea_max_fn = 'mb_mwea_max_statistics.csv'
    mb_mwea_max_df.to_csv(nsidc_csv_fp + mb_mwea_max_fn, index=False)
    
    mb_gta_max_fn = 'mb_gta_max_statistics.csv'
    mb_gta_max_df.to_csv(nsidc_csv_fp + mb_gta_max_fn, index=False)
    
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
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0) 
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values

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
                ncols=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols = 1
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=ncols, columnspacing=0.5, labelspacing=0.25, 
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
    fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    mb_slr_max_fn = 'mb_mmSLEyr_max_statistics.csv'
    mb_mmslea_max_df.to_csv(nsidc_csv_fp + mb_slr_max_fn, index=False)
    
    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE -----
    slr_cum_cns = ['Region', 'Scenario', 'med_mmSLE', 'mean_mmSLE', 'std_mmSLE', 'mad_mmSLE']
    slr_cum_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(slr_cum_cns))), columns=slr_cum_cns)
    ncount = 0
    
    all_slr_cum_df = pd.DataFrame(np.zeros((len(rcps),len(years)-1)), columns=years[:-1])
    
    for nreg, reg in enumerate(regions):
        for rcp_df_idx, rcp in enumerate(rcps):
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0) 
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values
            
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
                all_slr_cum_df.iloc[rcp_df_idx,:] = reg_slr_cum_med
            
            ncount += 1
    slr_cum_df.to_csv(nsidc_csv_fp + 'SLR_cum_2100_rel2015.csv', index=False)
    
    all_slr_cum_df.index = rcps
    all_slr_cum_df.to_csv(nsidc_csv_fp + 'all_SLR_cum_mmSLE_2100_rel2015_timeseries.csv')
    #%%
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
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0) 
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values

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
                ncols=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols = 1
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=ncols, columnspacing=0.5, labelspacing=0.25, 
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
    fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    # Export table
    all_idx = list(slrcum_multigcm_df['Region']).index('all')
    for rcp in rcps:
        slrcum_multigcm_df['slr_mmSLE_' + rcp + '_%'] = (
                slrcum_multigcm_df['slr_mmSLE_' + rcp] / slrcum_multigcm_df.loc[all_idx,'slr_mmSLE_' + rcp] * 100)
        slrcum_multigcm_df['slr_mmSLE_std_' + rcp + '_%'] =  (
                slrcum_multigcm_df['slr_mmSLE_std_' + rcp] / slrcum_multigcm_df.loc[all_idx,'slr_mmSLE_std_' + rcp] * 100)
    slr_cum_multigcm_fn = 'SLR_cum_multigcm_statistics.csv'
    slrcum_multigcm_df.to_csv(nsidc_csv_fp + slr_cum_multigcm_fn, index=False)
    
    
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
            nrcp = rcps_all.index(rcp)
            
            # Median and absolute median deviation
            if reg in ['all']:
                reg_vol = np.sum(ds.reg_mass_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp,:,:].values, axis=0) / pygem_prms.density_ice
                reg_area = np.sum(ds.reg_area_annual[:,nrcp,:,:].values, axis=0) 
            else:
                reg_vol = ds.reg_mass_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp,:,:].values / pygem_prms.density_ice
                reg_area = ds.reg_area_annual[reg-1,nrcp,:,:].values

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
        
#        if nax == 1:
#            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
#                leg_cols = 2
#            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
#                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
#                leg_cols = 2
#            elif 'rcp26' in rcps and len(rcps) == 3:
#                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
#                leg_cols = 1
#            elif 'ssp126' in rcps and len(rcps) == 4:
#                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#                leg_cols = 1
#            elif 'ssp126' in rcps and len(rcps) == 5:
#                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#                leg_cols = 1
#            ax.legend(loc=(-1.34,0.2), labels=labels, fontsize=10, ncol=leg_cols, columnspacing=0.5, labelspacing=0.25, 
#                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
#                      )
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncols = 1
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=ncols, columnspacing=0.5, labelspacing=0.25, 
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
    fig.savefig(nsidc_fig_fp+ fig_fn, bbox_inches='tight', dpi=300)
    
    
    #%% ----- FIGURE: GLOBAL COMBINED -----
    if 'ssp126' in rcps:
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
        
        pie_scenarios = rcps
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
                    nrcp_ds = rcps_all.index(rcp)
                    if reg in ['all']:
                        reg_vol = np.sum(ds.reg_mass_annual[:,nrcp_ds,:,:].values, axis=0) / pygem_prms.density_ice
                        reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp_ds,:,:].values, axis=0) / pygem_prms.density_ice
                        reg_area = np.sum(ds.reg_area_annual[:,nrcp_ds,:,:].values, axis=0) 
                    else:
                        reg_vol = ds.reg_mass_annual[reg-1,nrcp_ds,:,:].values / pygem_prms.density_ice
                        reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp_ds,:,:].values / pygem_prms.density_ice
                        reg_area = ds.reg_area_annual[reg-1,nrcp_ds,:,:].values
                    
                    # Median and absolute median deviation
                    reg_mass = reg_vol * pygem_prms.density_ice
            
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
                    nrcp_ds = rcps_all.index(rcp)
                    if reg in ['all']:
                        reg_vol = np.sum(ds.reg_mass_annual[:,nrcp_ds,:,:].values, axis=0) / pygem_prms.density_ice
                        reg_vol_bsl = np.sum(ds.reg_mass_bsl_annual[:,nrcp_ds,:,:].values, axis=0) / pygem_prms.density_ice
                        reg_area = np.sum(ds.reg_area_annual[:,nrcp_ds,:,:].values, axis=0) 
                    else:
                        reg_vol = ds.reg_mass_annual[reg-1,nrcp_ds,:,:].values / pygem_prms.density_ice
                        reg_vol_bsl = ds.reg_mass_bsl_annual[reg-1,nrcp_ds,:,:].values / pygem_prms.density_ice
                        reg_area = ds.reg_area_annual[reg-1,nrcp_ds,:,:].values
                    
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
            fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
    
    #%% ----- REGIONAL MASS BALANCE COMPONENTS -----
    for nrcp, rcp in enumerate(rcps):
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
            # Aggregate regional to global
            if nax == 0:
                reg_area_annual_gcms = np.sum(ds.reg_area_annual[:,nrcp,:,:-1].values, axis=0)
                reg_area_annual = np.nanmedian(reg_area_annual_gcms, axis=0)
                
                reg_melt_monthly = ds.reg_melt_monthly[:,nrcp,:,:].values
                reg_melt_annual_gcms = np.sum(reg_melt_monthly.reshape(reg_melt_monthly.shape[0],reg_melt_monthly.shape[1],int(len(time_values)/12),12).sum(3), axis=0)
                reg_melt_annual = np.nanmedian(reg_melt_annual_gcms, axis=0)
                
                reg_acc_monthly = ds.reg_acc_monthly[:,nrcp,:,:].values
                reg_acc_annual_gcms = np.sum(reg_acc_monthly.reshape(reg_acc_monthly.shape[0],reg_acc_monthly.shape[1],int(len(time_values)/12),12).sum(3), axis=0)
                reg_acc_annual = np.nanmedian(reg_acc_annual_gcms, axis=0)
                
                reg_refreeze_monthly = ds.reg_refreeze_monthly[:,nrcp,:,:].values
                reg_refreeze_annual_gcms = np.sum(reg_refreeze_monthly.reshape(reg_refreeze_monthly.shape[0],reg_refreeze_monthly.shape[1],int(len(time_values)/12),12).sum(3), axis=0)
                reg_refreeze_annual = np.nanmedian(reg_refreeze_annual_gcms, axis=0)
                
                reg_frontalablation_monthly = ds.reg_frontalablation_monthly[:,nrcp,:,:].values
                reg_frontalablation_annual_gcms = np.sum(reg_frontalablation_monthly.reshape(reg_frontalablation_monthly.shape[0],reg_frontalablation_monthly.shape[1],int(len(time_values)/12),12).sum(3), axis=0)
                reg_frontalablation_annual = np.nanmedian(reg_frontalablation_annual_gcms, axis=0)

            # Select regional data
            else:
                reg_area_annual = np.nanmedian(ds.reg_area_annual[reg-1,nrcp,:,:].values, axis=0)
                
                reg_melt_monthly = ds.reg_melt_monthly[reg-1,nrcp,:,:].values
                reg_melt_annual_gcms = reg_melt_monthly.reshape(reg_melt_monthly.shape[0],int(len(time_values)/12),12).sum(2)
                reg_melt_annual = np.nanmedian(reg_melt_annual_gcms, axis=0)
                
                reg_acc_monthly = ds.reg_acc_monthly[reg-1,nrcp,:,:].values
                reg_acc_annual_gcms = reg_acc_monthly.reshape(reg_acc_monthly.shape[0],int(len(time_values)/12),12).sum(2)
                reg_acc_annual = np.nanmedian(reg_acc_annual_gcms, axis=0)
                
                reg_refreeze_monthly = ds.reg_refreeze_monthly[reg-1,nrcp,:,:].values
                reg_refreeze_annual_gcms = reg_refreeze_monthly.reshape(reg_refreeze_monthly.shape[0],int(len(time_values)/12),12).sum(2)
                reg_refreeze_annual = np.nanmedian(reg_refreeze_annual_gcms, axis=0)
                
                reg_frontalablation_monthly = ds.reg_frontalablation_monthly[reg-1,nrcp,:,:].values
                reg_frontalablation_annual_gcms = reg_frontalablation_monthly.reshape(reg_frontalablation_monthly.shape[0],int(len(time_values)/12),12).sum(2)
                reg_frontalablation_annual = np.nanmedian(reg_frontalablation_annual_gcms, axis=0)

            # Periods
            if reg_acc_annual.shape[0] == 101:
                period_yrs = 20
                periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
                reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
                
                # Convert kg to mwea
                reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
                reg_acc_periods_mwea = reg_acc_periods / pygem_prms.density_water / reg_area_periods / period_yrs
                reg_refreeze_periods_mwea = reg_refreeze_periods / pygem_prms.density_water / reg_area_periods / period_yrs
                reg_melt_periods_mwea = reg_melt_periods / pygem_prms.density_water / reg_area_periods / period_yrs
                reg_frontalablation_periods_mwea = reg_frontalablation_periods / pygem_prms.density_water / reg_area_periods / period_yrs
                reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / pygem_prms.density_water / reg_area_periods / period_yrs
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
        fig_fn = 'mbcomponents_allregions_2000-2100_multigcm-' + rcp + '.png'
        fig.set_size_inches(8.5,11)
        fig.savefig(nsidc_fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
        
#%% ----- PROCESS DATASETS FOR INDIVIDUAL GLACIERS AND ELEVATION BINS -----
if option_process_nsidc_glaciers:
    print('processing individual glaciers')
    
    overwrite = True
    
    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    
    regions_calving = [1,3,4,5,7,9,17,19]
    
    year_values = np.arange(2000,2102)
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0, option_wateryear='calendar')
    time_values = dates_table.loc[:,'date'].tolist()
    
    zipped_fp_cmip5 = '/Volumes/LaCie/globalsims_backup/simulations-cmip5/_zipped/'
    zipped_fp_cmip6 = '/Volumes/LaCie/globalsims_backup/simulations-cmip6/_zipped/'
    unzipped_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_2proc/unzipped/'
    if not os.path.exists(unzipped_fp):
        os.makedirs(unzipped_fp)

    # ----- UNZIP DATASETS AND DETERMINE GLACIER LIST -----
    for reg in regions:
            
        glacno_list_pickle = 'R' + str(reg).zfill(2) + '-glacno_list.pkl'
        nsidc_pkl = nsidc_fp + 'pickle/'
        if not os.path.exists(nsidc_pkl):
            os.makedirs(nsidc_pkl)
        
        if os.path.exists(nsidc_pkl + glacno_list_pickle):
            with open(nsidc_pkl + glacno_list_pickle, 'rb') as f:
                glacno_list = pickle.load(f)
        else:
#        for batman in [0]:
            glacno_list = []
            for rcp in rcps:
                
                if 'rcp' in rcp:
                    zipped_fp = zipped_fp_cmip5
                elif 'ssp' in rcp:
                    zipped_fp = zipped_fp_cmip6
                
                if 'rcp' in rcp:
                    gcm_names = gcm_names_rcps
                elif 'ssp' in rcp:
                    if rcp in ['ssp119']:
                        gcm_names = gcm_names_ssp119
                    else:
                        gcm_names = gcm_names_ssps
                    
                for gcm_name in gcm_names:
                    # Zipped filepath
                    zipped_fn = zipped_fp + str(reg).zfill(2) + '/stats/' + gcm_name + '_' + rcp + '_stats.zip'
    
                    unzip_stats_fp = unzipped_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                    if not os.path.exists(unzip_stats_fp):
                        os.makedirs(unzip_stats_fp)

                    # Only include file runs (i.e., skip SSP1-1.9 missing gcms)
                    if os.path.exists(zipped_fn):
#                    for batman in [0]:
                        # Only unzip file once
                        if len(os.listdir(unzip_stats_fp)) <= 1:
                            with zipfile.ZipFile(zipped_fn, 'r') as zip_ref:
                                zip_ref.extractall(unzip_stats_fp)
                            
                        # Determine glacier list
                        glacno_list_gcmrcp = []
                        for i in os.listdir(unzip_stats_fp):
                            if i.endswith('.nc'):
                                glacno_list_gcmrcp.append(i.split('_')[0])
                        glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                        
                        print(reg, gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                        
                        # Only include the glaciers that were simulated by all GCM/RCP combinations
                        if len(glacno_list) == 0:
                            glacno_list = glacno_list_gcmrcp
                        else:
                            glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                        glacno_list = sorted(glacno_list)
                        
            with open(nsidc_pkl + glacno_list_pickle, 'wb') as f:
                pickle.dump(glacno_list, f)  
#            assert 1==0, 'here'
                
        #%%
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)

        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        
        #%%
        # CREATE BATCHES
        glacno_list_batches = []
        glacno_list_batch_single = []
        batch_start = 1
        batch_end = 1000
        batch_interval = 1000
        for nglac, glacno in enumerate(glacno_list):
            
            glacno_list_batch_single.append(glacno)
            
            glacno_int = int(glacno.split('.')[1])
            if not glacno == glacno_list[-1]:
                glacno_int_next =  int(glacno_list[nglac+1].split('.')[1])
                
            if glacno_int == batch_end or glacno == glacno_list[-1] or glacno_int_next > batch_end:
                glacno_list_batches.append(glacno_list_batch_single)
                glacno_list_batch_single = []
                
                # Update batch ranges
                batch_start += batch_interval
                batch_end += batch_interval
                
        #%%
        # Loop through batches to aggregate files
        for rcp in rcps:

            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            nsidc_glac_fp = nsidc_fp + 'glacier_stats/mass_annual/' + str(reg).zfill(2) + '/'
            ds_fn = ('R' + str(reg).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc')
            
            if not os.path.exists(nsidc_glac_fp + ds_fn) or overwrite:
                # Check that file doesn't already exist
                # Loop through batches
                batch_start = 1
                batch_end = 1000
                for nbatch, glacno_list in enumerate(glacno_list_batches):
                    
    #                glacno_list = glacno_list[0:10]
                    
                    batch_start = nbatch*1000 + 1
                    batch_end = nbatch*1000 + 1000
                    
                    nsidc_glac_fp = nsidc_fp + 'glacier_stats/mass_annual/' + str(reg).zfill(2) + '/'
                    ds_fn_batch = ('R' + str(reg).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                   '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')

                    # Process only GCM/Scenario combinations that don't exist already
                    if not os.path.exists(nsidc_glac_fp + ds_fn_batch):
    
                        reg_mass_rcp_gcms = None
                        reg_mass_rcp_gcms_bsl = None
                        reg_area_rcp_gcms = None
                        reg_runoff_rcps = None
                        reg_mass_rcp_gcms_mad = None
                        reg_mass_rcp_gcms_bsl_mad = None
                        reg_area_rcp_gcms_mad = None
                        reg_runoff_rcps_mad = None
                        for gcm_name in gcm_names:
                            print('  ', reg, rcp, gcm_name, batch_start, batch_end)
                            
                            unzip_stats_fp = unzipped_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
            
                            glacno_list = sorted(glacno_list)
                    
                            reg_mass_rcp_gcm_glacs = None
                            reg_mass_rcp_gcm_glacs_bsl = None
                            reg_area_rcp_gcm_glacs = None
                            reg_runoff_rcp_glacs = None
                            reg_mass_rcp_gcm_glacs_mad = None
                            reg_mass_rcp_gcm_glacs_bsl_mad = None
                            reg_area_rcp_gcm_glacs_mad = None
                            reg_runoff_rcp_glacs_mad = None
                            
                            for glacno in glacno_list:
                                
                                try:
                                    glacno_fn = glob.glob(unzip_stats_fp + glacno+'*.nc')[0]
                                    ds_glac = xr.open_dataset(glacno_fn)
                                    glac_mass_annual = ds_glac.glac_volume_annual.values * pygem_prms.density_ice
                                    glac_mass_annual_mad = ds_glac.glac_volume_annual_mad.values * pygem_prms.density_ice
                                    glac_mass_bsl_annual = ds_glac.glac_volume_bsl_annual.values * pygem_prms.density_ice
                                    glac_mass_bsl_annual_mad = ds_glac.glac_volume_bsl_annual_mad.values * pygem_prms.density_ice
                                    glac_area_annual = ds_glac.glac_area_annual.values
                                    glac_area_annual_mad = ds_glac.glac_area_annual_mad.values
                                    glac_runoff_monthly = ds_glac.glac_runoff_monthly.values
                                    glac_runoff_monthly_mad = ds_glac.glac_runoff_monthly_mad.values
                                except:
                                    glac_mass_annual = np.zeros((1,year_values.shape[0]))
                                    glac_mass_annual_mad = np.zeros((1,year_values.shape[0]))
                                    glac_mass_bsl_annual = np.zeros((1,year_values.shape[0]))
                                    glac_mass_bsl_annual_mad = np.zeros((1,year_values.shape[0]))
                                    glac_area_annual = np.zeros((1,year_values.shape[0]))
                                    glac_area_annual_mad = np.zeros((1,year_values.shape[0]))
                                    glac_runoff_monthly = np.zeros((1,len(time_values)))
                                    glac_runoff_monthly_mad = np.zeros((1,len(time_values)))
                                
                                if reg_mass_rcp_gcm_glacs is None:
                                    reg_mass_rcp_gcm_glacs = glac_mass_annual
                                    reg_mass_rcp_gcm_glacs_mad = glac_mass_annual_mad
                                    reg_mass_rcp_gcm_glacs_bsl = glac_mass_bsl_annual
                                    reg_mass_rcp_gcm_glacs_bsl_mad = glac_mass_bsl_annual_mad
                                    reg_area_rcp_gcm_glacs = glac_area_annual
                                    reg_area_rcp_gcm_glacs_mad = glac_area_annual_mad
                                    reg_runoff_rcp_gcm_glacs = glac_runoff_monthly
                                    reg_runoff_rcp_gcm_glacs_mad = glac_runoff_monthly_mad
                                else:
                                    reg_mass_rcp_gcm_glacs = np.concatenate((reg_mass_rcp_gcm_glacs, glac_mass_annual), axis=0)
                                    reg_mass_rcp_gcm_glacs_mad = np.concatenate((reg_mass_rcp_gcm_glacs_mad, glac_mass_annual_mad), axis=0)
                                    reg_mass_rcp_gcm_glacs_bsl = np.concatenate((reg_mass_rcp_gcm_glacs_bsl, glac_mass_bsl_annual), axis=0)
                                    reg_mass_rcp_gcm_glacs_bsl_mad = np.concatenate((reg_mass_rcp_gcm_glacs_bsl_mad, glac_mass_bsl_annual_mad), axis=0)
                                    reg_area_rcp_gcm_glacs = np.concatenate((reg_area_rcp_gcm_glacs, glac_area_annual), axis=0)
                                    reg_area_rcp_gcm_glacs_mad = np.concatenate((reg_area_rcp_gcm_glacs_mad, glac_area_annual_mad), axis=0)
                                    reg_runoff_rcp_gcm_glacs = np.concatenate((reg_runoff_rcp_gcm_glacs, glac_runoff_monthly), axis=0)
                                    reg_runoff_rcp_gcm_glacs_mad = np.concatenate((reg_runoff_rcp_gcm_glacs_mad, glac_runoff_monthly_mad), axis=0)
            
                            # Aggregate GCMs
                            if reg_mass_rcp_gcms is None:
                                reg_mass_rcp_gcms = reg_mass_rcp_gcm_glacs[np.newaxis,:,:]
                                reg_mass_rcp_gcms_mad = reg_mass_rcp_gcm_glacs_mad[np.newaxis,:,:]
                                reg_mass_rcp_gcms_bsl = reg_mass_rcp_gcm_glacs_bsl[np.newaxis,:,:]
                                reg_mass_rcp_gcms_bsl_mad = reg_mass_rcp_gcm_glacs_bsl_mad[np.newaxis,:,:]
                                reg_area_rcp_gcms = reg_area_rcp_gcm_glacs[np.newaxis,:,:]
                                reg_area_rcp_gcms_mad = reg_area_rcp_gcm_glacs_mad[np.newaxis,:,:]
                                reg_runoff_rcp_gcms = reg_runoff_rcp_gcm_glacs[np.newaxis,:,:]
                                reg_runoff_rcp_gcms_mad = reg_runoff_rcp_gcm_glacs_mad[np.newaxis,:,:]
                            else:
                                reg_mass_rcp_gcms = np.concatenate((reg_mass_rcp_gcms, reg_mass_rcp_gcm_glacs[np.newaxis,:,:]), axis=0)
                                reg_mass_rcp_gcms_mad = np.concatenate((reg_mass_rcp_gcms_mad, reg_mass_rcp_gcm_glacs_mad[np.newaxis,:,:]), axis=0)
                                reg_mass_rcp_gcms_bsl = np.concatenate((reg_mass_rcp_gcms_bsl, reg_mass_rcp_gcm_glacs_bsl[np.newaxis,:,:]), axis=0)
                                reg_mass_rcp_gcms_bsl_mad = np.concatenate((reg_mass_rcp_gcms_bsl_mad, reg_mass_rcp_gcm_glacs_bsl_mad[np.newaxis,:,:]), axis=0)
                                reg_area_rcp_gcms = np.concatenate((reg_area_rcp_gcms, reg_area_rcp_gcm_glacs[np.newaxis,:,:]), axis=0)
                                reg_area_rcp_gcms_mad = np.concatenate((reg_area_rcp_gcms_mad, reg_area_rcp_gcm_glacs_mad[np.newaxis,:,:]), axis=0)
                                reg_runoff_rcp_gcms = np.concatenate((reg_runoff_rcp_gcms, reg_runoff_rcp_gcm_glacs[np.newaxis,:,:]), axis=0)
                                reg_runoff_rcp_gcms_mad = np.concatenate((reg_runoff_rcp_gcms_mad, reg_runoff_rcp_gcm_glacs_mad[np.newaxis,:,:]), axis=0)
                        
                            
                        # ===== CREATE NETCDF FILES =====
                        rgiid_list = ['RGI60-' + x for x in glacno_list]
                        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
                        cenlon_list = list(main_glac_rgi.CenLon.values)
                        cenlat_list = list(main_glac_rgi.CenLat.values)
                        
                        # ----- MASS -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_mass_annual=(["Climate_Model", "glac", "year"], reg_mass_rcp_gcms),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                year=year_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Mass attributes
                        ds.glac_mass_annual.attrs['long_Name'] = 'Glacier mass'
                        ds.glac_mass_annual.attrs['unit'] = 'kg'
                        ds.glac_mass_annual.attrs['temporal_resolution'] = 'annual'
                        ds.glac_mass_annual.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year'
                        
                        # Year attributes
                        ds.year.attrs['long_name'] = 'years'
                        ds.year.attrs['year_type'] = 'calendar year'
                        ds.year.attrs['range'] = '2000 - 2101'
                        ds.year.attrs['comment'] = 'years referring to the start of each year'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/mass_annual/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                        
                        
                        # ----- MASS MAD -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_mass_annual_mad=(["Climate_Model", "glac", "year"], reg_mass_rcp_gcms_mad),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                year=year_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Mass mad attributes
                        ds.glac_mass_annual_mad.attrs['long_Name'] = 'Glacier mass median absolute deviation'
                        ds.glac_mass_annual_mad.attrs['unit'] = 'kg'
                        ds.glac_mass_annual_mad.attrs['temporal_resolution'] = 'annual'
                        ds.glac_mass_annual_mad.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year'
                        
                        # Year attributes
                        ds.year.attrs['long_name'] = 'years'
                        ds.year.attrs['year_type'] = 'calendar year'
                        ds.year.attrs['range'] = '2000 - 2101'
                        ds.year.attrs['comment'] = 'years referring to the start of each year'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/mass_annual_mad/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_mass_annual_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_mass_annual_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                    
                    
                        # ----- MASS BELOW SEA LEVEL-----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_mass_bsl_annual=(["Climate_Model", "glac", "year"], reg_mass_rcp_gcms_bsl),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                year=year_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Mass attributes
                        ds.glac_mass_bsl_annual.attrs['long_Name'] = 'Glacier mass below sea level'
                        ds.glac_mass_bsl_annual.attrs['unit'] = 'kg'
                        ds.glac_mass_bsl_annual.attrs['temporal_resolution'] = 'annual'
                        ds.glac_mass_bsl_annual.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year'
                        
                        # Year attributes
                        ds.year.attrs['long_name'] = 'years'
                        ds.year.attrs['year_type'] = 'calendar year'
                        ds.year.attrs['range'] = '2000 - 2101'
                        ds.year.attrs['comment'] = 'years referring to the start of each year'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/mass_bsl_annual/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                        
                        
                        # ----- MASS BELOW SEA LEVEL MEDIAN ABSOLUTE DEVIATION -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_mass_bsl_annual_mad=(["Climate_Model", "glac", "year"], reg_mass_rcp_gcms_bsl_mad),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                year=year_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Mass attributes
                        ds.glac_mass_bsl_annual_mad.attrs['long_Name'] = 'Glacier mass below sea level median absolute deviation'
                        ds.glac_mass_bsl_annual_mad.attrs['unit'] = 'kg'
                        ds.glac_mass_bsl_annual_mad.attrs['temporal_resolution'] = 'annual'
                        ds.glac_mass_bsl_annual_mad.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year'
                        
                        # Year attributes
                        ds.year.attrs['long_name'] = 'years'
                        ds.year.attrs['year_type'] = 'calendar year'
                        ds.year.attrs['range'] = '2000 - 2101'
                        ds.year.attrs['comment'] = 'years referring to the start of each year'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/mass_bsl_annual_mad/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                        
                        
                        # ----- AREA -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_area_annual=(["Climate_Model", "glac", "year"], reg_area_rcp_gcms),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                year=year_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Area attributes
                        ds.glac_area_annual.attrs['long_Name'] = 'Glacier area'
                        ds.glac_area_annual.attrs['unit'] = 'm2'
                        ds.glac_area_annual.attrs['temporal_resolution'] = 'annual'
                        ds.glac_area_annual.attrs['comment'] = 'area at start of the year'
                        
                        # Year attributes
                        ds.year.attrs['long_name'] = 'years'
                        ds.year.attrs['year_type'] = 'calendar year'
                        ds.year.attrs['range'] = '2000 - 2101'
                        ds.year.attrs['comment'] = 'years referring to the start of each year'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/area_annual/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_area_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_area_annual_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                        
                        # ----- AREA MEDIAN ABSOLUTE DEVIATION -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_area_annual_mad=(["Climate_Model", "glac", "year"], reg_area_rcp_gcms_mad),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                year=year_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Area attributes
                        ds.glac_area_annual_mad.attrs['long_Name'] = 'Glacier area median absolute deviation'
                        ds.glac_area_annual_mad.attrs['unit'] = 'm2'
                        ds.glac_area_annual_mad.attrs['temporal_resolution'] = 'annual'
                        ds.glac_area_annual_mad.attrs['comment'] = 'area at start of the year'
                        
                        # Year attributes
                        ds.year.attrs['long_name'] = 'years'
                        ds.year.attrs['year_type'] = 'calendar year'
                        ds.year.attrs['range'] = '2000 - 2101'
                        ds.year.attrs['comment'] = 'years referring to the start of each year'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/area_annual_mad/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_area_annual_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_area_annual_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                        
                        
                        # ----- RUNOFF -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_runoff_monthly=(["Climate_Model", "glac", "time"], reg_runoff_rcp_gcms),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                time=time_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Mass attributes
                        ds.glac_runoff_monthly.attrs['long_Name'] = 'Glacier-wide runoff'
                        ds.glac_runoff_monthly.attrs['unit'] = 'm3'
                        ds.glac_runoff_monthly.attrs['temporal_resolution'] = 'monthly'
                        ds.glac_runoff_monthly.attrs['comment'] = 'runoff from the glacier terminus, which moves over time'
                        
                        # Time attributes
                        ds.time.attrs['long_name'] = 'time'
                        ds.time.attrs['year_type'] = 'calendar year'
                        ds.time.attrs['comment'] = 'start of the month'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/runoff_monthly/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_runoff_monthly_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_runoff_monthly_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                        
                        
                        # ----- RUNOFF MEDIAN ABSOLUTE DEVIATION -----
                        # Data with variable attributes
                        ds = xr.Dataset(
                                data_vars=dict(
                                        glac_runoff_monthly_mad=(["Climate_Model", "glac", "time"], reg_runoff_rcp_gcms_mad),
                                        RGIId=(["glac"], rgiid_list),
                                        CenLon=(["glac"], cenlon_list),
                                        CenLat=(["glac"], cenlat_list),
                                        ),
                                        coords=dict(
                                                Climate_Model=np.arange(1,len(gcm_names)+1),
                                                glac=np.arange(1,len(glacno_list)+1),
                                                time=time_values,
                                        ),
                                        attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                               'source': 'PyGEMv0.1.0',
                                               'institution': 'Carnegie Mellon University',
                                               'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                               'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                        )
                        
                        # Mass attributes
                        ds.glac_runoff_monthly_mad.attrs['long_Name'] = 'Glacier-wide runoff median absolute deviation'
                        ds.glac_runoff_monthly_mad.attrs['unit'] = 'm3'
                        ds.glac_runoff_monthly_mad.attrs['temporal_resolution'] = 'monthly'
                        ds.glac_runoff_monthly_mad.attrs['comment'] = 'runoff from the glacier terminus, which moves over time'
                        
                        # Time attributes
                        ds.time.attrs['long_name'] = 'time'
                        ds.time.attrs['year_type'] = 'calendar year'
                        ds.time.attrs['comment'] = 'start of the month'
                        
                        # Glacier number attributes
                        ds.glac.attrs['long_name'] = 'glacier index'
                        ds.glac.attrs['comment'] = 'glacier index linked to RGIId and model results'
                        
                        ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                        ds.RGIId.attrs['comment'] = 'RGIv6.0'
                        
                        ds.CenLon.attrs['long_name'] = 'center longitude'
                        ds.CenLon.attrs['units'] = 'degrees E'
                        ds.CenLon.attrs['comment'] = 'value from RGIv6.0'
                        
                        ds.CenLat.attrs['long_name'] = 'center latitude'
                        ds.CenLat.attrs['units'] = 'degrees N'
                        ds.CenLat.attrs['comment'] = 'value from RGIv6.0'
                        
                        # Scenario and Climate Model attributes
                        if 'rcp' in rcp:
                            cmip_no = '5'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'CanESM2',
                                    '2':'CCSM4',
                                    '3':'CNRM-CM5',
                                    '4':'CSIRO-Mk3-6-0', 
                                    '5':'GFDL-CM3',
                                    '6':'GFDL-ESM2M', 
                                    '7':'GISS-E2-R', 
                                    '8':'IPSL-CM5A-LR', 
                                    '9':'MPI-ESM-LR', 
                                    '10':'NorESM1-M'}
                        elif 'ssp' in rcp:
                            cmip_no = '6'
                            climate_model_dict = {
                                    'long_name': 'General Circulation Model Name',
                                    'comment': 'Models from the Coupled Model Intercomparison Project ' + cmip_no,
                                    '1':'BCC-CSM2-MR',
                                    '2':'CESM2',
                                    '3':'CESM2-WACCM',
                                    '4':'EC-Earth3', 
                                    '5':'EC-Earth3-Veg',
                                    '6':'FGOALS-f3-L', 
                                    '7':'GFDL-ESM4', 
                                    '8':'INM-CM4-8', 
                                    '9':'INM-CM5-0', 
                                    '10':'MPI-ESM1-2-HR',
                                    '11':'MRI-ESM2-0',
                                    '12':'NorESM2-MM'}
                        
                        for atr_name in climate_model_dict.keys():
                            ds.Climate_Model.attrs[atr_name] = climate_model_dict[atr_name]
                        
                        nsidc_glac_fp = nsidc_fp + 'glacier_stats/runoff_monthly_mad/' + str(reg).zfill(2) + '/'
                        if not os.path.exists(nsidc_glac_fp):
                            os.makedirs(nsidc_glac_fp, exist_ok=True)
                        
    #                    ds_fn = 'R' + str(reg).zfill(2) + '_glac_runoff_monthly_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + '.nc'
                        ds_fn = ('R' + str(reg).zfill(2) + '_glac_runoff_monthly_mad_c2_ba1_50sets_2000_2100' + '-' + rcp + 
                                 '-Batch-' + str(batch_start) + '-' + str(batch_end) + '.nc')
                        
                        ds.to_netcdf(nsidc_glac_fp + ds_fn)
                      
                #%% ----- MERGE BATCHES -----
    #            vns = ['mass_annual']
                vns = ['mass_annual', 'mass_annual_mad', 'mass_bsl_annual', 'mass_bsl_annual_mad', 
                       'area_annual', 'area_annual_mad', 'runoff_monthly', 'runoff_monthly_mad']
    
                for vn in vns:
                    vn_fp = nsidc_fp + 'glacier_stats/' + vn + '/' + str(reg).zfill(2) + '/'
            
                    fn_merge_list = []
                    fn_merge_list_start = []
                    for i in os.listdir(vn_fp):
                        if i.endswith('.nc') and rcp in i and 'Batch' in i:
                            fn_merge_list.append(i)
                            fn_merge_list_start.append(int(i.split('-')[-2]))
                
                    if len(fn_merge_list) > 0:
                        fn_merge_list = [x for _,x in sorted(zip(fn_merge_list_start,fn_merge_list))]
                    
                        ds = None
                        for fn in fn_merge_list:
                            ds_batch = xr.open_dataset(vn_fp + fn)
                            
                            if ds is None:
                                ds = ds_batch
                            else:
                                ds = xr.concat([ds, ds_batch], dim="glac")
                        ds['glac'] = np.arange(1,ds.glac.shape[0]+1)
                        
                        ds_fn = fn.split('Batch')[0][:-1] + '.nc'
                        ds.to_netcdf(vn_fp + ds_fn)
                        
                        ds_batch.close()
                        
#                        assert 1==0, 'here'
                        for fn in fn_merge_list:
                            os.remove(vn_fp + fn)
                        
                
                #%% ----- DELETE THE UNZIPPED DIRECTORIES -----
                for gcm_name in gcm_names:
                    
                    if 'rcp' in rcp:
                        zipped_fp = zipped_fp_cmip5
                    elif 'ssp' in rcp:
                        zipped_fp = zipped_fp_cmip6
                    
                    # Zipped filepath
                    zipped_fn = zipped_fp + str(reg).zfill(2) + '/stats/' + gcm_name + '_' + rcp + '_stats.zip'
                    
                    # Unzipped filepath
                    unzip_stats_fp = unzipped_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
    
                    if os.path.exists(zipped_fn):
                        if len(os.listdir(unzip_stats_fp)) > 1:
                            print('removing unzipped files in', rcp, gcm_name)
                            # Delete zipped files on local computer
                            for i in os.listdir(unzip_stats_fp):    
                                os.remove(unzip_stats_fp + i)
                            os.rmdir(unzip_stats_fp)


#%%
if option_process_nsidc_metadata_regional:
    
    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    
    for scenario in ['rcps', 'ssps']:
        
        ds_fn = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-' + scenario + '.nc'
        ds = xr.open_dataset(nsidc_fp + ds_fn)


        # -----------------------------------------------------
        # Update dataset
        ds2 = ds.copy()

        # Drop variables
        for i in list(ds.keys()):
            ds2 = ds2.drop(labels=i)
   
        vns = ['reg_mass_annual', 'reg_mass_bsl_annual', 'reg_area_annual', 'reg_melt_monthly',
               'reg_acc_monthly', 'reg_refreeze_monthly', 'reg_frontalablation_monthly']
        
        # Encoding (specify _FillValue, offsets, etc.)
        encoding = {}
        #%%
        for ds_vn in vns:
            print(ds_vn)
            
            ds2[ds_vn] = ds[ds_vn]
                
            # Correct long_Name to long_name
            if 'long_Name' in list(ds2[ds_vn].attrs.keys()):
                ds2[ds_vn].attrs['long_name'] = ds[ds_vn].attrs['long_Name']
                del ds2[ds_vn].attrs['long_Name']
                ds[ds_vn].attrs['long_name'] = ds[ds_vn].attrs['long_Name']
                
            # Correct 'unit' to 'units'
            if 'unit' in list(ds2[ds_vn].attrs.keys()):
                ds2[ds_vn].attrs['units'] = ds[ds_vn].attrs['unit']
                del ds2[ds_vn].attrs['unit']
                
            # Reorder attributes
            del ds2[ds_vn].attrs['long_name']
            ds2[ds_vn].attrs['long_name'] = ds[ds_vn].attrs['long_name']
            del ds2[ds_vn].attrs['temporal_resolution']
            ds2[ds_vn].attrs['temporal_resolution'] = ds[ds_vn].attrs['temporal_resolution']
            del ds2[ds_vn].attrs['units']
            if 'unit' in list(ds[ds_vn].attrs.keys()):
                ds2[ds_vn].attrs['units'] = ds[ds_vn].attrs['unit']
            elif 'units' in list(ds[ds_vn].attrs.keys()):
                ds2[ds_vn].attrs['units'] = ds[ds_vn].attrs['units']
            if 'comment' in list(ds[ds_vn].attrs.keys()):
                del ds2[ds_vn].attrs['comment']
                ds2[ds_vn].attrs['comment'] = ds[ds_vn].attrs['comment']

        # Change dimension names
        ds2 = ds2.rename({'Climate_Model': 'model', 
                          'Region':'region',
                          'Scenario':'scenario'})
        
        # Climate model
        if 'ssp' in scenario:
            ds_climate = xr.Dataset(data_vars=dict(Climate_Model=(["model"], gcm_names_ssps),),coords=dict(model=list(ds.Climate_Model.values),),)
            cmip_no = '6'
        elif 'rcp' in scenario:
            ds_climate = xr.Dataset(data_vars=dict(Climate_Model=(["model"], gcm_names_rcps),),coords=dict(model=list(ds.Climate_Model.values),),)
            cmip_no = '5'
        ds_climate.Climate_Model.attrs['long_name'] = 'General Circulation Model Name'
            
        ds_climate.Climate_Model.attrs['comment'] = 'Models from the Coupled Model Intercomparison Project ' + str(cmip_no)
        ds2 = xr.merge([ds2, ds_climate])
        
        # Regions
        region_names = ['Alaska', 'Western Canada and U.S.', 'Arctic Canada North', 'Arctic Canada South', 'Greenland Periphery', 
                        'Iceland', 'Svalbard', 'Scandinavia', 'Russian Arctic', 'North Asia', 'Central Europe', 'Caucasus and Middle East', 
                        'Central Asia', 'South Asia West', 'South Asia East', 'Low Latitudes', 'Southern Andes', 'New Zealand', 
                        'Antarctic and Subantarctic']
        ds_region = xr.Dataset(data_vars=dict(Region=(["region"], region_names),),coords=dict(region=list(ds.Region.values),),)
        ds_region.Region.attrs['long_name'] = 'Randolph Glacier Inventory Order 1 Region Name'
        ds_region.Region.attrs['comment'] = 'RGIv6.0'
        ds_region.Region.attrs['cf_role'] = 'timeseries_id'
        ds2 = xr.merge([ds2, ds_region])
        
        
        # Scenarios
        if 'rcp' in scenario:
            scenarios = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            scenario_name = 'Representative Concentration Pathway'
        elif 'ssp' in scenario:
            scenarios = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            scenario_name = 'Shared Socioeconomic Pathway'
        ds_scenario = xr.Dataset(data_vars=dict(Scenario=(["scenario"], scenarios),),coords=dict(region=list(ds.Scenario.values),),)
        ds_scenario.Scenario.attrs['long_name'] = scenario_name
        if 'ssp' in scenario:
            ds_scenario.Scenario.attrs['comment'] = 'Only a subset of climate models had SSP1-1.9'
        ds2 = xr.merge([ds2, ds_scenario])
        
        # Years
        # ----- NEW WORK AROUND FOR YEAR THAT MIGHT NOT WORK -----           
        ds_yr = xr.Dataset(data_vars=dict(year2=(["year"], list(ds.year.values)),), coords=dict(year=list(ds.year.values),),)
        ds2 = xr.merge([ds2, ds_yr])
        ds2 = ds2.drop(labels='year')
        ds2 = ds2.rename({'year2': 'year'})
        ds2.year.attrs['long_name'] = 'calendar year'
        ds2.year.attrs['range'] = '2000 - 2101'
        ds2.year.attrs['comment'] = 'years referring to the start of each year'
        
        ds2.attrs['source'] = ds.attrs['source']
        ds2.attrs['institution'] = ds.attrs['institution']
        ds2.attrs['history'] = ds.attrs['history']
        ds2.attrs['references'] = ds.attrs['references']
        ds2.attrs['Conventions'] = 'CF-1.9'
        ds2.attrs['featureType'] = 'timeSeries'
        
        ds2 = ds2.drop(labels='model')
        ds2 = ds2.drop(labels='scenario')
        ds2 = ds2.drop(labels='region')
        
        ds2 = ds2.set_coords(('Climate_Model', 'Region', 'Scenario', 'year', 'time'))
        
        
        ds2.time.attrs['long_name'] = 'time'
        ds2.time.attrs['comment'] = 'start of the month'
        ds2.time.attrs['year_type'] = 'calendar year'
#        ds2.time.attrs['units'] = 'days since 2000-01-01 00:00:00'
        ds2.time.encoding['calendar'] = 'proleptic_gregorian'
        ds2.time.encoding['units'] = 'days since 2000-01-01 00:00:00'


        for ds_vn in vns:

            if 'annual' in ds_vn:
                ds2[ds_vn].attrs['coordinates'] = 'Region Scenario Climate_Model year'
            elif 'monthly' in ds_vn:
                ds2[ds_vn].attrs['coordinates'] = 'Region Scenario Climate_Model time'
                
            encoding[ds_vn] = {'_FillValue': None,
                               'zlib':True,
                               'complevel':9
                               }
            
#        assert 1==0, 'here'
        
        vn_fp_export = nsidc_fp + 'glacier_stats_nsidc_compliant/'
        if not os.path.exists(vn_fp_export):
            os.makedirs(vn_fp_export)
        
        ds2.to_netcdf(vn_fp_export + ds_fn, encoding=encoding)


#%%
if option_process_nsidc_metadata:
#    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19]
    regions = [17]
#    vns = ['area_annual', 'area_annual_mad', 'mass_annual', 'mass_annual_mad',
#           'mass_bsl_annual', 'mass_bsl_annual_mad']
#    vns = ['mass_annual', 'mass_annual_mad',
#           'mass_bsl_annual', 'mass_bsl_annual_mad']
    vns = ['mass_annual_mad',
           'mass_bsl_annual', 'mass_bsl_annual_mad']
#    vns = ['runoff_monthly']
#    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    nsidc_fp = '/Volumes/LaCie/pygem_oggm_nsidc/'
    
    for reg in regions:
        
        for vn in vns:
            vn_fp = nsidc_fp + 'glacier_stats/' + vn + '/' + str(reg).zfill(2) + '/'
            
            print(reg, vn)
            
            for i in os.listdir(vn_fp):
                
                fn_prefix = 'R' + str(reg).zfill(2) + '_glac_' + vn + '_c2_ba1_'
                
                if i.endswith('.nc') and i.startswith(fn_prefix):
        
                    ds = xr.open_dataset(vn_fp + i)
                    
                    ds_vn = 'glac_' + vn
                        
                    # Update dataset
                    ds2 = ds.copy()

                    # Change RGIId attributes
                    ds2.RGIId.attrs['comment'] = 'RGIv6.0 (https://nsidc.org/data/nsidc-0770/versions/6)'
                    ds2.RGIId.attrs['cf_role'] = 'timeseries_id'
                        
                    # Year
                    if vn not in ['runoff_monthly', 'runoff_monthly_mad']:
                        ds2['year'].attrs['long_name'] = 'calendar year'
                        del ds2['year'].attrs['year_type']
                        
                    # Longitude
                    ds2 = ds2.drop(labels='CenLon')
                    ds_lon = xr.Dataset(data_vars=dict(lon=(["glac"], list(ds.CenLon.values)),),coords=dict(glac=list(ds.glac.values),),)
                    ds_lon.lon.attrs['standard_name'] = 'longitude'
                    ds_lon.lon.attrs['long_name'] = 'longitude of glacier center'
                    ds_lon.lon.attrs['units'] = 'degrees_east'
                    ds_lon.lon.attrs['_FillValue'] = np.nan
                    ds2 = xr.merge([ds2, ds_lon])
                        
                    # Latitude
                    ds2 = ds2.drop(labels='CenLat')
                    ds_lat = xr.Dataset(data_vars=dict(lat=(["glac"], list(ds.CenLat.values)),),coords=dict(glac=list(ds.glac.values),),)
                    ds_lat.lat.attrs['standard_name'] = 'latitude'
                    ds_lat.lat.attrs['long_name'] = 'latitude of glacier center'
                    ds_lat.lat.attrs['units'] = 'degrees_north'
                    ds_lat.lat.attrs['_FillValue'] = np.nan
                    ds2 = xr.merge([ds2, ds_lat])
                    
                    # Drop glac_volume_annual
                    if 'glac_volume_annual' in list(ds.keys()):
                        ds2 = ds2.drop(labels='glac_volume_annual')
                    elif 'glac_volume_annual_mad' in list(ds.keys()):
                        ds2 = ds2.drop(labels='glac_volume_annual_mad')
                    elif 'glac_volume_bsl_annual' in list(ds.keys()):
                        ds2 = ds2.drop(labels='glac_volume_bsl_annual')
                    elif 'glac_volume_bsl_annual_mad' in list(ds.keys()):
                        ds2 = ds2.drop(labels='glac_volume_bsl_annual_mad')
                    
                    
                    # Change dimension names
                    ds2 = ds2.rename({'Climate_Model': 'model', 
                                      'glac': 'glacier'})
                        
                    # Climate model
                    if 'ssp' in i:
                        ds_climate = xr.Dataset(data_vars=dict(Climate_Model=(["model"], gcm_names_ssps),),coords=dict(model=list(ds.Climate_Model.values),),)
                    elif 'rcp' in i:
                        ds_climate = xr.Dataset(data_vars=dict(Climate_Model=(["model"], gcm_names_rcps),),coords=dict(model=list(ds.Climate_Model.values),),)
                    ds_climate.Climate_Model.attrs['long_name'] = 'General Circulation Model Name'
                    if 'rcp' in i:
                        cmip_no = '5'
                    elif 'ssp' in i:
                        cmip_no = '6'
                    ds_climate.Climate_Model.attrs['comment'] = 'Models from the Coupled Model Intercomparison Project ' + str(cmip_no)
                    ds2 = xr.merge([ds2, ds_climate])
                                
                    # Add grid mapping
                    ds_gridmap = xr.Dataset(data_vars=dict(crs=([], None),))
                    ds_gridmap.crs.attrs['grid_mapping_name'] = 'latitude_longitude'
                    ds_gridmap.crs.attrs['longitude_of_prime_meridian'] = 0.0
                    ds_gridmap.crs.attrs['semi_major_axis'] = 6378137.0
                    ds_gridmap.crs.attrs['inverse_flattening'] = 298.257223563
                    ds_gridmap.crs.attrs['proj4text'] = '+proj=longlat +datum=WGS84 +no_defs'
                    ds_gridmap.crs.attrs['crs_wkt'] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
                    ds2 = xr.merge([ds2, ds_gridmap])

                    ds2[ds_vn].attrs['long_name'] = ds[ds_vn].attrs['long_Name']
                    del ds2[ds_vn].attrs['long_Name']
                    del ds2[ds_vn].attrs['temporal_resolution']
                    ds2[ds_vn].attrs['temporal_resolution'] = ds[ds_vn].attrs['temporal_resolution']
                    if 'unit' in list(ds2[ds_vn].attrs.keys()):
                        del ds2[ds_vn].attrs['unit']
                        ds2[ds_vn].attrs['units'] = ds[ds_vn].attrs['unit']
                    elif 'units' in list(ds2[ds_vn].attrs.keys()):
                        del ds2[ds_vn].attrs['units']
                        ds2[ds_vn].attrs['units'] = ds[ds_vn].attrs['units']
                    del ds2[ds_vn].attrs['comment']
                    ds2[ds_vn].attrs['comment'] = ds[ds_vn].attrs['comment']
                    ds2[ds_vn].attrs['grid_mapping'] = 'crs'
#                    ds2[ds_vn].attrs['coordinates'] = 'lat lon Climate_Model RGIId year'
                    
#                    assert 1==0, 'Coordinates should be lower-case but then does not export - wait for advice.'
                        
                    # Global attributes
                    ds2.attrs = ds.attrs
                    ds2.attrs['Conventions'] = 'CF-1.9'
                    ds2.attrs['featureType'] = 'timeSeries'
                    
                    # Fix RGIIds if don't have starting 0
                    if reg < 10 and int(ds2.RGIId.values[0][6]) > 0:
                        rgiids_fixed = ['RGI60-0' + x.split('-')[1] for x in ds.RGIId.values]
                    else:
                        rgiids_fixed = ds2.RGIId.values
                    ds2.RGIId.values = rgiids_fixed
                    
                    
                    ds2 = ds2.drop(labels='model')
                    ds2 = ds2.drop(labels='glacier')
                    
                    # ----- NEW WORK AROUND FOR YEAR THAT MIGHT NOT WORK -----
                    if vn not in ['runoff_monthly', 'runoff_monthly_mad']:
                        # Years                    
                        ds_yr = xr.Dataset(data_vars=dict(year2=(["year"], list(ds.year.values)),), coords=dict(year=list(ds.year.values),),)
    #                    ds_lo = xr.Dataset(data_vars=dict(long=(["glac"], list(ds.CenLon.values)),),coords=dict(glac=list(ds.glac.values),),)
                        ds_yr.year.attrs['long_name'] = 'calendar year'
                        ds_yr.year.attrs['range'] = '2000 - 2101'
                        ds_yr.year.attrs['comment'] = 'years referring to the start of each year'
                        ds2 = xr.merge([ds2, ds_yr])
                        ds2 = ds2.drop(labels='year')
                        ds2 = ds2.rename({'year2': 'year'})
       
                        ds2 = ds2.set_coords(('lat', 'lon', 'Climate_Model', 'RGIId', 'year'))
                        # -----
                        
                        # Year attributes
                        ds2.year.attrs['long_name'] = 'calendar year'
                        ds2.year.attrs['range'] = '2000 - 2101'
                        ds2.year.attrs['comment'] = 'years referring to the start of each year'
                        
                    else:
                        # Years                    
                        ds_time = xr.Dataset(data_vars=dict(time2=(["time"], list(ds.time.values)),), coords=dict(year=list(ds.time.values),),)
                        ds_time.time.attrs['long_name'] = 'time'
                        ds_time.time.attrs['year_type'] = 'calendar year'
                        ds_time.time.attrs['comment'] = 'start of the month'
                        ds2 = xr.merge([ds2, ds_time])
                        ds2 = ds2.drop(labels='time')
                        ds2 = ds2.rename({'time2': 'time'})
       
                        ds2 = ds2.set_coords(('lat', 'lon', 'Climate_Model', 'RGIId', 'time'))
                        # -----
                        
                        # Time attributes
                        ds2.time.attrs['long_name'] = 'time'
                        ds2.time.attrs['year_type'] = 'calendar year'
                        ds2.time.attrs['comment'] = 'start of the month'
                        
                    # Global attributes
                    ds2.attrs['Region'] = ds.attrs['Region']
                    ds2.attrs['source'] = ds.attrs['source']
                    ds2.attrs['institution'] = ds.attrs['institution']
                    ds2.attrs['history'] = ds.attrs['history']
                    ds2.attrs['references'] = ds.attrs['references']
                    ds2.attrs['Conventions'] = 'CF-1.9'
                    ds2.attrs['featureType'] = 'timeSeries'
                    
                    # Encoding (specify _FillValue, offsets, etc.)
                    encoding = {}
                    encoding[ds_vn] = {'_FillValue': None,
                                       'zlib':True,
                                       'complevel':9
                                       }

                    vn_fp_export = nsidc_fp + 'glacier_stats_nsidc_compliant/' + vn + '/' + str(reg).zfill(2) + '/'
                    if not os.path.exists(vn_fp_export):
                        os.makedirs(vn_fp_export)
                    
                    ds2.to_netcdf(vn_fp_export + i, encoding=encoding)

                        
print('Total processing time:', time.time()-time_start, 's')


#%%
if option_process_nsidc_metadata_runoff_ultee:
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#    regions = [6]
    vns = ['runoff_monthly']
    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_ultee/sims_aggregated/'
    nsidc_fp_export = '/Volumes/LaCie/pygem_oggm_nsidc/runoff_ultee/'
    
    for reg in regions:
        
        for vn in vns:
            vn_fp = nsidc_fp + vn + '/' + str(reg).zfill(2) + '/'
            
            print(reg, vn)
            
            for i in os.listdir(vn_fp):
                
                fn_prefix = 'R' + str(reg).zfill(2) + '_' + vn + '_c2_ba1_'
                
                if i.endswith('.nc') and i.startswith(fn_prefix):
        
                    ds = xr.open_dataset(vn_fp + i)
                    
                    ds_vn = 'glac_' + vn
                    
                    # Data with variable attributes
                    gcm_names = list(ds.Climate_Model.values)
                    rgiid_list = list(ds.RGIId.values)
                    ds2 = xr.Dataset(
                            data_vars=dict(
                                    glac_runoff_fixed_monthly=(["model", "glacier", "time"], ds[ds_vn].values),
                                    RGIId=(["glacier"], rgiid_list),
                                    Climate_Model=(["model"], gcm_names),
                                    lon=(["glacier"], list(ds.lon.values)),
                                    lat=(["glacier"], list(ds.lat.values)),
                                    crs=([], None),
                                    ),
                                    coords=dict(
                                            model=np.arange(1,len(gcm_names)+1),
                                            glacier=np.arange(1,len(rgiid_list)+1),
                                            time=list(ds.time.values),
                                    ),
                                    attrs={'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                           'source': 'PyGEMv0.1.0',
                                           'institution': 'Carnegie Mellon University',
                                           'history': 'Created by David Rounce (drounce@cmu.edu) on ' + pygem_prms.model_run_date,
                                           'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
                                    )
                    
                    # Runoff attributes
                    ds2.glac_runoff_fixed_monthly.attrs['long_name'] = 'Glacier-wide runoff'
                    ds2.glac_runoff_fixed_monthly.attrs['unit'] = 'm3'
                    ds2.glac_runoff_fixed_monthly.attrs['temporal_resolution'] = 'monthly'
                    ds2.glac_runoff_fixed_monthly.attrs['comment'] = 'runoff from a fixed-gauge at glacier terminus that does not move over time'
                    ds2.glac_runoff_fixed_monthly.attrs['grid_mapping'] = 'crs'
#                    ds2.glac_runoff_monthly.attrs['coordinates'] = 'lat lon Climate_Model RGIId time'
                    
                    # Time attributes
                    ds2.time.attrs['long_name'] = 'time'
                    ds2.time.attrs['range'] = '2000 - 2101'
                    ds2.time.attrs['comment'] = 'start of the month'
                    
                    # RGIId attributes
                    ds2.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
                    ds2.RGIId.attrs['comment'] = 'RGIv6.0 (https://nsidc.org/data/nsidc-0770/versions/6)'
                    ds2.RGIId.attrs['cf_role'] = 'timeseries_id'
                    
                    # Climate Model attributes
                    ds2.Climate_Model.attrs['long_name'] = 'General Circulation Model Name'
                    cmip_no = '6'
                    ds2.Climate_Model.attrs['comment'] = 'Models from the Coupled Model Intercomparison Project ' + cmip_no
                    
                    # Longitude attributes                        
                    ds2.lon.attrs['standard_name'] = 'longitude'
                    ds2.lon.attrs['long_name'] = 'longitude of glacier center'
                    ds2.lon.attrs['units'] = 'degrees_east'
                    ds2.lon.attrs['_FillValue'] = np.nan
                    
                    # Latitude attributes                        
                    ds2.lat.attrs['standard_name'] = 'latitude'
                    ds2.lat.attrs['long_name'] = 'latitude of glacier center'
                    ds2.lat.attrs['units'] = 'degrees_north'
                    ds2.lat.attrs['_FillValue'] = np.nan
                    
                    # Add grid mapping
                    ds2.crs.attrs['grid_mapping_name'] = 'latitude_longitude'
                    ds2.crs.attrs['longitude_of_prime_meridian'] = 0.0
                    ds2.crs.attrs['semi_major_axis'] = 6378137.0
                    ds2.crs.attrs['inverse_flattening'] = 298.257223563
                    ds2.crs.attrs['proj4text'] = '+proj=longlat +datum=WGS84 +no_defs'
                    ds2.crs.attrs['crs_wkt'] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
                        
                    # Global attributes
                    ds2.attrs['Region'] = ds.attrs['Region']
                    ds2.attrs['source'] = ds.attrs['source']
                    ds2.attrs['institution'] = ds.attrs['institution']
                    ds2.attrs['history'] = ds.attrs['history']
                    ds2.attrs['references'] = ds.attrs['references']
                    ds2.attrs['Conventions'] = 'CF-1.9'
                    ds2.attrs['featureType'] = 'timeSeries'
                    
                    # Drop variables that are dimensions
                    ds2 = ds2.drop(labels='model')
                    ds2 = ds2.drop(labels='glacier')
                    
                    ds2 = ds2.set_coords(('lat', 'lon', 'Climate_Model', 'RGIId', 'time'))
                    
                    # Encoding (specify _FillValue, offsets, etc.)
                    encoding = {}
                    encoding[ds_vn] = {'_FillValue': None,
                                       'zlib':True,
                                       'complevel':9
                                       }

#                    # Update dataset
#                    ds2 = ds.copy()
#
#                    # Change attributes
#                    ds2['lon'].attrs['units'] = 'degrees_east'
#                    ds2['lat'].attrs['units'] = 'degrees_north'
#                    
#                    # Drop variables that are dimensions
#                    ds2 = ds2.drop(labels='model')
#                    ds2 = ds2.drop(labels='glacier')
#                    
#                    ds2 = ds2.set_coords(('lat', 'lon', 'Climate_Model', 'RGIId', 'time'))
#
#                    # Encoding (specify _FillValue, offsets, etc.)
#                    encoding = {}
#                    encoding[ds_vn] = {'_FillValue': None,
#                                       'zlib':True,
#                                       'complevel':9
#                                       }
#                    
                    vn_fp_export = nsidc_fp_export + 'glacier_stats_nsidc_compliant/' + vn + '/' + str(reg).zfill(2) + '/'
                    if not os.path.exists(vn_fp_export):
                        os.makedirs(vn_fp_export)
                        
                    ds2.to_netcdf(vn_fp_export + i.replace(vn, 'glac_' + vn))
                        
print('Total processing time:', time.time()-time_start, 's')



#%%
if option_update_tw_glaciers_nsidc_data_perglacier:
    
    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    calving_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v6/'
    
    for reg in regions:
        
        mass_annual_fp = nsidc_fp + 'glacier_stats/mass_annual/' + str(reg).zfill(2) + '/'
        mass_annual_mad_fp = nsidc_fp + 'glacier_stats/mass_annual_mad/' + str(reg).zfill(2) + '/'
        mass_bsl_annual_fp = nsidc_fp + 'glacier_stats/mass_bsl_annual/' + str(reg).zfill(2) + '/'
        mass_bsl_annual_mad_fp = nsidc_fp + 'glacier_stats/mass_bsl_annual_mad/' + str(reg).zfill(2) + '/'
        area_annual_fp = nsidc_fp + 'glacier_stats/area_annual/' + str(reg).zfill(2) + '/'
        area_annual_mad_fp = nsidc_fp + 'glacier_stats/area_annual_mad/' + str(reg).zfill(2) + '/'
        runoff_monthly_fp = nsidc_fp + 'glacier_stats/runoff_monthly/' + str(reg).zfill(2) + '/'
        runoff_monthly_mad_fp = nsidc_fp + 'glacier_stats/runoff_monthly_mad/' + str(reg).zfill(2) + '/'
    
        for rcp in rcps:
            
            mass_annual_fn = 'R' + str(reg).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            mass_annual_mad_fn = 'R' + str(reg).zfill(2) + '_glac_mass_annual_mad_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            mass_bsl_annual_fn = 'R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            mass_bsl_annual_mad_fn = 'R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_mad_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            area_annual_fn = 'R' + str(reg).zfill(2) + '_glac_area_annual_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            area_annual_mad_fn = 'R' + str(reg).zfill(2) + '_glac_area_annual_mad_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            runoff_monthly_fn = 'R' + str(reg).zfill(2) + '_glac_runoff_monthly_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            runoff_monthly_mad_fn = 'R' + str(reg).zfill(2) + '_glac_runoff_monthly_mad_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            
            ds_mass = xr.open_dataset(mass_annual_fp + mass_annual_fn)
            ds_mass_mad = xr.open_dataset(mass_annual_mad_fp + mass_annual_mad_fn)
            ds_mass_bsl = xr.open_dataset(mass_bsl_annual_fp + mass_bsl_annual_fn)
            ds_mass_bsl_mad = xr.open_dataset(mass_bsl_annual_mad_fp + mass_bsl_annual_mad_fn)
            ds_area = xr.open_dataset(area_annual_fp + area_annual_fn)
            ds_area_mad = xr.open_dataset(area_annual_mad_fp + area_annual_mad_fn)
            ds_runoff = xr.open_dataset(runoff_monthly_fp + runoff_monthly_fn)
            ds_runoff_mad = xr.open_dataset(runoff_monthly_mad_fp + runoff_monthly_mad_fn)
            
            glacnos_all = [x.split('-')[1] for x in list(ds_mass.RGIId.values)]
            
            gcm_order = []
            gcm_names_raw = []
            for dict_key in ds_mass.Climate_Model.attrs:
                if dict_key not in ['long_name', 'comment']:
                    gcm_order.append(int(dict_key)-1)
                    gcm_names_raw.append(ds_mass.Climate_Model.attrs[dict_key])
            # Sort list to ensure proper indexing
            ds_gcm_names = [x for _,x in sorted(zip(gcm_order, gcm_names_raw))]
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                print(reg, rcp, gcm_name)
                
                netcdf_fp = calving_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/' + 'stats/'
                
                calving_fns = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        calving_fns.append(i)
                calving_fns = sorted(calving_fns)
                
                for calving_fn in calving_fns:
                    ds_stats = xr.open_dataset(netcdf_fp + calving_fn)
                    glacno = ds_stats.RGIId.values[0].split('-')[1]
                    if reg < 10:
                        glacno = glacno[1:]
                    glac_idx = glacnos_all.index(glacno)
                    
                    gcm_idx = ds_gcm_names.index(gcm_name)
                    
                    # Update data
                    glac_mass = ds_stats.glac_volume_annual.values[0,:] * pygem_prms.density_ice
                    glac_mass_mad = ds_stats.glac_volume_annual_mad.values[0,:] * pygem_prms.density_ice
                    glac_mass_bsl = ds_stats.glac_volume_bsl_annual.values[0,:] * pygem_prms.density_ice
                    glac_mass_bsl_mad = ds_stats.glac_volume_bsl_annual_mad.values[0,:] * pygem_prms.density_ice
                    glac_area = ds_stats.glac_area_annual.values[0,:]
                    glac_area_mad = ds_stats.glac_area_annual_mad.values[0,:]
                    glac_runoff = ds_stats.glac_runoff_monthly.values[0,:]
                    glac_runoff_mad = ds_stats.glac_runoff_monthly_mad.values[0,:]
                    
                    # Update main dataset
                    ds_mass['glac_mass_annual'][gcm_idx, glac_idx,:] = glac_mass
                    ds_mass_mad['glac_mass_annual_mad'][gcm_idx, glac_idx,:] = glac_mass_mad
                    ds_mass_bsl['glac_mass_bsl_annual'][gcm_idx, glac_idx,:] = glac_mass_bsl
                    ds_mass_bsl_mad['glac_mass_bsl_annual_mad'][gcm_idx, glac_idx,:] = glac_mass_bsl_mad
                    ds_area['glac_area_annual'][gcm_idx, glac_idx,:] = glac_area
                    ds_area_mad['glac_area_annual_mad'][gcm_idx, glac_idx,:] = glac_area_mad
                    ds_runoff['glac_runoff_monthly'][gcm_idx, glac_idx,:] = glac_runoff
                    ds_runoff_mad['glac_runoff_monthly_mad'][gcm_idx, glac_idx,:] = glac_runoff_mad
               
            
            # Export dataset
            new_fp = nsidc_fp.replace('nsidc','nsidc_updated')
            new_mass_annual_fp = new_fp + 'glacier_stats/mass_annual/' + str(reg).zfill(2) + '/'
            new_mass_annual_mad_fp = new_fp + 'glacier_stats/mass_annual_mad/' + str(reg).zfill(2) + '/'
            new_mass_bsl_annual_fp = new_fp + 'glacier_stats/mass_bsl_annual/' + str(reg).zfill(2) + '/'
            new_mass_bsl_annual_mad_fp = new_fp + 'glacier_stats/mass_bsl_annual_mad/' + str(reg).zfill(2) + '/'
            new_area_annual_fp = new_fp + 'glacier_stats/area_annual/' + str(reg).zfill(2) + '/'
            new_area_annual_mad_fp = new_fp + 'glacier_stats/area_annual_mad/' + str(reg).zfill(2) + '/'
            new_runoff_monthly_fp = new_fp + 'glacier_stats/runoff_monthly/' + str(reg).zfill(2) + '/'
            new_runoff_monthly_mad_fp = new_fp + 'glacier_stats/runoff_monthly_mad/' + str(reg).zfill(2) + '/'
            
            if not os.path.exists(new_mass_annual_fp):
                os.makedirs(new_mass_annual_fp)
                os.makedirs(new_mass_annual_mad_fp)
                os.makedirs(new_mass_bsl_annual_fp)
                os.makedirs(new_mass_bsl_annual_mad_fp)
                os.makedirs(new_area_annual_fp)
                os.makedirs(new_area_annual_mad_fp)
                os.makedirs(new_runoff_monthly_fp)
                os.makedirs(new_runoff_monthly_mad_fp)
                
            ds_mass.to_netcdf(new_mass_annual_fp + mass_annual_fn)
            ds_mass_mad.to_netcdf(new_mass_annual_mad_fp + mass_annual_mad_fn)
            ds_mass_bsl.to_netcdf(new_mass_bsl_annual_fp + mass_bsl_annual_fn)
            ds_mass_bsl_mad.to_netcdf(new_mass_bsl_annual_mad_fp + mass_bsl_annual_mad_fn)
            ds_area.to_netcdf(new_area_annual_fp + area_annual_fn)
            ds_area_mad.to_netcdf(new_area_annual_mad_fp + area_annual_mad_fn)
            ds_runoff.to_netcdf(new_runoff_monthly_fp + runoff_monthly_fn)
            ds_runoff_mad.to_netcdf(new_runoff_monthly_mad_fp + runoff_monthly_mad_fn)
            
     
if option_update_tw_glaciers_nsidc_data_globalreg:
    
    
    nsidc_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/'
    calving_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v6/'
    calving_fp_old = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    
    regions = [1,3,4,5,7,9]
    
#    rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps = ['rcp26', 'rcp45', 'rcp85']
    
    if 'rcp' in rcps[0]:
        global_fn = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-rcps.nc'
    elif 'ssp' in rcps[0]:
        global_fn = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-ssps.nc'
    
    ds_global = xr.open_dataset(nsidc_fp + global_fn)
    regions_all = list(ds_global.Region.values)
    
    for rcp in rcps:
        
        for reg in regions:
            
            # Regional data
            reg_idx = regions_all.index(reg)
            
            # Glacier data aggregating to replace regional data
            mass_annual_fp = nsidc_fp + 'glacier_stats/mass_annual/' + str(reg).zfill(2) + '/'
            mass_bsl_annual_fp = nsidc_fp + 'glacier_stats/mass_bsl_annual/' + str(reg).zfill(2) + '/'
            area_annual_fp = nsidc_fp + 'glacier_stats/area_annual/' + str(reg).zfill(2) + '/'
            
            mass_annual_fn = 'R' + str(reg).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            mass_bsl_annual_fn = 'R' + str(reg).zfill(2) + '_glac_mass_bsl_annual_c2_ba1_50sets_2000_2100-' + rcp + '.nc'
            area_annual_fn = 'R' + str(reg).zfill(2) + '_glac_area_annual_c2_ba1_50sets_2000_2100-' + rcp + '.nc'

            ds_mass = xr.open_dataset(mass_annual_fp + mass_annual_fn)
            ds_mass_bsl = xr.open_dataset(mass_bsl_annual_fp + mass_bsl_annual_fn)
            ds_area = xr.open_dataset(area_annual_fp + area_annual_fn)
            
            gcm_order = []
            gcm_names_raw = []
            for dict_key in ds_mass.Climate_Model.attrs:
                if dict_key not in ['long_name', 'comment']:
                    gcm_order.append(int(dict_key)-1)
                    gcm_names_raw.append(ds_mass.Climate_Model.attrs[dict_key])
            # Sort list to ensure proper indexing
            ds_gcm_names = [x for _,x in sorted(zip(gcm_order, gcm_names_raw))]
            
            # scenario index
            scenario_order = []
            scenario_names_raw = []
            for dict_key in ds_global.Scenario.attrs:
                if dict_key not in ['long_name', 'comment']:
                    scenario_order.append(int(dict_key)-1)
                    scenario_names_raw.append(ds_global.Scenario.attrs[dict_key])
            scenario_names = [x for _,x in sorted(zip(scenario_order, scenario_names_raw))]
            scenario_idx = scenario_names.index(rcp_namedict[rcp])
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                print(reg, rcp, gcm_name)
                
                # ----- REGIONAL DATA -----
                gcm_idx = ds_gcm_names.index(gcm_name)
                
                # Aggregate values
                reg_mass_annual_gcms = ds_mass.glac_mass_annual.values.sum(1)
                reg_mass_bsl_annual_gcms = ds_mass_bsl.glac_mass_bsl_annual.values.sum(1)
                reg_area_annual_gcms = ds_area.glac_area_annual.values.sum(1)

                # Update value
                ds_global['reg_mass_annual'][reg_idx, scenario_idx,:,:] = reg_mass_annual_gcms
                ds_global['reg_mass_bsl_annual'][reg_idx, scenario_idx,:,:] = reg_mass_bsl_annual_gcms
                ds_global['reg_area_annual'][reg_idx, scenario_idx,:,:] = reg_area_annual_gcms
                
                
                # ----- MASS BALANCE COMPONENTS -----
                reg_melt_monthly_rcpgcm = ds_global.reg_melt_monthly[reg_idx, scenario_idx, gcm_idx, :].values
                reg_acc_monthly_rcpgcm = ds_global.reg_acc_monthly[reg_idx, scenario_idx, gcm_idx, :].values
                reg_refreeze_monthly_rcpgcm = ds_global.reg_refreeze_monthly[reg_idx, scenario_idx, gcm_idx, :].values
                reg_frontalablation_monthly_rcpgcm = ds_global.reg_frontalablation_monthly[reg_idx, scenario_idx, gcm_idx, :].values
                
                reg_melt_init = reg_melt_monthly_rcpgcm.copy()
                reg_acc_init = reg_acc_monthly_rcpgcm.copy()
                reg_refreeze_init = reg_refreeze_monthly_rcpgcm.copy()
                reg_frontalablation_init = reg_frontalablation_monthly_rcpgcm.copy()
                
                # New simulations
                netcdf_fp = calving_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/' + 'stats/'
                calving_fns = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        calving_fns.append(i)
                calving_fns = sorted(calving_fns)
                
                # Old simulations
                netcdf_fp_old = calving_fp_old + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/' + 'stats/'
                calving_fns_old = []
                for i in os.listdir(netcdf_fp_old):
                    if i.endswith('.nc'):
                        calving_fns_old.append(i)
                calving_fns_old = sorted(calving_fns_old)
                
                calving_fns_2proc = sorted(list(set(calving_fns).intersection(calving_fns_old)))
                
                for calving_fn in calving_fns_2proc:
                    ds_stats = xr.open_dataset(netcdf_fp + calving_fn)
                    ds_stats_old = xr.open_dataset(netcdf_fp_old + calving_fn)
                    
                    # Convert glacier sims from m3 w.e. to kg to be consistent with regional data
                    glac_melt_monthly_rcpgcm = ds_stats.glac_melt_monthly.values[0,:] * pygem_prms.density_water
                    glac_acc_monthly_rcpgcm = ds_stats.glac_acc_monthly.values[0,:] * pygem_prms.density_water
                    glac_refreeze_monthly_rcpgcm = ds_stats.glac_refreeze_monthly.values[0,:] * pygem_prms.density_water
                    glac_frontalablation_monthly_rcpgcm = ds_stats.glac_frontalablation_monthly.values[0,:] * pygem_prms.density_water
                    
                    glac_melt_monthly_rcpgcm_old = ds_stats_old.glac_melt_monthly.values[0,:] * pygem_prms.density_water
                    glac_acc_monthly_rcpgcm_old = ds_stats_old.glac_acc_monthly.values[0,:] * pygem_prms.density_water
                    glac_refreeze_monthly_rcpgcm_old = ds_stats_old.glac_refreeze_monthly.values[0,:] * pygem_prms.density_water
                    glac_frontalablation_monthly_rcpgcm_old = ds_stats_old.glac_frontalablation_monthly.values[0,:] * pygem_prms.density_water
                    
                    reg_melt_monthly_rcpgcm = reg_melt_monthly_rcpgcm + glac_melt_monthly_rcpgcm - glac_melt_monthly_rcpgcm_old
                    reg_acc_monthly_rcpgcm = reg_acc_monthly_rcpgcm + glac_acc_monthly_rcpgcm - glac_acc_monthly_rcpgcm_old
                    reg_refreeze_monthly_rcpgcm = reg_refreeze_monthly_rcpgcm + glac_refreeze_monthly_rcpgcm - glac_refreeze_monthly_rcpgcm_old
                    reg_frontalablation_monthly_rcpgcm = reg_frontalablation_monthly_rcpgcm + glac_frontalablation_monthly_rcpgcm - glac_frontalablation_monthly_rcpgcm_old
                    
                reg_melt_dif = np.round((reg_melt_monthly_rcpgcm.sum() - reg_melt_init.sum()) / reg_melt_init.sum(), 2)
                reg_acc_dif = np.round((reg_acc_monthly_rcpgcm.sum() - reg_acc_init.sum()) / reg_acc_init.sum(), 2)
                reg_refreeze_dif = np.round((reg_refreeze_monthly_rcpgcm.sum() - reg_refreeze_init.sum()) / reg_refreeze_init.sum(), 2)
                reg_frontalablation_dif = np.round((reg_frontalablation_monthly_rcpgcm.sum() - reg_frontalablation_init.sum()) / reg_frontalablation_init.sum(), 2)
                print('  dif:', reg_melt_dif, reg_acc_dif, reg_refreeze_dif, reg_frontalablation_dif)

                # ----- Update data -----
                ds_global['reg_melt_monthly'][reg_idx, scenario_idx, gcm_idx, :] = reg_melt_monthly_rcpgcm
                ds_global['reg_acc_monthly'][reg_idx, scenario_idx, gcm_idx, :] = reg_acc_monthly_rcpgcm
                ds_global['reg_refreeze_monthly'][reg_idx, scenario_idx, gcm_idx, :] = reg_refreeze_monthly_rcpgcm
                ds_global['reg_frontalablation_monthly'][reg_idx, scenario_idx, gcm_idx, :] = reg_frontalablation_monthly_rcpgcm
                
    # Export dataset
    new_fp = nsidc_fp.replace('nsidc','nsidc_updated')
    if not os.path.exists(new_fp):
        os.makedirs(new_fp)
        
    ds_global.to_netcdf(new_fp + global_fn)
    
    
#%% ----- SWAP UPDATED TIDEWATER GLACIERS INTO THE ZIPPED STORAGE FILES -----
if option_update_tw_glaciers_zipped:
    
    print('processing individual glaciers')
    
    regions = [1,3,4,5,7,9] 
    
    zipped_fp_cmip5 = '/Volumes/LaCie/globalsims_backup/simulations-cmip5/_zipped/'
    zipped_fp_cmip6 = '/Volumes/LaCie/globalsims_backup/simulations-cmip6/_zipped/'
    unzipped_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_2proc/unzipped/'
    calving_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v6/'
    if not os.path.exists(unzipped_fp):
        os.makedirs(unzipped_fp)

    # ----- UNZIP DATASETS AND DETERMINE GLACIER LIST -----
    for reg in regions:
            
        for rcp in rcps:
            
            if 'rcp' in rcp:
                zipped_fp = zipped_fp_cmip5
            elif 'ssp' in rcp:
                zipped_fp = zipped_fp_cmip6
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                # Zipped filepath
                zipped_fn = zipped_fp + str(reg).zfill(2) + '/stats/' + gcm_name + '_' + rcp + '_stats.zip'
                zipped_fn_binned = zipped_fp + str(reg).zfill(2) + '/binned/' + gcm_name + '_' + rcp + '_binned.zip'

                unzip_stats_fp = unzipped_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                if not os.path.exists(unzip_stats_fp):
                    os.makedirs(unzip_stats_fp)
                    
                unzip_binned_fp = unzipped_fp + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                if not os.path.exists(unzip_binned_fp):
                    os.makedirs(unzip_binned_fp)

                # Only include file runs (i.e., skip SSP1-1.9 missing gcms)
                if os.path.exists(zipped_fn):
                    # Only unzip file once
                    if len(os.listdir(unzip_stats_fp)) <= 1:
                        with zipfile.ZipFile(zipped_fn, 'r') as zip_ref:
                            zip_ref.extractall(unzip_stats_fp)
                        
                        with zipfile.ZipFile(zipped_fn_binned, 'r') as zip_ref:
                            zip_ref.extractall(unzip_binned_fp)
                            
                    # Files to replace
                    
                        
                    assert 1==0, 'swap files'
                    assert 1==0, 'zip files back on external'
                    assert 1==0, 'overwrite external files'
#                    # Determine glacier list
#                    glacno_list_gcmrcp = []
#                    for i in os.listdir(unzip_stats_fp):
#                        if i.endswith('.nc'):
#                            glacno_list_gcmrcp.append(i.split('_')[0])
#                    glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
#                    
#                    print(reg, gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
#                    
#                    # Only include the glaciers that were simulated by all GCM/RCP combinations
#                    if len(glacno_list) == 0:
#                        glacno_list = glacno_list_gcmrcp
#                    else:
#                        glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
#                    glacno_list = sorted(glacno_list)