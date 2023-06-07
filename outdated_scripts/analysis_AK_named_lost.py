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

regions = [1]

deg_groups = [1.5,2,2.7,3,4]
deg_groups_bnds = [0.25, 0.5, 0.5, 0.5, 0.5]
deg_group_colors = ['#4575b4','#74add1', '#fee090', '#fdae61', '#f46d43', '#d73027']
#deg_groups = [1.5,2,3,4]
#deg_groups_bnds = [0.25, 0.5, 0.5, 0.5]
#deg_groups_bnds = [0.25, 0.25, 0.25, 0.25]
deg_group_colors = ['#4575b4', '#fee090', '#fdae61', '#f46d43', '#d73027']
temp_colordict = {}
for ngroup, deg_group in enumerate(deg_groups):
    temp_colordict[deg_group] = deg_group_colors[ngroup]

gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
#rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
#rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

normyear = 2015

netcdf_fp = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/glacier_stats/'

temp_dev_fn = 'Global_mean_temp_deviation_2081_2100_rel_1850_1900.csv'

ds_marzeion2020_fn = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving_v3/Marzeion_etal_2020_results.nc'

analysis_fp = netcdf_fp.replace('simulations','analysis')
fig_fp = analysis_fp + '/figures/'
csv_fp = analysis_fp + '/csv/'
pickle_fp = analysis_fp + '/pickle/'
#pickle_fp = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving_v5/pickle/'



rgi_shp_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_all_simplified2_robinson.shp'
rgi_regions_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_regions_robinson-v2.shp'

    
degree_size = 0.1

vn_title_dict = {'massbal':'Mass\nBalance',                                                                      
                 'precfactor':'Precipitation\nFactor',                                                              
                 'tempchange':'Temperature\nBias',                                                               
                 'ddfsnow':'Degree-Day \nFactor of Snow'}
vn_label_dict = {'massbal':'Mass Balance\n[mwea]',                                                                      
                 'precfactor':'Precipitation Factor\n[-]',                                                              
                 'tempchange':'Temperature Bias\n[$^\circ$C]',                                                               
                 'ddfsnow':'Degree Day Factor of Snow\n[mwe d$^{-1}$ $^\circ$C$^{-1}$]',
                 'dif_masschange':'Mass Balance [mwea]\n(Observation - Model)'}
vn_label_units_dict = {'massbal':'[mwea]',                                                                      
                       'precfactor':'[-]',                                                              
                       'tempchange':'[$^\circ$C]',                                                               
                       'ddfsnow':'[mwe d$^{-1}$ $^\circ$C$^{-1}$]'}
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
rcp_namedict = {'rcp26':'RCP2.6',
                'rcp45':'RCP4.5',
                'rcp85':'RCP8.5',
                'ssp119':'SSP1-1.9',
                'ssp126':'SSP1-2.6',
                'ssp245':'SSP2-4.5',
                'ssp370':'SSP3-7.0',
                'ssp585':'SSP5-8.5'}
# Colors list
#rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
#                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
#rcp_colordict = {'ssp119':'#253494', 'ssp126':'#41b6c4', 'ssp245':'#F1EA8A', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
rcp_colordict = {'ssp119':'#081d58', 'ssp126':'#1d91c0', 'ssp245':'#7fcdbb', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
                 
rcp_styledict = {'rcp26':':', 'rcp45':':', 'rcp85':':',
                 'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp585':'-'}

#%% ===== FUNCTIONS =====
def slr_mmSLEyr(reg_vol, reg_vol_bsl):
    """ Calculate annual SLR accounting for the ice below sea level following Farinotti et al. (2019) """
    # Farinotti et al. (2019)
#    reg_vol_asl = reg_vol - reg_vol_bsl
#    return (-1*(reg_vol_asl[:,1:] - reg_vol_asl[:,0:-1]) * 
#            pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000)
    # OGGM new approach
    if len(reg_vol.shape) == 2:
        return (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                 (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
    elif len(reg_vol.shape) == 1:
        return (-1*(((reg_vol[1:] - reg_vol[0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                 (reg_vol_bsl[1:] - reg_vol_bsl[0:-1])) / pygem_prms.area_ocean * 1000))

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid defined by WGS84
    (From https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7)
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    (from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7)
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
#                    from numpy import meshgrid, deg2rad, gradient, cos
#                    from xarray import DataArray

    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = np.deg2rad(np.gradient(ylat, axis=0))
    dlon = np.deg2rad(np.gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * np.cos(np.deg2rad(ylat))

    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda
        
#%% ===== Alaskan glaciers lost =====     
startyear_plot=2000
endyear_plot=2100

# ----- PROCESS DATA -----
# Set up processing
reg_mass_all = {}
reg_mass_bsl_all = {}
reg_area_all = {} 
reg_glac_mass_all = {}
reg_glac_mass_bsl_all = {}
reg_melt_all = {}
reg_acc_all = {}
reg_refreeze_all = {}
reg_fa_all = {}
reg_glac_rgiids_all = {}
        
for reg in regions:
    reg_mass_all[reg] = {}
    reg_mass_bsl_all[reg] = {}
    reg_area_all[reg] = {}
    reg_glac_mass_all[reg] = {}
    reg_glac_mass_bsl_all[reg] = {}
    reg_melt_all[reg] = {}
    reg_acc_all[reg] = {}
    reg_refreeze_all[reg] = {}
    reg_fa_all[reg] = {}
    reg_glac_rgiids_all[reg] = {}
    
    
    main_glac_rgi_reg = None
    for rcp in rcps:
        reg_mass_all[reg][rcp] = {}
        reg_mass_bsl_all[reg][rcp] = {}
        reg_area_all[reg][rcp] = {}
        reg_glac_mass_all[reg][rcp] = {}
        reg_glac_mass_bsl_all[reg][rcp] = {}
        reg_melt_all[reg][rcp] = {}
        reg_acc_all[reg][rcp] = {}
        reg_refreeze_all[reg][rcp] = {}
        reg_fa_all[reg][rcp] = {}
        reg_glac_rgiids_all[reg][rcp] = {}
            
        # ----- NETCDF FILEPATHS AND FILENAMES -----
        # Model detail string
        model_str = '_c2_ba1_50sets_2000_2100-'
        
        # Filenames
        fp_reg_mass_annual = netcdf_fp + 'mass_annual/' + str(reg).zfill(2) + '/'
        fp_reg_mass_bsl_annual = netcdf_fp + 'mass_bsl_annual/' + str(reg).zfill(2) + '/'
        fp_reg_area_annual = netcdf_fp + 'area_annual/' + str(reg).zfill(2) + '/'
        
        fn_reg_mass_annual = 'R' + str(reg).zfill(2) + '_glac_mass_annual' + model_str + rcp + '.nc'
        fn_reg_mass_bsl_annual = 'R' + str(reg).zfill(2) + '_glac_mass_bsl_annual' + model_str + rcp + '.nc'
        fn_reg_area_annual = 'R' + str(reg).zfill(2) + '_glac_area_annual' + model_str + rcp + '.nc'
        
        fp_reg_mbcomponents = netcdf_fp
        if 'ssp' in rcp:
            fn_reg_mbcomponents = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-ssps.nc'
        elif 'rcp' in rcp:
            fn_reg_mbcomponents = 'Global_reg_allvns_c2_ba1_50sets_2000_2100-rcps.nc'
        
        # Mass
        ds_mass = xr.open_dataset(fp_reg_mass_annual + fn_reg_mass_annual)
        ds_mass_bsl = xr.open_dataset(fp_reg_mass_bsl_annual + fn_reg_mass_bsl_annual)
        
        ds_area = xr.open_dataset(fp_reg_area_annual + fn_reg_area_annual)
        glac_nos_reg = [x.split('-')[1] for x in ds_mass.RGIId.values]
        if main_glac_rgi_reg is None:
            main_glac_rgi_reg = modelsetup.selectglaciersrgitable(glac_no=glac_nos_reg)
            
        gcm_order = []
        gcm_names_raw = []
        for dict_key in ds_mass.Climate_Model.attrs:
            if dict_key not in ['long_name', 'comment']:
                gcm_order.append(int(dict_key)-1)
                gcm_names_raw.append(ds_mass.Climate_Model.attrs[dict_key])
        
        # Sort list to ensure proper indexing
        gcm_names = [x for _,x in sorted(zip(gcm_order, gcm_names_raw))]
        
        if 'rcp' in rcp:
            gcm_names_2proc = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names_2proc = gcm_names_ssp119
            else:
                gcm_names_2proc = gcm_names_ssps
        
        for ngcm, gcm_name in enumerate(gcm_names):
            if gcm_name in gcm_names_2proc:
                print(reg, rcp, gcm_name)
                
                # Mass
                reg_glac_mass_annual_gcm = ds_mass.glac_mass_annual[ngcm,:,:].values
                reg_mass_annual = reg_glac_mass_annual_gcm.sum(0)
                
                # Mass below sea level
                # - replace bsl of non-tidewater glaciers with zeros for correction
                reg_glac_mass_bsl_annual_gcm = ds_mass_bsl.glac_mass_bsl_annual[ngcm,:,:].values
                reg_glac_mass_bsl_annual_gcm_tidewateronly = reg_glac_mass_bsl_annual_gcm.copy()
                tw_idx = np.where((main_glac_rgi_reg.TermType.values == 1) | (main_glac_rgi_reg.TermType.values == 5))[0]
                if len(tw_idx) == 0:
                    tw_idx = []
                nontw_idx = [x for x in main_glac_rgi_reg.index.values if x not in tw_idx]
                reg_glac_mass_bsl_annual_gcm_tidewateronly[nontw_idx,:] = 0
                reg_mass_bsl_annual = reg_glac_mass_bsl_annual_gcm_tidewateronly.sum(0)
                
                # Area
                reg_glac_area_annual_gcm = ds_area.glac_area_annual[ngcm,:,:].values
                reg_area_annual = reg_glac_area_annual_gcm.sum(0)
                
                # Mass balance components
                ds_mbcomp = xr.open_dataset(fp_reg_mbcomponents + fn_reg_mbcomponents)
                
                # scenario index
#                    scenario_order = []
#                    scenario_names_raw = []
#                    for dict_key in ds_mbcomp.Scenario.values:
#                        print(dict_key)
#                        if dict_key not in ['long_name', 'comment']:
#                            scenario_order.append(int(dict_key)-1)
#                            scenario_names_raw.append(ds_mbcomp.Scenario.attrs[dict_key])
#                    scenario_names = [x for _,x in sorted(zip(scenario_order, scenario_names_raw))]
#                    scenario_idx = scenario_names.index(rcp_namedict[rcp])
                
                scenario_names = list(ds_mbcomp.Scenario.values)
                scenario_idx = scenario_names.index(rcp_namedict[rcp])
                
                # region index
                reg_idx = reg - 1
                
                # Convert mass (kg) to volume (m3 water)
                reg_acc_monthly = ds_mbcomp.reg_acc_monthly[reg_idx, scenario_idx, ngcm, :].values / pygem_prms.density_water
                reg_refreeze_monthly = ds_mbcomp.reg_refreeze_monthly[reg_idx, scenario_idx, ngcm, :].values  / pygem_prms.density_water
                reg_melt_monthly = ds_mbcomp.reg_melt_monthly[reg_idx, scenario_idx, ngcm, :].values  / pygem_prms.density_water
                reg_frontalablation_monthly = ds_mbcomp.reg_frontalablation_monthly[reg_idx, scenario_idx, ngcm, :].values  / pygem_prms.density_water

                # Record data
                reg_mass_all[reg][rcp][gcm_name] = reg_mass_annual
                if reg_mass_bsl_annual is None:
                    reg_mass_bsl_all[reg][rcp][gcm_name] = np.zeros(reg_mass_annual.shape)
                    reg_glac_mass_bsl_all[reg][rcp][gcm_name] = np.zeros(reg_glac_mass_annual_gcm.shape)
                else:
                    reg_mass_bsl_all[reg][rcp][gcm_name] = reg_mass_bsl_annual
                    reg_glac_mass_bsl_all[reg][rcp][gcm_name] = reg_glac_mass_bsl_annual_gcm_tidewateronly
                reg_area_all[reg][rcp][gcm_name] = reg_area_annual  
                reg_glac_mass_all[reg][rcp][gcm_name] = reg_glac_mass_annual_gcm
                reg_melt_all[reg][rcp][gcm_name] = reg_melt_monthly
                reg_acc_all[reg][rcp][gcm_name] = reg_acc_monthly
                reg_refreeze_all[reg][rcp][gcm_name] = reg_refreeze_monthly
                reg_fa_all[reg][rcp][gcm_name] = reg_frontalablation_monthly
                reg_glac_rgiids_all[reg][rcp][gcm_name] = glac_nos_reg.copy()
                
    
#%%
# Take average for each scenario
ak_glac_mass_all_rcp_med = {}
main_glac_rgi = None
for reg in regions:
    for rcp in rcps:
        ak_glac_mass_all_rcp_array = None
        if 'rcp' in rcp:
            gcm_names_2proc = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names_2proc = gcm_names_ssp119
            else:
                gcm_names_2proc = gcm_names_ssps
                
        for ngcm, gcm_name in enumerate(gcm_names):
            if gcm_name in gcm_names_2proc:
                ak_glac_mass_all_rcp_single = reg_glac_mass_all[reg][rcp][gcm_name][:,:,np.newaxis]
    
                if ak_glac_mass_all_rcp_array is None:
                    ak_glac_mass_all_rcp_array = ak_glac_mass_all_rcp_single
                else:
                    ak_glac_mass_all_rcp_array = np.concatenate((ak_glac_mass_all_rcp_array,ak_glac_mass_all_rcp_single), axis=2)
        
        ak_glac_mass_all_med = np.median(ak_glac_mass_all_rcp_array, axis=2)
        ak_glac_mass_all_rcp_med[rcp] = ak_glac_mass_all_med
        
        # Determine glaciers lost
        if main_glac_rgi is None:
            rgi_cols_drop = pygem_prms.rgi_cols_drop
            rgi_cols_drop.remove('Name')
            main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              rgi_cols_drop=rgi_cols_drop)
            
        ak_glac_mass_all_med_2100 = ak_glac_mass_all_med[:,-1]
        lost_idx = np.where(ak_glac_mass_all_med_2100 == 0)[0]
        
        main_glac_rgi_lost = main_glac_rgi.loc[lost_idx,:]
        
        main_glac_rgi_lost_names = sorted([x for x in list(main_glac_rgi_lost.Name.values) if x is not np.nan])

        # Export
        with open('../ak_lost_glacier_names_' + rcp + '.txt', 'w') as f:
            for name in main_glac_rgi_lost_names:
                f.write("%s\n" % name)
        
        #%%

