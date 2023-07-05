""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
import collections
import os
import pickle
import sys
import time
import zipfile
# External libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import xarray as xr
# Local libraries
try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

##from oggm import utils
#from pygem.oggm_compat import single_flowline_glacier_directory
#from pygem.shop import debris 
#from oggm import tasks


#%% ===== Input data =====
# Script options
option_check_output = False              # Scratch to check output
option_check_glaciers = False            # Check that all batches have been completed for per_glacier
option_standardize_reg = False           # Standardize regional output by making sure only includes glaciers with all sims completed

option_aggregate_files = True          # Aggregate files into format for GlacierMIP3
option_qc_area_standardize_reg = False   # Quality control by area and standardize
option_qc_growing_glaciers = False       # Quality control against growing glaciers (they grow after they have already reached equilibrium)
option_plot_output = False              # Plot regional datasets
scratch = False

#regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [6,7,9,12]

# 2 - ukesm1-0-ll 1951-1970 (0.97 area) 170001-19000
# 8 - ipsl-cm6a-lr 1851-1870 (0.32) 1001-4000
# 8 - 1901-1920 (0.32 each) --> complete
# 11 - ipsl 1901-1920 (0.23) --> complete

# GCMs and RCP scenarios
gcm_names = ['gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0', 'ukesm1-0-ll']
scenarios = ['1851-1870', '1901-1920', '1951-1970', '1995-2014',
             'ssp126_2021-2040', 'ssp126_2041-2060', 'ssp126_2061-2080', 'ssp126_2081-2100',
             'ssp370_2021-2040', 'ssp370_2041-2060', 'ssp370_2061-2080', 'ssp370_2081-2100',
             'ssp585_2021-2040', 'ssp585_2041-2060', 'ssp585_2061-2080', 'ssp585_2081-2100']
#gcm_names = ['ipsl-cm6a-lr']
#scenarios = ['1901-1920']

netcdf_fp_cmip5 = '/Users/drounce/Documents/glaciermip3/spc_backup/'
#netcdf_fp_cmip5 = pygem_prms.main_directory + '/../Output/simulations/'


fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
pickle_fp = fig_fp + '../pickle/'

rgi_reg_dict = {'all':'Global',
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


time_start = time.time()

if option_aggregate_files:
    for region in regions:
        for gcm_name in gcm_names:
            for scenario in scenarios:
                
                print(region, gcm_name, scenario)
                
                netcdf_fp_sims = netcdf_fp_cmip5 + str(region).zfill(2) + '/' + gcm_name + '/' + scenario + '/'
                
                netcdf_fns = []
                for i in os.listdir(netcdf_fp_sims):
                    if i.endswith('.nc'):
                        netcdf_fns.append(i)
                netcdf_fns = sorted(netcdf_fns)
                
                print('# of files:', len(netcdf_fns))
                
                #%%
                reg_vol = None
                batch_start = 1
                batch_end = 1000
                batch_interval = 1000
                ds_all = None
                
                # Concatenate each individual dataset to produce the glacier-specific file
                for n_fn, netcdf_fn in enumerate(netcdf_fns):
                    
                    glacno = int(netcdf_fn.split('.')[-2])
#                    print('glacno:', glacno)
                        
                    # ----- Process the data ----
                    ds = xr.open_dataset(netcdf_fp_sims + netcdf_fn) 

                    volume = ds.volume_m3.values[:,0]
                    area = ds.area_m2.values[:,0]
                    vol_idx = np.where(~np.isnan(volume))[0]
                    
                    # Only apply if there are nan values
                    if any(np.isnan(volume)):
                        # If nan values, check that there are more than 20 real values
                        if len(vol_idx) > 20:
                            vol_count = 20
                        else:
                            vol_count = len(vol_idx)
                        
                        # Equilibrium volume & area
                        vol_idx_equi = vol_idx[-vol_count:]
                        volume_equi = volume[vol_idx_equi].mean()
                        area_equi = area[vol_idx_equi].mean()
                        
                        # Replace nan with remaining
                        volume[vol_idx[-1]+1:] = volume_equi
                        area[vol_idx[-1]+1:] = area_equi
                    
                        ds['volume_m3'].values = volume[:,np.newaxis]
                        ds['area_m2'].values = area[:,np.newaxis]
                    
                    if ds_all is None:
                        ds_all = ds.copy()
                    else:
                        ds_all = xr.concat([ds_all, ds], dim='rgi_id')
                        
#                    # Regional data
#                    #  - calc separately since individual glaciers will be chunked
#                    if reg_vol is None:
#                        reg_vol = volume
#                        reg_area = area
#                    else:
#                        reg_vol += volume
#                        reg_area += area
                        
                    # ----- EXPORT GLACIER BATCHES -----
                    if n_fn < len(netcdf_fns)-1:
                        glacno_next =  int(netcdf_fns[n_fn+1].split('.')[-2])
                    
                    if glacno == batch_end or n_fn == len(netcdf_fns)-1 or glacno_next > batch_end:
                        print('  exporting ' + str(batch_start) + ' to ' + str(batch_end))
                        
                        # export file
                        glacno = ds.rgi_id.values[0].split('-')[1]
                        if glacno.startswith('0'):
                            glacno = glacno[1:]
                        ds_all_glacier_fn = netcdf_fn.replace('_' + glacno, '_Batch-' + str(batch_start) + '-' + str(batch_end))
                        print('    ', ds_all_glacier_fn)
                        ds_all_fp = netcdf_fp_cmip5 + 'Final/per_glacier/' + str(region).zfill(2) + '/'
                        if not os.path.exists(ds_all_fp):
                            os.makedirs(ds_all_fp)
                        ds_all.to_netcdf(ds_all_fp + ds_all_glacier_fn)
                        
                        # Update batch ranges
                        batch_start += batch_interval
                        batch_end += batch_interval
                        ds_all = None
                        
#                    # ----- EXPORT REGIONAL DATASET -----
#                    if n_fn == len(netcdf_fns)-1:
#                        # ----- EXPORT REGIONAL DATASET -----
#                        ds_all_reg = xr.Dataset()
#    
#                        for attr in ds.attrs.keys():
#                            ds_all_reg.attrs[attr] = ds.attrs[attr]
#                           
#                        ds_all_reg['simulation_year'] = (('simulation_year'), ds.simulation_year.values)
#                        
#                        varname = 'volume_m3'
#                        ds_all_reg[varname] = (('simulation_year',), reg_vol)
#                        ds_all_reg[varname].attrs['units'] = 'm3'
#                        ds_all_reg[varname].attrs['long_name'] = 'Glacier volume at timestamp'
#                        
#                        varname = 'area_m2'
#                        ds_all_reg[varname] = (('simulation_year'), reg_area)
#                        ds_all_reg[varname].attrs['units'] = 'm2'
#                        ds_all_reg[varname].attrs['long_name'] = 'Glacier area at timestamp'
#                        
#                        # This is the same for all files
#                        encoding = {
#                            'simulation_year': {"dtype": "int16"},
#                            'volume_m3': {"dtype": "float32"},
#                            'area_m2': {"dtype": "float32"},
#                        }
#                        
#                        # export file
#                        ds_all_fp_reg = netcdf_fp_cmip5 + 'Final/regional/' + str(region).zfill(2) + '/'
#                        if not os.path.exists(ds_all_fp_reg):
#                            os.makedirs(ds_all_fp_reg)
#                        ds_all_reg_fn = netcdf_fn.replace('_' + glacno, '')
#                        ds_all_reg_fn = ds_all_reg_fn.replace('glaciers','sum')
#                        ds_all_reg.to_netcdf(ds_all_fp_reg + ds_all_reg_fn, encoding=encoding)
    
            
#%%
# ----- SCRATCH -----
if option_check_output:
    
    # GCMs and RCP scenarios
    gcm_names = ['gfdl-esm4']
    scenarios = ['1851-1870']
    regions = [17]
    
    area_threshold = 10 # m2

    netcdf_fp_cmip5 = '/Users/drounce/Documents/glaciermip3/spc_backup/Final/per_glacier/'
    
    for reg in regions:
        
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all')
        
        reg_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/'
        
        #%%
        for scenario in scenarios:
            
            for gcm_name in gcm_names:
            
                batch_names = []
                for i in os.listdir(reg_fp):
                    if gcm_name in i and scenario in i and i.endswith('.nc'):
                        batch_names.append(i)
                
                batch_names = sorted(batch_names)
                        
                for batch_name in batch_names:
                    ds = xr.open_dataset(reg_fp + batch_name)
                    
                    batch_glacnos = [x.split('-')[1] for x in list(ds.rgi_id.values)]
                    
                    main_glac_rgi_batch = modelsetup.selectglaciersrgitable(glac_no=batch_glacnos)
                    
                    batch_area_rgi = main_glac_rgi_batch.Area.values*1e6
                    batch_area_ds = ds.area_m2.values[0,:]
                    
                    batch_area_dif = batch_area_ds - batch_area_rgi
                    
                    batch_area_dif_abs = np.absolute(batch_area_dif)
                    batch_area_dif_abs[batch_area_dif_abs < area_threshold] = 0
                    
                    if batch_area_dif_abs.sum() > 0:
                        redo_idx = np.nonzero(batch_area_dif_abs)[0]
                        
                    rgiids_redo = [batch_glacnos[x] for x in list(redo_idx)]
                    
                    print(rgiids_redo)
                    
#                    for rgiids_redo in 
                    
                    assert 1==0, 'here'

                #%%
                
#                # ----- Find batch filenames to load -----
#                if 'ssp' not in scenario:
#                    ending = 'hist'
#                else:
#                    ending = scenario.split('_')[0]
#                
#                batch_fn_prefix = 'Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name + '_' + ending
#                glacno_int = int(glacno.split('.')[1])
#                batch_startno = int(np.floor(glacno_int/1000))*1000+1
#                batch_endno = int(np.floor(glacno_int/1000))*1000+1000
#                batch_fn = batch_fn_prefix + '_Batch-' + str(batch_startno) + '-' + str(batch_endno) + '.nc'
#
#                # ----- Load batch data -----                    
#                ds = xr.open_dataset(reg_fp + batch_fn)
#                
#                rgiids_list = list(ds.rgi_id.values)
#                glacno_idx = rgiids_list.index('RGI60-' + str(reg).zfill(2) + '.' + str(glacno_int))
#                
#                glac_area_km2 = ds.area_m2.values[:,glacno_idx]/1e6
#                
#                if glac_areas_km2_gcm is None:
#                    glac_areas_km2_gcm = glac_area_km2[np.newaxis,:]
#                else:
#                    glac_areas_km2_gcm = np.concatenate((glac_areas_km2_gcm,glac_area_km2[np.newaxis,:]), axis=0)
#                
##                glac_areas_km2_gcm.append(glac_area_km2/1e6)
#                
#                if glac_area_km2[0] < 1:
#                    
#                    glac_areas_km2_all = ds.area_m2.values[0,:]/1e6
#                    print(glac_areas_km2_all[glacno_idx-3:glacno_idx+2])
#                    
#                    assert 1==0, 'here stop'
#                    
#            assert 1==0, 'here'
#            
##    # GCMs and RCP scenarios
##    gcm_names = ['gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0', 'ukesm1-0-ll']
##    scenarios = ['1851-1870', '1901-1920', '1951-1970', '1995-2014',
##                 'ssp126_2021-2040', 'ssp126_2041-2060', 'ssp126_2061-2080', 'ssp126_2081-2100',
##                 'ssp370_2021-2040', 'ssp370_2041-2060', 'ssp370_2061-2080', 'ssp370_2081-2100',
##                 'ssp585_2021-2040', 'ssp585_2041-2060', 'ssp585_2061-2080', 'ssp585_2081-2100']
##
##    netcdf_fp_cmip5 = '/Users/drounce/Documents/glaciermip3/spc_backup/Final/per_glacier/'
##
##    glacnos = ['1.15769']
##    
##    batch_glacnos_ints = list(np.arange(15001,16001))
##    batch_glacnos = ['1.' + str(x).zfill(5) for x in batch_glacnos_ints]
##    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=batch_glacnos)
##    
##    for glacno in glacnos:
##        
##        print(glacno)
##
##        reg = int(glacno.split('.')[0])
##        
##        reg_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/'
##        
##        for scenario in scenarios:
##            
##            glac_areas_km2_gcm = None
##            
##            for gcm_name in gcm_names:
##            
##                print(scenario)
##                
##                # ----- Find batch filenames to load -----
##                if 'ssp' not in scenario:
##                    ending = 'hist'
##                else:
##                    ending = scenario.split('_')[0]
##                
##                batch_fn_prefix = 'Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name + '_' + ending
##                glacno_int = int(glacno.split('.')[1])
##                batch_startno = int(np.floor(glacno_int/1000))*1000+1
##                batch_endno = int(np.floor(glacno_int/1000))*1000+1000
##                batch_fn = batch_fn_prefix + '_Batch-' + str(batch_startno) + '-' + str(batch_endno) + '.nc'
##
##                # ----- Load batch data -----                    
##                ds = xr.open_dataset(reg_fp + batch_fn)
##                
##                rgiids_list = list(ds.rgi_id.values)
##                glacno_idx = rgiids_list.index('RGI60-' + str(reg).zfill(2) + '.' + str(glacno_int))
##                
##                glac_area_km2 = ds.area_m2.values[:,glacno_idx]/1e6
##                
##                if glac_areas_km2_gcm is None:
##                    glac_areas_km2_gcm = glac_area_km2[np.newaxis,:]
##                else:
##                    glac_areas_km2_gcm = np.concatenate((glac_areas_km2_gcm,glac_area_km2[np.newaxis,:]), axis=0)
##                
###                glac_areas_km2_gcm.append(glac_area_km2/1e6)
##                
##                if glac_area_km2[0] < 1:
##                    
##                    glac_areas_km2_all = ds.area_m2.values[0,:]/1e6
##                    print(glac_areas_km2_all[glacno_idx-3:glacno_idx+2])
##                    
##                    assert 1==0, 'here stop'
##                    
##            assert 1==0, 'here'


if option_check_glaciers:
    for reg in regions:
        main_glac_rgi = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all', 
            include_landterm=True, include_laketerm=True, include_tidewater=True)
        
        #%%
        bad_batches = []
        for gcm_name in gcm_names:
            
            for scenario in scenarios:

                nbatches_expected = int(np.ceil(main_glac_rgi.shape[0]/1000))
                
                netcdf_perglac_fp = netcdf_fp_cmip5 + 'Final/per_glacier/' + str(reg).zfill(2) + '/'

                gcm_scenario_batch_fns = []
                for i in os.listdir(netcdf_perglac_fp):
                    if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name):
                        gcm_scenario_batch_fns.append(i)
                
                print(gcm_name, scenario, len(gcm_scenario_batch_fns))
                
                
                if len(gcm_scenario_batch_fns) < nbatches_expected:
                    bad_batches.append(gcm_name + '_' + scenario)

                
    netcdf_fp_cmip5 = '/Users/drounce/Documents/glaciermip3/spc_backup/'


#%%
if option_standardize_reg:
    for reg in regions:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all', 
            include_landterm=True, include_laketerm=True, include_tidewater=True)

        rgiids_quality = []
        vol_init_allcombos = np.zeros((main_glac_rgi_all.shape[0], len(gcm_names)*len(scenarios)))
        ncol = 0
        for gcm_name in gcm_names:
            
            for scenario in scenarios:

                nbatches_expected = int(np.ceil(main_glac_rgi_all.shape[0]/1000))
                
                netcdf_perglac_fp = netcdf_fp_cmip5 + 'Final/per_glacier/' + str(reg).zfill(2) + '-v2/'

                gcm_scenario_batch_fns = []
                for i in os.listdir(netcdf_perglac_fp):
                    if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name):
                        gcm_scenario_batch_fns.append(i)
                
                rgiids_gcm_scenario = []
                rgiid_vol_dict = {}
                for batch_fn in gcm_scenario_batch_fns:
                    ds = xr.open_dataset(netcdf_perglac_fp + batch_fn)
    
                    rgiids_batch = list(ds.rgi_id.values)
                    
                    rgiids_gcm_scenario += rgiids_batch
                    
                    vol_init_batch = list(ds.volume_m3.values[0,:])
                    rgiid_vol_dict_batch = dict(zip(rgiids_batch, vol_init_batch))
                    
                    rgiid_vol_dict.update(rgiid_vol_dict_batch)

                #%%
                vol_init_gcmscen = main_glac_rgi_all.RGIId.map(rgiid_vol_dict).values
                vol_init_allcombos[:,ncol] = vol_init_gcmscen
                ncol += 1
                #%%
            
                print(gcm_name, scenario, len(rgiids_gcm_scenario))
                
                if len(rgiids_quality) == 0:
                    rgiids_quality = rgiids_gcm_scenario
                else:
                    rgiids_quality = list(set(rgiids_quality).intersection(rgiids_gcm_scenario))
                
                print('  # rgiids all:', len(rgiids_quality))
        
        rgiids_quality = sorted(rgiids_quality)
    
        # Glaciers with successful runs to process
        glacno_list_quality = [x.split('-')[1] for x in rgiids_quality]
        main_glac_rgi_quality = modelsetup.selectglaciersrgitable(glac_no=glacno_list_quality)
        
        print('\nGCM/RCPs successfully simulated all:\n  -', main_glac_rgi_quality.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi_quality.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi_quality.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi_quality.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        
        # Mean volume
        vol_init_mean = np.nanmean(vol_init_allcombos, axis=1)
        
        if len(np.where(np.isnan(vol_init_mean))) > 0:
            rgiids_exclude = list(main_glac_rgi_all.loc[np.where(np.isnan(vol_init_mean))[0],'RGIId'].values)
        else:
            rgiids_exclude = []
            
        glacno_list_exclude = [x.split('-')[1] for x in rgiids_exclude]
        main_glac_rgi_exclude = modelsetup.selectglaciersrgitable(glac_no=glacno_list_exclude)
        print('\n  Excluded', main_glac_rgi_exclude.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi_exclude.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi_exclude.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi_exclude.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        
        #%% ----- LOOP THROUGH SIMULATIONS AND AGGREGATE REGIONAL ESTIMATES FILLING IN GLACIERS BASED ON REGIONAL MEANS -----
        for gcm_name in gcm_names:
            
            for scenario in scenarios:
                
                print(gcm_name, scenario)

                gcm_scenario_batch_fns = []
                for i in os.listdir(netcdf_perglac_fp):
                    if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name):
                        gcm_scenario_batch_fns.append(i)
                
                # Order batch filenames for aggregation
                gcm_scenario_batch_fns_count = [int(x.split('-')[-2]) for x in gcm_scenario_batch_fns]
                gcm_scenario_batch_fns_zip = sorted(zip(gcm_scenario_batch_fns_count, gcm_scenario_batch_fns))
                gcm_scenario_batch_fns = [x for _, x in gcm_scenario_batch_fns_zip]

                # Aggregate batches
                ds = None
                for batch_fn in gcm_scenario_batch_fns:
                    ds_batch = xr.open_dataset(netcdf_perglac_fp + batch_fn)
                    
                    if ds is None:
                        ds = ds_batch
                    else:
                        ds = xr.concat([ds, ds_batch], 'rgi_id')
                        
                # ----- CREATE AND FILL AN EMPTY DATASET -----
                # Run info
                contributor = 'Rounce'
                reg_str = str(reg).zfill(2)
                rgi_reg = 'rgi' + reg_str
                agg_level = 'glaciers'
                gcm = gcm_name
                if 'ssp' in scenario:
                    ssp = scenario.split('_')[0]
                    yr_str = scenario.split('_')[1]
                else:
                    ssp = 'hist'
                    yr_str = scenario
                    
                filename = f'{contributor}_{rgi_reg}_{agg_level}_{scenario}_{gcm}_{ssp}.nc'
                                
                ds_glac = xr.Dataset()
        
                ds_glac.attrs['contributor'] = contributor
                ds_glac.attrs['contributor_email'] = 'drounce@cmu.edu'
                ds_glac.attrs['creation_date'] = ds.creation_date
                ds_glac.attrs['rgi-region'] = rgi_reg
                ds_glac.attrs['aggregation-level'] = agg_level
                ds_glac.attrs['period'] = scenario
                ds_glac.attrs['gcm'] = gcm
                ds_glac.attrs['ssp'] = ssp
                ds_glac.attrs['information'] = 'PyGEM for mass balance and calibration with OGGM for glacier dynamics'
                ds_glac.attrs['stop_criterion'] = 'Simulations were stopped if volume was 0 for 20 years or 100-yr avg mb was within +/- 10 mm w.e.'
                
                ds_glac['simulation_year'] = (('simulation_year'), list(ds.simulation_year.values))
                ds_glac['rgi_id'] = (('rgi_id'), list(main_glac_rgi_all.RGIId.values))
                
                # ----- VOLUME FILLED -----
                varname = 'volume_m3'
                volume = np.zeros((len(ds.simulation_year.values),main_glac_rgi_all.shape[0]))
                volume[:] = np.nan
                
                rgiids_ds = list(ds.rgi_id.values)
                volume_ds = ds.volume_m3.values
                for ncol, rgiid in enumerate(main_glac_rgi_all.RGIId.values): 
                    try:
                        rgiid_idx = rgiids_ds.index(rgiid)
                        volume[:,ncol] = volume_ds[:,rgiid_idx]
                    except:
                        pass
                # Fill remaining values
                fill_idxs = np.where(np.isnan(volume[0,:]))[0]
                rgiids_fill = ds_glac.rgi_id.values[fill_idxs]
                
                volume_relative = np.nansum(volume, axis=1) / np.nansum(volume[0,:])
                
                if len(fill_idxs) > 0:
                    
                    for fill_idx in fill_idxs:
                        fill_vol_init = vol_init_mean[fill_idx]
                        if not np.isnan(fill_vol_init):
                            volume[:,fill_idx] = fill_vol_init * volume_relative
                                 
                ds_glac[varname] = (('simulation_year', 'rgi_id'), volume)
                ds_glac[varname].attrs['units'] = 'm3'
                ds_glac[varname].attrs['long_name'] = 'Glacier volume at timestamp'
                
                # ----- AREA FILLED -----
                varname = 'area_m2'
                area = np.zeros((len(ds.simulation_year.values),main_glac_rgi_all.shape[0]))
                area[:] = np.nan
                area_ds = ds.area_m2.values
                for ncol, rgiid in enumerate(main_glac_rgi_all.RGIId.values): 
                    try:
                        rgiid_idx = rgiids_ds.index(rgiid)
                        area[:,ncol] = area_ds[:,rgiid_idx]
                    except:
                        pass
                # Fill remaining values
                fill_idxs = np.where(np.isnan(area[0,:]))[0]
                rgiids_fill = ds_glac.rgi_id.values[fill_idxs]
                
                area_relative = np.nansum(area, axis=1) / np.nansum(area[0,:])
                
                if len(fill_idxs) > 0:
                    
                    for fill_idx in fill_idxs:
                        fill_area_init = main_glac_rgi_all.loc[fill_idx,'Area']*1e6
                        if not main_glac_rgi_all.loc[fill_idx,'RGIId'] in rgiids_exclude: 
                            area[:,fill_idx] = fill_area_init * area_relative

                ds_glac[varname] = (('simulation_year', 'rgi_id'), area)
                ds_glac[varname].attrs['units'] = 'm2'
                ds_glac[varname].attrs['long_name'] = 'Glacier area at timestamp'
                

                #%% ----- EXPORT REGIONAL DATASET -----
                ds_all_reg = xr.Dataset()

                for attr in ds_glac.attrs.keys():
                    ds_all_reg.attrs[attr] = ds_glac.attrs[attr]
                   
                ds_all_reg['simulation_year'] = (('simulation_year'), ds_glac.simulation_year.values)
                
                varname = 'volume_m3'
                reg_vol = np.nansum(volume, axis=1)
                ds_all_reg[varname] = (('simulation_year',), reg_vol)
                ds_all_reg[varname].attrs['units'] = 'm3'
                ds_all_reg[varname].attrs['long_name'] = 'Glacier volume at timestamp'
                
                varname = 'area_m2'
                reg_area = np.nansum(area, axis=1)
                ds_all_reg[varname] = (('simulation_year'), reg_area)
                ds_all_reg[varname].attrs['units'] = 'm2'
                ds_all_reg[varname].attrs['long_name'] = 'Glacier area at timestamp'
                
                # This is the same for all files
                encoding = {
                    'simulation_year': {"dtype": "int16"},
                    'volume_m3': {"dtype": "float32"},
                    'area_m2': {"dtype": "float32"},
                }
                
                # export file
                ds_all_fp_reg = netcdf_fp_cmip5 + 'Final/regional-filled/' + str(reg).zfill(2) + '/'
                if not os.path.exists(ds_all_fp_reg):
                    os.makedirs(ds_all_fp_reg)
                ds_all_reg_fn = f'{contributor}_{rgi_reg}_sum_{yr_str}_{gcm}_{ssp}.nc'
                ds_all_reg.to_netcdf(ds_all_fp_reg + ds_all_reg_fn, encoding=encoding)

                
                #%% ----- SPLIT FULL FILE INTO BATCHES -----
                rgiids_all = list(ds_glac.rgi_id.values)
                glacno_all = [int(x.split('-')[1].split('.')[1]) for x in rgiids_all]
                for n_fn, netcdf_fn in enumerate(gcm_scenario_batch_fns):
                    batch_start = int(netcdf_fn.split('-')[-2])
                    batch_end = int(netcdf_fn.split('-')[-1].split('.')[0])
                    
                    batch_start_idx = np.where(np.array(glacno_all) >= batch_start)[0][0]
                    batch_end_idx = np.where(np.array(glacno_all) <= batch_end)[0][-1]

                    ds_batch = ds_glac.isel(rgi_id = slice(batch_start_idx,batch_end_idx+1))
                    
                    # export file
                    ds_batch_fp = netcdf_fp_cmip5 + 'Final/per_glacier-filled/' + str(reg).zfill(2) + '/'
                    if not os.path.exists(ds_batch_fp):
                        os.makedirs(ds_batch_fp)

                    batch_start_str = str(batch_start)
                    batch_end_str = str(batch_end)
                    ds_batch_fn = f'{contributor}_{rgi_reg}_glaciers_{yr_str}_{gcm}_{ssp}_Batch-{batch_start_str}-{batch_end_str}.nc'
                    
                    ds_batch.to_netcdf(ds_batch_fp + ds_batch_fn, encoding=encoding)
                
                
#%%
if option_qc_area_standardize_reg:

    area_threshold = 1e3 # m2
    
    for reg in regions:
        
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all')
        
        nbatches_expected = int(np.ceil(main_glac_rgi_all.shape[0]/1000))
        
        netcdf_perglac_fp = netcdf_fp_cmip5 + 'Final/per_glacier/' + str(reg).zfill(2) + '/'
        
        vol_init_mean_fn = 'R' + str(reg).zfill(2) + '_vol_init_mean.csv'
        vol_init_mean_fp = netcdf_fp_cmip5 + 'vol_init_mean/'
        if not os.path.exists(vol_init_mean_fp):
            os.makedirs(vol_init_mean_fp)
        
        if not os.path.exists(vol_init_mean_fp + vol_init_mean_fn):
            vol_init_allcombos = np.zeros((main_glac_rgi_all.shape[0], len(gcm_names)*len(scenarios)))
            ncol = 0
            for gcm_name in gcm_names:
                for scenario in scenarios:
                    
                    print(gcm_name, scenario)
                
                    gcm_scenario_batch_fns = []
                    for i in os.listdir(netcdf_perglac_fp):
                        if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name):
                            gcm_scenario_batch_fns.append(i)
                    
                    # Order batch filenames for aggregation
                    gcm_scenario_batch_fns_count = [int(x.split('-')[-2]) for x in gcm_scenario_batch_fns]
                    gcm_scenario_batch_fns_zip = sorted(zip(gcm_scenario_batch_fns_count, gcm_scenario_batch_fns))
                    gcm_scenario_batch_fns = [x for _, x in gcm_scenario_batch_fns_zip]
    
                    # Aggregate batches
                    ds = None
                    for batch_fn in gcm_scenario_batch_fns:
                        ds_batch = xr.open_dataset(netcdf_perglac_fp + batch_fn)
                        
                        if ds is None:
                            ds = ds_batch
                        else:
                            ds = xr.concat([ds, ds_batch], 'rgi_id')
                    
                    #%%
                    # Initial volume - remove bad areas
                    glacnos_gcmscenario = [x.split('-')[1] for x in list(ds.rgi_id.values)]
                    
                    main_glac_rgi_batch = modelsetup.selectglaciersrgitable(glac_no=glacnos_gcmscenario)
                    
                    area_rgi = main_glac_rgi_batch.Area.values*1e6
                    area_ds = ds.area_m2.values[0,:]
                    
                    area_dif = np.absolute(area_ds - area_rgi)
                    
                    bad_idxs = np.where(area_dif > area_threshold)[0]
                    
                    vol_init_ds = ds.volume_m3.values[0,:]
                    vol_init_ds[bad_idxs] = np.nan
                    
                    rgiid_vol_dict = dict(zip(ds.rgi_id.values, vol_init_ds))
                    
                    vol_init_gcmscen = main_glac_rgi_all.RGIId.map(rgiid_vol_dict).values
                    vol_init_allcombos[:,ncol] = vol_init_gcmscen
                    ncol += 1
                    
            # Mean volume
            vol_init_mean = np.nanmean(vol_init_allcombos, axis=1)
            
            # Save mean volume
            np.savetxt(vol_init_mean_fp + vol_init_mean_fn, vol_init_mean, delimiter=',')
        
        else:
            vol_init_mean = np.genfromtxt(vol_init_mean_fp + vol_init_mean_fn, delimiter=',')
            
        if len(np.where(np.isnan(vol_init_mean))) > 0:
            rgiids_exclude = list(main_glac_rgi_all.loc[np.where(np.isnan(vol_init_mean))[0],'RGIId'].values)
        else:
            rgiids_exclude = []
            
        glacno_list_exclude = [x.split('-')[1] for x in rgiids_exclude]
        if len(glacno_list_exclude) > 0:
            main_glac_rgi_exclude = modelsetup.selectglaciersrgitable(glac_no=glacno_list_exclude)
            print('\n  Excluded', main_glac_rgi_exclude.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
                  '(', np.round(main_glac_rgi_exclude.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
            print('  -', np.round(main_glac_rgi_exclude.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
                  '(', np.round(main_glac_rgi_exclude.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        else:
            print('All glaciers were modeled at least once')
        
        #%%
        # ----- PROCESS DATA FILLING BAD VALUES -----
        for gcm_name in gcm_names:
            
            for scenario in scenarios:
                
                print(gcm_name, scenario)

                gcm_scenario_batch_fns = []
                for i in os.listdir(netcdf_perglac_fp):
                    if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + scenario + '_' + gcm_name):
                        gcm_scenario_batch_fns.append(i)
                
                # Order batch filenames for aggregation
                gcm_scenario_batch_fns_count = [int(x.split('-')[-2]) for x in gcm_scenario_batch_fns]
                gcm_scenario_batch_fns_zip = sorted(zip(gcm_scenario_batch_fns_count, gcm_scenario_batch_fns))
                gcm_scenario_batch_fns = [x for _, x in gcm_scenario_batch_fns_zip]

                # Aggregate batches
                ds = None
                for batch_fn in gcm_scenario_batch_fns:
                    ds_batch = xr.open_dataset(netcdf_perglac_fp + batch_fn)
                    
                    if ds is None:
                        ds = ds_batch
                    else:
                        ds = xr.concat([ds, ds_batch], 'rgi_id')
                        
                #%% Initial volume - remove bad areas
                glacnos_gcmscenario = [x.split('-')[1] for x in list(ds.rgi_id.values)]
                
                main_glac_rgi_batch = modelsetup.selectglaciersrgitable(glac_no=glacnos_gcmscenario)
                
                area_rgi = main_glac_rgi_batch.Area.values*1e6
                area_ds = ds.area_m2.values[0,:]
                
                area_dif = np.absolute(area_ds - area_rgi)
                
                bad_idxs = np.where(area_dif > area_threshold)[0]
                
                for bad_idx in bad_idxs:
                    ds.area_m2.values[:,bad_idx] = np.nan
                    ds.volume_m3.values[:,bad_idx] = np.nan
  
                #%% ----- CREATE AND FILL AN EMPTY DATASET -----
                # Run info
                contributor = 'Rounce'
                reg_str = str(reg).zfill(2)
                rgi_reg = 'rgi' + reg_str
                agg_level = 'glaciers'
                gcm = gcm_name
                if 'ssp' in scenario:
                    ssp = scenario.split('_')[0]
                    yr_str = scenario.split('_')[1]
                else:
                    ssp = 'hist'
                    yr_str = scenario
                    
                filename = f'{contributor}_{rgi_reg}_{agg_level}_{scenario}_{gcm}_{ssp}.nc'
                                
                ds_glac = xr.Dataset()
        
                ds_glac.attrs['contributor'] = contributor
                ds_glac.attrs['contributor_email'] = 'drounce@cmu.edu'
                ds_glac.attrs['creation_date'] = ds.creation_date
                ds_glac.attrs['rgi-region'] = rgi_reg
                ds_glac.attrs['aggregation-level'] = agg_level
                ds_glac.attrs['period'] = scenario
                ds_glac.attrs['gcm'] = gcm
                ds_glac.attrs['ssp'] = ssp
                ds_glac.attrs['information'] = 'PyGEM for mass balance and calibration with OGGM for glacier dynamics'
                ds_glac.attrs['stop_criterion'] = 'Simulations were stopped if volume was 0 for 20 years or 100-yr avg mb was within +/- 10 mm w.e.'
                
                ds_glac['simulation_year'] = (('simulation_year'), list(ds.simulation_year.values))
                ds_glac['rgi_id'] = (('rgi_id'), list(main_glac_rgi_all.RGIId.values))
                
                # ----- VOLUME FILLED -----
                varname = 'volume_m3'
                volume = np.zeros((len(ds.simulation_year.values),main_glac_rgi_all.shape[0]))
                volume[:] = np.nan
                
                rgiids_ds = list(ds.rgi_id.values)
                volume_ds = ds.volume_m3.values
                for ncol, rgiid in enumerate(main_glac_rgi_all.RGIId.values): 
                    try:
                        rgiid_idx = rgiids_ds.index(rgiid)
                        volume[:,ncol] = volume_ds[:,rgiid_idx]
                    except:
                        pass
                # Fill remaining values
                fill_idxs = np.where(np.isnan(volume[0,:]))[0]
                rgiids_fill = ds_glac.rgi_id.values[fill_idxs]
                
                volume_relative = np.nansum(volume, axis=1) / np.nansum(volume[0,:])
                
                if len(fill_idxs) > 0:
                    
                    for fill_idx in fill_idxs:
                        fill_vol_init = vol_init_mean[fill_idx]
                        if not np.isnan(fill_vol_init):
                            volume[:,fill_idx] = fill_vol_init * volume_relative
                                 
                ds_glac[varname] = (('simulation_year', 'rgi_id'), volume)
                ds_glac[varname].attrs['units'] = 'm3'
                ds_glac[varname].attrs['long_name'] = 'Glacier volume at timestamp'
                
                # ----- AREA FILLED -----
                varname = 'area_m2'
                area = np.zeros((len(ds.simulation_year.values),main_glac_rgi_all.shape[0]))
                area[:] = np.nan
                area_ds = ds.area_m2.values
                for ncol, rgiid in enumerate(main_glac_rgi_all.RGIId.values): 
                    try:
                        rgiid_idx = rgiids_ds.index(rgiid)
                        area[:,ncol] = area_ds[:,rgiid_idx]
                    except:
                        pass
                # Fill remaining values
                fill_idxs = np.where(np.isnan(area[0,:]))[0]
                rgiids_fill = ds_glac.rgi_id.values[fill_idxs]
                
                area_relative = np.nansum(area, axis=1) / np.nansum(area[0,:])
                
                if len(fill_idxs) > 0:
                    
                    for fill_idx in fill_idxs:
                        fill_area_init = main_glac_rgi_all.loc[fill_idx,'Area']*1e6
                        if not main_glac_rgi_all.loc[fill_idx,'RGIId'] in rgiids_exclude: 
                            area[:,fill_idx] = fill_area_init * area_relative

                ds_glac[varname] = (('simulation_year', 'rgi_id'), area)
                ds_glac[varname].attrs['units'] = 'm2'
                ds_glac[varname].attrs['long_name'] = 'Glacier area at timestamp'
                
                #%% ----- EXPORT REGIONAL DATASET -----
                ds_all_reg = xr.Dataset()

                for attr in ds_glac.attrs.keys():
                    ds_all_reg.attrs[attr] = ds_glac.attrs[attr]
                   
                ds_all_reg['simulation_year'] = (('simulation_year'), ds_glac.simulation_year.values)
                
                varname = 'volume_m3'
                reg_vol = np.nansum(volume, axis=1)
                ds_all_reg[varname] = (('simulation_year',), reg_vol)
                ds_all_reg[varname].attrs['units'] = 'm3'
                ds_all_reg[varname].attrs['long_name'] = 'Glacier volume at timestamp'
                
                varname = 'area_m2'
                reg_area = np.nansum(area, axis=1)
                ds_all_reg[varname] = (('simulation_year'), reg_area)
                ds_all_reg[varname].attrs['units'] = 'm2'
                ds_all_reg[varname].attrs['long_name'] = 'Glacier area at timestamp'
                
                # This is the same for all files
                encoding = {
                    'simulation_year': {"dtype": "int16"},
                    'volume_m3': {"dtype": "float32"},
                    'area_m2': {"dtype": "float32"},
                }
                
                # export file
                ds_all_fp_reg = netcdf_fp_cmip5 + 'Final/regional-filled/' + str(reg).zfill(2) + '/'
                if not os.path.exists(ds_all_fp_reg):
                    os.makedirs(ds_all_fp_reg)
                ds_all_reg_fn = f'{contributor}_{rgi_reg}_sum_{yr_str}_{gcm}_{ssp}.nc'
                ds_all_reg.to_netcdf(ds_all_fp_reg + ds_all_reg_fn, encoding=encoding)

                
                #%% ----- SPLIT FULL FILE INTO BATCHES -----
                rgiids_all = list(ds_glac.rgi_id.values)
                glacno_all = [int(x.split('-')[1].split('.')[1]) for x in rgiids_all]
                for n_fn, netcdf_fn in enumerate(gcm_scenario_batch_fns):
                    batch_start = int(netcdf_fn.split('-')[-2])
                    batch_end = int(netcdf_fn.split('-')[-1].split('.')[0])
                    
                    batch_start_idx = np.where(np.array(glacno_all) >= batch_start)[0][0]
                    batch_end_idx = np.where(np.array(glacno_all) <= batch_end)[0][-1]

                    ds_batch = ds_glac.isel(rgi_id = slice(batch_start_idx,batch_end_idx+1))
                    
                    # export file
                    ds_batch_fp = netcdf_fp_cmip5 + 'Final/per_glacier-filled/' + str(reg).zfill(2) + '/'
                    if not os.path.exists(ds_batch_fp):
                        os.makedirs(ds_batch_fp)

                    batch_start_str = str(batch_start)
                    batch_end_str = str(batch_end)
                    ds_batch_fn = f'{contributor}_{rgi_reg}_glaciers_{yr_str}_{gcm}_{ssp}_Batch-{batch_start_str}-{batch_end_str}.nc'
                    
                    ds_batch.to_netcdf(ds_batch_fp + ds_batch_fn, encoding=encoding)
       
        
#%%
if option_qc_growing_glaciers:

    for reg in regions:
        
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all')
        
        nbatches_expected = int(np.ceil(main_glac_rgi_all.shape[0]/1000))
        
        netcdf_perglac_fp = netcdf_fp_cmip5 + 'Final/per_glacier-filled/' + str(reg).zfill(2) + '/'
        
        for gcm_name in gcm_names:
            for scenario in scenarios:
                
                print(gcm_name, scenario)
                
                if 'ssp' in scenario:
                    ssp = scenario.split('_')[0]
                    yr_str = scenario.split('_')[1]
                else:
                    ssp = 'hist'
                    yr_str = scenario
            
                gcm_scenario_batch_fns = []
                for i in os.listdir(netcdf_perglac_fp):
                    if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + yr_str + '_' + gcm_name + '_' + ssp):
                        gcm_scenario_batch_fns.append(i)
                
                # Order batch filenames for aggregation
                gcm_scenario_batch_fns_count = [int(x.split('-')[-2]) for x in gcm_scenario_batch_fns]
                gcm_scenario_batch_fns_zip = sorted(zip(gcm_scenario_batch_fns_count, gcm_scenario_batch_fns))
                gcm_scenario_batch_fns = [x for _, x in gcm_scenario_batch_fns_zip]

                # Aggregate batches
                ds = None
                for batch_fn in gcm_scenario_batch_fns:
                    ds_batch = xr.open_dataset(netcdf_perglac_fp + batch_fn)
                    
                    if ds is None:
                        ds = ds_batch
                    else:
                        ds = xr.concat([ds, ds_batch], 'rgi_id')
                        
#                reg_vol_raw = np.sum(ds.volume_m3.values, axis=1)

                vol = ds.volume_m3.values
                vol_runningmean = uniform_filter(vol, size=(100,1))
                if reg in [2,8,10,11,12,13,14,15,16,18]:
                    check_yr = 1500
                else:
                    check_yr = 4000
                vol_chg = vol_runningmean[-1,:] / vol_runningmean[check_yr,:]
                vol_chg[np.isnan(vol_chg)] = 0
                vol_chg_init = vol[-1,:] / vol[0,:]
                vol_chg[vol_chg_init < 1] = 0
                check_idxs = np.where(vol_chg > 1)[0]
                
                #%%
                if len(check_idxs) > 0:
                    vol_equi = np.nan
                    for check_idx in check_idxs:
                        vol_glac = ds.volume_m3.values[:,check_idx]
                        area_glac = ds.area_m2.values[:,check_idx]
                        spec_mb = (vol_glac[1:] - vol_glac[:-1]) / area_glac[:-1] * pygem_prms.density_ice
                        spec_mb_avg = np.zeros(vol_glac.shape)
                        for yr_idx in np.arange(vol.shape[0]):
                            if yr_idx > 100:
                                
                                # Specific mass balance
                                spec_mb_avg[yr_idx] = spec_mb[yr_idx-100:yr_idx].mean()
                                if np.abs(spec_mb_avg[yr_idx]) < 10:
                                    vol_equi = vol_glac[yr_idx-20:yr_idx].mean()
                                    area_equi = area_glac[yr_idx-20:yr_idx].mean()
                                    ds.volume_m3.values[yr_idx:,check_idx] = vol_equi
                                    ds.area_m2.values[yr_idx:, check_idx] = area_equi
                                    break
                                
                        # Major Growth
                        vol_runningmean = uniform_filter(vol_glac, 100)    
                        vol_runningmean_dif = vol_runningmean[1:] - vol_runningmean[:-1]
                        break_idx_raw = np.where(vol_runningmean_dif > 0)[0]
                        if len(break_idx_raw) > 1:
                            break_idx = [x for x in list(break_idx_raw) if x > 100][0]
                            vol_equi = vol_glac[break_idx-20:break_idx].mean()
                            area_equi = area_glac[break_idx-20:break_idx].mean()
                            ds.volume_m3.values[break_idx:,check_idx] = vol_equi
                            ds.area_m2.values[break_idx:, check_idx] = area_equi
                     
#                reg_vol = np.sum(ds.volume_m3.values, axis=1)
#                years = ds.simulation_year.values
#                fig, ax = plt.subplots()
#                ax.plot(years, reg_vol_raw)
#                ax.plot(years, reg_vol)
#                plt.show()
                            
                #%% ----- EXPORT REGIONAL DATASET -----
                ds_all_reg = xr.Dataset()

                for attr in ds.attrs.keys():
                    ds_all_reg.attrs[attr] = ds.attrs[attr]
                   
                ds_all_reg['simulation_year'] = (('simulation_year'), ds.simulation_year.values)
                
                varname = 'volume_m3'
                reg_vol = np.nansum(ds.volume_m3.values, axis=1)
                ds_all_reg[varname] = (('simulation_year',), reg_vol)
                ds_all_reg[varname].attrs['units'] = 'm3'
                ds_all_reg[varname].attrs['long_name'] = 'Glacier volume at timestamp'
                
                varname = 'area_m2'
                reg_area = np.nansum(ds.area_m2.values, axis=1)
                ds_all_reg[varname] = (('simulation_year'), reg_area)
                ds_all_reg[varname].attrs['units'] = 'm2'
                ds_all_reg[varname].attrs['long_name'] = 'Glacier area at timestamp'
                
                # This is the same for all files
                encoding = {
                    'simulation_year': {"dtype": "int16"},
                    'volume_m3': {"dtype": "float32"},
                    'area_m2': {"dtype": "float32"},
                }
                
                contributor = 'Rounce'
                reg_str = str(reg).zfill(2)
                rgi_reg = 'rgi' + reg_str
                agg_level = 'glaciers'
                gcm = gcm_name
                if 'ssp' in scenario:
                    ssp = scenario.split('_')[0]
                    yr_str = scenario.split('_')[1]
                else:
                    ssp = 'hist'
                    yr_str = scenario
                
                # export file
                ds_all_fp_reg = netcdf_fp_cmip5 + 'Final/regional-filled-norunaway/' + str(reg).zfill(2) + '/'
                if not os.path.exists(ds_all_fp_reg):
                    os.makedirs(ds_all_fp_reg)
                ds_all_reg_fn = f'{contributor}_{rgi_reg}_sum_{yr_str}_{gcm}_{ssp}.nc'
                ds_all_reg.to_netcdf(ds_all_fp_reg + ds_all_reg_fn, encoding=encoding)

                
                #%% ----- SPLIT FULL FILE INTO BATCHES -----
                rgiids_all = list(ds.rgi_id.values)
                glacno_all = [int(x.split('-')[1].split('.')[1]) for x in rgiids_all]
                for n_fn, netcdf_fn in enumerate(gcm_scenario_batch_fns):
                    batch_start = int(netcdf_fn.split('-')[-2])
                    batch_end = int(netcdf_fn.split('-')[-1].split('.')[0])
                    
                    batch_start_idx = np.where(np.array(glacno_all) >= batch_start)[0][0]
                    batch_end_idx = np.where(np.array(glacno_all) <= batch_end)[0][-1]

                    ds_batch = ds.isel(rgi_id = slice(batch_start_idx,batch_end_idx+1))
                    
                    # export file
                    ds_batch_fp = netcdf_fp_cmip5 + 'Final/per_glacier-filled-norunaway/' + str(reg).zfill(2) + '/'
                    if not os.path.exists(ds_batch_fp):
                        os.makedirs(ds_batch_fp)

                    batch_start_str = str(batch_start)
                    batch_end_str = str(batch_end)
                    ds_batch_fn = f'{contributor}_{rgi_reg}_glaciers_{yr_str}_{gcm}_{ssp}_Batch-{batch_start_str}-{batch_end_str}.nc'
                    
                    ds_batch.to_netcdf(ds_batch_fp + ds_batch_fn, encoding=encoding)
                    
                #%%
                

# ----- SCRIPT PLOTS THE OUTPUT OF VOLUME AND AREA FOR EACH REGION AND SCENARIO -----
if option_plot_output:

    for reg in regions:
        
        reg_fp = netcdf_fp_cmip5 + '/Final/regional-filled-norunaway/' + str(reg).zfill(2) + '/'
        reg_fig_fp = netcdf_fp_cmip5 + 'Final/figures/regional-filled-norunaway/' + str(reg).zfill(2) + '/'
        if not os.path.exists(reg_fig_fp):
            os.makedirs(reg_fig_fp)
            
        # ----- Load glaciers that are included -----
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all')
        
        vol_init_mean_fn = 'R' + str(reg).zfill(2) + '_vol_init_mean.csv'
        vol_init_mean_fp = netcdf_fp_cmip5 + 'vol_init_mean/'
        vol_init_mean = np.genfromtxt(vol_init_mean_fp + vol_init_mean_fn, delimiter=',')
            
        if len(np.where(np.isnan(vol_init_mean))) > 0:
            rgiids_exclude = list(main_glac_rgi_all.loc[np.where(np.isnan(vol_init_mean))[0],'RGIId'].values)
        else:
            rgiids_exclude = []
            
        glacno_list_exclude = [x.split('-')[1] for x in rgiids_exclude]
        if len(glacno_list_exclude) > 0:
            main_glac_rgi_exclude = modelsetup.selectglaciersrgitable(glac_no=glacno_list_exclude)
            print('\n  Excluded', main_glac_rgi_exclude.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
                  '(', np.round(main_glac_rgi_exclude.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
            print('  -', np.round(main_glac_rgi_exclude.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
                  '(', np.round(main_glac_rgi_exclude.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        else:
            print('All glaciers were modeled at least once')
        
        vol_gcm_scenarios = None
        for nscenario, scenario in enumerate(scenarios):
            
            reg_vol_gcms = None
            
            for ngcm, gcm_name in enumerate(gcm_names):
                
                if 'ssp' in scenario:
                    ssp = scenario.split('_')[0]
                    yr_str = scenario.split('_')[1]
                else:
                    ssp = 'hist'
                    yr_str = scenario
                    
                reg_fn = 'Rounce_rgi' + str(reg).zfill(2) + '_sum_' + yr_str + '_' + gcm_name + '_' + ssp + '.nc'
                    
                ds = xr.open_dataset(reg_fp + reg_fn)
                
                reg_area_frac = ds.area_m2.values[0]/1e6 / main_glac_rgi_all.Area.sum()
                farinotti_vol_dict = {1:18.98, 2:1.06, 3:28.33, 4:8.61, 5:15.69, 6:3.77, 7:7.47, 8:0.30, 9:14.64,
                                      10:0.14, 11:0.13, 12:0.06, 13:3.27, 14:2.87, 15:0.88, 16:0.1, 17:5.34, 18:0.07, 19:46.47}
                reg_vol_frac = ds.volume_m3.values[0]/1e12 / farinotti_vol_dict[reg]
                print(reg, 'area:', np.round(reg_area_frac,2), 'volume:', np.round(reg_vol_frac,2), gcm_name, scenario)
                
                if vol_gcm_scenarios is None:
                    vol_gcm_scenarios = np.zeros((len(gcm_names),len(scenarios),len(ds.simulation_year.values)))
                    area_gcm_scenarios = np.zeros((len(gcm_names),len(scenarios),len(ds.simulation_year.values)))
                
                vol_gcm_scenarios[ngcm,nscenario,:] = ds.volume_m3.values
                area_gcm_scenarios[ngcm,nscenario,:] = ds.area_m2.values
            
        #%%
        # ----- VOLUME -----
        years = ds.simulation_year.values
        gcm_color_dict = {'gfdl-esm4':'#2171b5',
                          'ipsl-cm6a-lr':'#bae4b3',
                          'mpi-esm1-2-hr':'#fdbe85',
                          'mri-esm2-0':'#fcae91',
                          'ukesm1-0-ll':'#2ca25f'}
        for nscenario, scenario in enumerate(scenarios):
            # Plot lines as well as mean
            fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                                   gridspec_kw = {'wspace':0, 'hspace':0})
    
            # Median and absolute median deviation
            for ngcm, gcm in enumerate(gcm_names):
                reg_vol_gcm_scenario = vol_gcm_scenarios[ngcm,nscenario,:]
                ax[0,0].plot(years, reg_vol_gcm_scenario/1e12, color=gcm_color_dict[gcm], linestyle='-', linewidth=1, zorder=3, label=gcm)
               
            ax[0,0].set_ylim(0)
            ax[0,0].set_ylabel('Volume (10$^{3}$ km$^{3}$)')
            ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                         verticalalignment='top', transform=ax[0,0].transAxes)
            ax[0,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,)
            ax[0,0].tick_params(direction='inout', right=True)
            # Save figure
            fig.set_size_inches(4,3)
            fig_fp_vol = reg_fig_fp + 'volume/'
            if not os.path.exists(fig_fp_vol):
                os.makedirs(fig_fp_vol)
            fig_fn = str(reg).zfill(2) + '_' + 'volume_' + scenario + '.png'
            fig.savefig(fig_fp_vol + fig_fn, bbox_inches='tight', dpi=300)
           
                
        #%%
        # ----- MULTI-GCM MEAN -----
        scenarios_color_dict = {'1851-1870':'#2171b5', '1901-1920':'#6baed6', '1951-1970':'#bdd7e7', '1995-2014':'#eff3ff',
                                 'ssp126_2021-2040':'#bae4b3', 'ssp126_2041-2060':'#74c476', 'ssp126_2061-2080':'#31a354', 'ssp126_2081-2100':'#006d2c',
                                 'ssp370_2021-2040':'#fdbe85', 'ssp370_2041-2060':'#fd8d3c', 'ssp370_2061-2080':'#e6550d', 'ssp370_2081-2100':'#a63603',
                                 'ssp585_2021-2040':'#fcae91', 'ssp585_2041-2060':'#fb6a4a', 'ssp585_2061-2080':'#de2d26', 'ssp585_2081-2100':'#a50f15'}
        # Plot lines as well as mean
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})

        # Median and absolute median deviation
        reg_vol_multigcm_mean = np.mean(vol_gcm_scenarios, axis=0)
        
        for nscenario, scenario in enumerate(scenarios):
            ax[0,0].plot(years, reg_vol_multigcm_mean[nscenario,:]/1e12, color=scenarios_color_dict[scenario], linestyle='-', linewidth=1, zorder=3, label=scenario)
           
        ax[0,0].set_ylim(0)
        ax[0,0].set_ylabel('Volume (10$^{3}$ km$^{3}$)')
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(fontsize=8, labelspacing=0.25, handlelength=1, handletextpad=0.25,bbox_to_anchor=(1.01, 1))
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig.set_size_inches(4,3)
        fig_fp_vol = reg_fig_fp + '../all_multigcm_mean/'
        if not os.path.exists(fig_fp_vol):
            os.makedirs(fig_fp_vol)
        fig_fn = str(reg).zfill(2) + '_' + 'volume_multigcm_mean.png'
        fig.savefig(fig_fp_vol + fig_fn, bbox_inches='tight', dpi=300)
            

#%%
if scratch:
    
    #%%
#    ds = xr.open_dataset('/Users/drounce/Documents/glaciermip3/Output/simulations/01/mri-esm2-0/ssp585_2081-2100/Rounce_rgi01_glaciers_ssp585_2081-2100_mri-esm2-0_ssp585_1.26736.nc')
#    ds = xr.open_dataset('/Users/drounce/Documents/glaciermip3/Output/simulations/19/gfdl-esm4/ssp585_2081-2100/Rounce_rgi19_glaciers_ssp585_2081-2100_gfdl-esm4_ssp585_19.02147.nc')
    ds = xr.open_dataset('/Users/drounce/Documents/glaciermip3/Output/simulations/19/gfdl-esm4/ssp126_2081-2100/Rounce_rgi19_glaciers_ssp126_2081-2100_gfdl-esm4_ssp126_19.02147.nc')
    
    
    years = ds.simulation_year.values
    vol = ds.volume_m3.values[:,0]
    
    fig, ax = plt.subplots()
    ax.plot(years, vol)
    plt.show()
    #%%
    
#    for reg in regions:
#        
#        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all',rgi_glac_number='all')
#        
#        nbatches_expected = int(np.ceil(main_glac_rgi_all.shape[0]/1000))
#        
#        netcdf_perglac_fp = netcdf_fp_cmip5 + 'Final/per_glacier-filled/' + str(reg).zfill(2) + '/'
#        
#        for gcm_name in gcm_names:
#            for scenario in scenarios:
#                
#                print(gcm_name, scenario)
#                
#                if 'ssp' in scenario:
#                    ssp = scenario.split('_')[0]
#                    yr_str = scenario.split('_')[1]
#                else:
#                    ssp = 'hist'
#                    yr_str = scenario
#            
#                gcm_scenario_batch_fns = []
#                for i in os.listdir(netcdf_perglac_fp):
#                    if i.startswith('Rounce_rgi' + str(reg).zfill(2) + '_glaciers_' + yr_str + '_' + gcm_name + '_' + ssp):
#                        gcm_scenario_batch_fns.append(i)
#                
#                # Order batch filenames for aggregation
#                gcm_scenario_batch_fns_count = [int(x.split('-')[-2]) for x in gcm_scenario_batch_fns]
#                gcm_scenario_batch_fns_zip = sorted(zip(gcm_scenario_batch_fns_count, gcm_scenario_batch_fns))
#                gcm_scenario_batch_fns = [x for _, x in gcm_scenario_batch_fns_zip]
#
#                # Aggregate batches
#                ds = None
#                for batch_fn in gcm_scenario_batch_fns:
#                    ds_batch = xr.open_dataset(netcdf_perglac_fp + batch_fn)
#                    
#                    if ds is None:
#                        ds = ds_batch
#                    else:
#                        ds = xr.concat([ds, ds_batch], 'rgi_id')
#                        
##                reg_vol_raw = np.sum(ds.volume_m3.values, axis=1)
#
#                vol = ds.volume_m3.values
#                area = ds.area_m2.values
#                
#                #%%
##                area_init = area[]
     
#%%
print('Total processing time:', time.time()-time_start, 's')