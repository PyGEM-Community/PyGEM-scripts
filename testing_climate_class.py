#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing climate_class products

@author: drounce
"""
# Built-in libraries
import time
# External libraries
import numpy as np
import xarray as xr
# Local libraries
import pygem.params as pygem_prms
import pygem.gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup
from pygem.class_climate import GCM



#%% Testing
if __name__ == '__main__':
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=pygem_prms.glac_no)
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2019, spinupyears=0, 
                                           option_wateryear=pygem_prms.gcm_wateryear)
    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.gcm_startyear, endyear=pygem_prms.gcm_endyear, spinupyears=pygem_prms.gcm_spinupyears,
            option_wateryear=pygem_prms.gcm_wateryear)

    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 'GFDL-ESM4', 
                 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    for gcm_name in gcm_names:
        for scenario in scenarios:
                    
            ds = xr.open_dataset(pygem_prms.cmip6_fp_prefix + gcm_name + '/' + 
                                 gcm_name + '_' + scenario + '_r1i1p1f1_tas.nc')
            
            print(gcm_name, scenario, ds.time[0].values, ds.time[-1].values)
            
            # Load GCM    
            gcm = GCM(name=gcm_name, scenario=scenario)

            # ===== LOAD CLIMATE DATA =====
            # Climate class
            if gcm_name in ['ERA5', 'ERA-Interim', 'COAWST']:
                gcm = GCM(name=gcm_name)
                if pygem_prms.option_synthetic_sim == 0:
                    assert pygem_prms.gcm_endyear <= int(time.strftime("%Y")), 'Climate data not available to gcm_endyear'
            else:
                # GCM object
                gcm = GCM(name=gcm_name, scenario=scenario)
                # Reference GCM
                ref_gcm = GCM(name=pygem_prms.ref_gcm_name)
                # Adjust reference dates in event that reference is longer than GCM data
                if pygem_prms.ref_startyear >= pygem_prms.gcm_startyear:
                    ref_startyear = pygem_prms.ref_startyear
                else:
                    ref_startyear = pygem_prms.gcm_startyear
                if pygem_prms.ref_endyear <= pygem_prms.gcm_endyear:
                    ref_endyear = pygem_prms.ref_endyear
                else:
                    ref_endyear = pygem_prms.gcm_endyear
                dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear,
                                                           spinupyears=pygem_prms.ref_spinupyears,
                                                           option_wateryear=pygem_prms.ref_wateryear)
            
            # Air temperature [degC]
            gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                          dates_table)
            if pygem_prms.option_ablation != 2:
                gcm_tempstd = np.zeros(gcm_temp.shape)
            elif pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
                gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                                main_glac_rgi, dates_table)
            elif pygem_prms.option_ablation == 2 and pygem_prms.ref_gcm_name in ['ERA5']:
                # Compute temp std based on reference climate data
                ref_tempstd, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.tempstd_fn, ref_gcm.tempstd_vn,
                                                                                    main_glac_rgi, dates_table_ref)
                # Monthly average from reference climate data
                gcm_tempstd = gcmbiasadj.monthly_avg_array_rolled(ref_tempstd, dates_table_ref, dates_table)
            else:
                gcm_tempstd = np.zeros(gcm_temp.shape)
        
            # Precipitation [m]
            gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                          dates_table)
            # Elevation [m asl]
            gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
            # Lapse rate
            if gcm_name in ['ERA-Interim', 'ERA5']:
                gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
            else:
                # Compute lapse rates based on reference climate data
                ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi,
                                                                                dates_table_ref)
                # Monthly average from reference climate data
                gcm_lr = gcmbiasadj.monthly_avg_array_rolled(ref_lr, dates_table_ref, dates_table)
                
            print('    ', 'elev:', int(gcm_elev[0]), ' T:', np.round(gcm_temp.mean(),1), 'C   P:', np.round(gcm_prec.sum() / (gcm_prec.shape[1]/12),1), 'm')
