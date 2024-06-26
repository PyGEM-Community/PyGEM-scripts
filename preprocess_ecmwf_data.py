"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os, sys
import argparse
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Local libraries
import pygem_input as pygem_prms

#%% TO-DO LIST:
# - clean up create lapse rate input data (put it all in pygem_prms.py)

#%%
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    createlapserates : int
        Switch for processing lapse rates (default = 0 (no))
    createtempstd : int
        Switch for processing hourly temp data into monthly standard deviation (default = 0 (no))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="select pre-processing options")
    # add arguments
    parser.add_argument('-createlapserates', action='store_true',
                        help='flag, create lapse rates')
    parser.add_argument('-createtempstd', action='store_true',
                        help='flag, create temperature std of daily data')
    parser.add_argument('-mergetimes', action='store_true',
                        help='flag, merge files [hard-coded]')
    return parser

if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()
    
    #%% Option to merge files together
    if args.mergetimes:
        # Merge completed files together
        ds_fp = pygem_prms.main_directory + '/../climate_data/ERA5/'
        
#        vn2process = 't2m'
#        vn2process = 'tp'
        vn2process = 'lapserate'
        
        if vn2process in ['t2m']:
            # Temperature data
            ds_merged_fn = 'ERA5_temp_monthly_1979_2023.nc'
            ds_fns = ['ERA5_temp_monthly.nc', 'ERA5_temp_monthly_2020_2023.nc']
            encoding = {'t2m':{'_FillValue': False,
                                   'zlib':True,
                                   'complevel':9}}
        elif vn2process in ['tp']:
            # Precipitation data
            ds_merged_fn = 'ERA5_totalprecip_monthly_1979_2023.nc'
            ds_fns = ['ERA5_totalprecip_monthly.nc', 'ERA5_totalprecip_monthly_2020_2023.nc']
            encoding = {'tp':{'_FillValue': False,
                                   'zlib':True,
                                   'complevel':9}}
        elif vn2process in ['lapserate']:
            # Precipitation data
            ds_merged_fn = 'ERA5_lapserates_monthly_1979_2023.nc'
            ds_fns = ['ERA5_lapserates_monthly.nc', 'ERA5_lapserates_monthly_2020_2023.nc']
            encoding = {'lapserate':{'_FillValue': False,
                                   'zlib':True,
                                   'complevel':9}}
        else:
            assert 1==0, 'variable name not specified'
        
        time_end_idx = 492
        expver_idx = 0
        
        # Open datasets and combine
        for nfile, ds_fn in enumerate(ds_fns):
            print(ds_fn)
            ds = xr.open_dataset(ds_fp + ds_fn)
            # Merge datasets of stats into one output
            if nfile == 0:
                ds_all = ds.where(ds['time.year'] < 2020, drop=True)
            else:
                ds_all = xr.concat([ds_all, ds], dim='time')
        
        # Export to netcdf
        ds_all.to_netcdf(ds_fp + ds_merged_fn, encoding=encoding)
        
    
    #%% Create netcdf file of lapse rates from temperature pressure level data
    if args.createlapserates:

        # Input data
        gcm_fp = pygem_prms.era5_fp
        gcm_fn = pygem_prms.era5_pressureleveltemp_fn
        output_fn= pygem_prms.era5_lr_fn
        
        tempname = 't'
        levelname = 'level'
        elev_idx_max = 0
        # elev_idx_min = 20
        expver_idx = 0
        
        
        # Open dataset
        ds = xr.open_dataset(gcm_fp + gcm_fn)    
        # extract the pressure levels [Pa]
        if ds[levelname].attrs['units'] == 'millibars':
            # convert pressure levels from millibars to Pa
            levels = ds[levelname].values * 100
    
        # get highest presesure level (min elev)
        elev_idx_min = len(levels)

        # Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
        elev = (-pygem_prms.R_gas * pygem_prms.temp_std / (pygem_prms.gravity * pygem_prms.molarmass_air) * 
                np.log(levels/pygem_prms.pressure_std))
    
        # Calculate lapse rates by year
        lr = np.zeros((ds.time.shape[0], ds.latitude.shape[0], ds.longitude.shape[0]))
        for ntime, t in enumerate(ds.time.values):        
            print('time:', ntime, t)
            
            if 'expver' in ds.keys():
                ds_subset = ds[tempname][ntime, expver_idx, elev_idx_max:elev_idx_min+1, :, :].values
            else:
                ds_subset = ds[tempname][ntime, elev_idx_max:elev_idx_min+1, :, :].values
            ds_subset_reshape = ds_subset.reshape(ds_subset.shape[0],-1)
            lr[ntime,:,:] = (np.polyfit(elev[elev_idx_max:elev_idx_min+1], ds_subset_reshape, deg=1)[0]
                             .reshape(ds_subset.shape[1:]))

        # Export lapse rates with attibutes
        output_ds = ds.copy()
        output_ds = output_ds.drop('t')
        str_max = ds['level'][elev_idx_max].values
        try:
            str_min = str(ds['level'][elev_idx_min].values)
        except:
            str_min = str(ds['level'][elev_idx_min-1].values)
        levels_str = str(ds['level'][elev_idx_max].values) + ' to ' + str_min
        output_ds['lapserate'] = (('time', 'latitude', 'longitude'), lr, 
                                  {'long_name': 'lapse rate', 
                                   'units': 'degC m-1',
                                   'levels': levels_str})
        # Drop excess coordinate from 2020 data
        if 'expver' in output_ds.coords:
            output_ds = output_ds.drop('expver')
    
        encoding = {'lapserate':{'_FillValue': False,
                                 'zlib':True,
                                 'complevel':9}}
        
        output_ds.to_netcdf(gcm_fp + output_fn, encoding=encoding)
       
         
    #%%
    if args.createtempstd:
        ds_fp = pygem_prms.main_directory + '/../climate_data/ERA5/'
    #    ds_fn = 't2m_hourly_1979_1989.nc'
    #    ds_fn = 't2m_hourly_1990_1999.nc'
    #    ds_fn = 't2m_hourly_2000_2009.nc'
    #    ds_fn = 't2m_hourly_2010_2019.nc'
        ds_fn = 't2m_hourly_2020.nc'
        ds_all_fn = 'ERA5_tempstd_monthly.nc'
        merge_files = True
        expver_idx = 0
        
        # Merge completed files together
        if merge_files:
            
            #%%
            tempstd_fns = []
            for i in os.listdir(ds_fp):
                if i.startswith('ERA5_tempstd_monthly') and i.endswith('.nc'):
                    tempstd_fns.append(i)
            tempstd_fns = sorted(tempstd_fns)

            # Open datasets and combine
            for nfile, tempstd_fn in enumerate(tempstd_fns):
                print(tempstd_fn)
                ds = xr.open_dataset(ds_fp + tempstd_fn)
                # Merge datasets of stats into one output
                if nfile == 0:
                    ds_all = ds
                else:
                    ds_all = xr.concat([ds_all, ds], dim='time')
                
            # Drop excess coordinate from 2020 data
            if 'expver' in ds_all.coords:
                ds_all = ds_all.drop('expver')
                
            # Export to netcdf
            encoding = {'t2m_std':{'_FillValue': False,
                                   'zlib':True,
                                   'complevel':9}}
            ds_all.to_netcdf(ds_fp + ds_all_fn, encoding=encoding)
                
                #%%
            
        else:
        
            output_fn= 'ERA5_tempstd_monthly_' + ds_fn.split('_')[2]
            
            ds = xr.open_dataset(ds_fp + ds_fn)
        
        #    ds_subset = ds.t2m[0:30*24,:,:].values
        #    t2m_daily = np.moveaxis(np.moveaxis(ds_subset, 0, -1).reshape(-1,24).mean(axis=1)
        #                            .reshape(ds_subset.shape[1],ds_subset.shape[2],int(ds_subset.shape[0]/24)), -1, 0)
            
            # Calculate daily mean temperature
            ndays = int(ds.time.shape[0] / 24)
            t2m_daily = np.zeros((ndays, ds.latitude.shape[0], ds.longitude.shape[0]))
            for nday in np.arange(ndays):
                if nday%50 == 0:
                    print(str(nday) + ' out of ' + str(ndays))
                    
                if 'expver' in ds.keys():
                    ds_subset = ds.t2m[nday*24:(nday+1)*24, expver_idx, :, :].values
                else:
                    ds_subset = ds.t2m[nday*24:(nday+1)*24, :, :].values
                t2m_daily[nday,:,:] = (
                        np.moveaxis(np.moveaxis(ds_subset, 0, -1).reshape(-1,24).mean(axis=1)
                                    .reshape(ds_subset.shape[1],ds_subset.shape[2],int(ds_subset.shape[0]/24)), -1, 0))
        
            # Calculate monthly temperature standard deviation
            date = ds.time[::24].values
            date_month = [pd.Timestamp(date[x]).month for x in np.arange(date.shape[0])]
            date_year = [pd.Timestamp(date[x]).year for x in np.arange(date.shape[0])]
            
            date_yyyymm = [str(date_year[x]) + '-' + str(date_month[x]).zfill(2) for x in np.arange(date.shape[0])]
            date_yyyymm_unique = sorted(list(set(date_yyyymm)))
            
            t2m_monthly_std = np.zeros((len(date_yyyymm_unique), ds.latitude.shape[0], ds.longitude.shape[0]))
            date_monthly = []
            for count, yyyymm in enumerate(date_yyyymm_unique):
                if count%12 == 0:
                    print(yyyymm)
                date_idx = np.where(np.array(date_yyyymm) == yyyymm)[0]
                date_monthly.append(date[date_idx[0]])
                t2m_monthly_std[count,:,:] = t2m_daily[date_idx,:,:].std(axis=0)
        
            # Export lapse rates with attibutes
            output_ds = ds.copy()
            output_ds = output_ds.drop('t2m')
            output_ds = output_ds.drop('time')
            output_ds['time'] = date_monthly
            output_ds['t2m_std'] = (('time', 'latitude', 'longitude'), t2m_monthly_std, 
                                     {'long_name': 'monthly 2m temperature standard deviation', 
                                      'units': 'K'})
            encoding = {'t2m_std':{'_FillValue': False,
                                   'zlib':True,
                                   'complevel':9}}
            output_ds.to_netcdf(ds_fp + output_fn, encoding=encoding)
        
            # Close dataset
            ds.close()