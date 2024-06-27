""" derive binned monthly ice thickness and mass from PyGEM simulation """

# Built-in libraries
import argparse
import collections
import copy
import inspect
import multiprocessing
import os
import glob
import sys
import time
# External libraries
import pandas as pd
import pickle
import numpy as np
import xarray as xr

try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')

# Local libraries
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

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
    scenario (optional) : str
        representative concentration pathway or shared socioeconomic pathway (ex. 'rcp26', 'ssp585')
    realization (optional) : str
        single realization from large ensemble (ex. '1011.001', '1301.020')
        see CESM2 Large Ensemble Community Project by NCAR for more information
    realization_list (optional) : str
        text file that contains the realizations to be used in the model simulation
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn (optional) : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-binned_simdir', action='store', type=str, default=None,
                        help='Directory with binned simulations for which to process monthly thickness')
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-scenario', action='store', type=str, default=None,
                        help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)')
    parser.add_argument('-realization', action='store', type=str, default=None,
                        help='realization from large ensemble used for model run (ex. 1011.001 or 1301.020)')
    parser.add_argument('-realization_list', action='store', type=str, default=None,
                        help='text file full of realizations to run')
    parser.add_argument('-gcm_bc_startyear', action='store', type=int, default=pygem_prms.gcm_bc_startyear,
                        help='start year for bias correction')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms.gcm_startyear,
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms.gcm_endyear,
                        help='start year for the model run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-option_ordered', action='store_true',
                        help='Flag to keep glacier lists ordered (default is off)')
    parser.add_argument('-option_parallels', action='store_true',
                        help='Flag to use or not use parallels (default is off)')

    return parser



def get_binned_monthly(bin_massbalclim_monthly, bin_massbalclim_annual, bin_mass_annual, bin_thick_annual):
    """
    funciton to calculate the monthly binned ice thickness and mass
    from annual climatic mass balance and annual ice thickness products

    to determine monthlyt thickness and mass, we must account for flux divergence
    this is not so straight-forward, as PyGEM accounts for ice dynamics at the 
    end of each model year and not on a monthly timestep.
    here, monthly thickness and mass is determined assuming 
    the flux divergence is constant throughout the year.
    
    annual flux divergence is first estimated by combining the annual binned change in ice 
    thickness and the annual binned mass balance. then, assume flux divergence is constant 
    throughout the year (divide annual by 12 to get monthly flux divergence).

    monthly binned flux divergence can then be combined with 
    monthly binned climatic mass balance to get monthly binned change in ice thickness

    
    Parameters
    ----------
    bin_massbalclim_monthly : float
        ndarray containing the climatic mass balance for each model month computed by PyGEM
        shape : [#glac, #elevbins, #months]
    bin_massbalclim_annual : float
        ndarray containing the climatic mass balance for each model year computed by PyGEM
        shape : [#glac, #elevbins, #years]
    bin_mass_annual : float
        ndarray containing the average (or median) binned ice mass computed by PyGEM
        shape : [#glac, #elevbins, #years]
    bin_thick_annual : float
        ndarray containing the average (or median) binned ice thickness at computed by PyGEM
        shape : [#glac, #elevbins, #years]

    Returns
    -------
    bin_thick_monthly: float
        ndarray containing the binned monthly ice thickness
        shape : [#glac, #elevbins, #years]

    bin_mass_monthly: float
        ndarray containing the binned monthly ice mass
        shape : [#glac, #elevbins, #years]
    """

    # get change in thickness from previous year for each elevation bin
    delta_thick_annual = np.diff(bin_thick_annual, axis=-1)

    # get annual binned flux divergence as annual binned climatic mass balance (-) annual binned ice thickness
    # account for density contrast (convert climatic mass balance in m w.e. to m ice)
    flux_div_annual = (
            (bin_massbalclim_annual[:,:,1:] * 
            pygem_prms.density_ice / 
            pygem_prms.density_water) - 
            delta_thick_annual)

    ### to get monthly thickness and mass we need monthly flux divergence ###
    # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
    # ie. take annual values and divide by 12 - use numpy repeat to repeat values across 12 months
    flux_div_monthly = np.repeat(flux_div_annual / 12, 12, axis=-1)

    # get monthly binned change in thickness assuming constant flux divergence throughout the year
    # account for density contrast (convert monthly climatic mass balance in m w.e. to m ice)
    bin_thickchange_monthly = (
            (bin_massbalclim_monthly *
            pygem_prms.density_ice /
            pygem_prms.density_water) -
            flux_div_monthly)
    
    # get binned monthly thickness = running thickness change + initial thickness
    running_delta_thick_monthly = np.cumsum(bin_thickchange_monthly, axis=-1)
    bin_thick_monthly =  running_delta_thick_monthly + bin_thick_annual[:,:,0][:,:,np.newaxis] 

    ### get monthly mass ###
    # note, this requires knowledge of binned glacier area
    # we do not have monthly binned area (as glacier dynamics are performed on an annual timestep in PyGEM),
    # so we'll resort to using the annual binned glacier mass and thickness in order to get to binned glacier area
    ########################
    # first convert bin_mass_annual to bin_voluma_annual
    bin_volume_annual = bin_mass_annual / pygem_prms.density_ice
    # now get area: use numpy divide where denominator is greater than 0 to avoid divide error
    # note, indexing of [:,:,1:] so that annual area array has same shape as flux_div_annual
    bin_area_annual = np.divide(
            bin_volume_annual[:,:,1:], 
            bin_thick_annual[:,:,1:], 
            out=np.full(bin_thick_annual[:,:,1:].shape, np.nan), 
            where=bin_thick_annual[:,:,1:]>0)

    # tile to get monthly area, assuming area is constant thoughout the year
    bin_area_monthly = np.tile(bin_area_annual, 12)

    # combine monthly thickess and area to get mass
    bin_mass_monthly = bin_thick_monthly * bin_area_monthly * pygem_prms.density_ice

    return bin_thick_monthly, bin_mass_monthly


def update_xrdataset(input_ds, bin_thick_monthly, bin_mass_monthly):
    """
    update xarray dataset to add new fields

    Parameters
    ----------
    xrdataset : xarray Dataset
        existing xarray dataset
    newdata : ndarray 
        new data array
    description: str
        describing new data field

    output_ds : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # coordinates
    glac_values = input_ds.glac.values
    time_values = input_ds.time.values
    bin_values = input_ds.bin.values

    output_coords_dict = collections.OrderedDict()
    output_coords_dict['bin_thick_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))
    output_coords_dict['bin_mass_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['bin_thick_monthly'] = {
            'long_name': 'binned monthly ice thickness',
            'units': 'm',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice thickness binned by surface elevation'}
    output_attrs_dict['bin_mass_monthly'] = {
            'long_name': 'binned monthly ice mass',
            'units': 'kg',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice mass binned by surface elevation'}


    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        count_vn += 1
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        encoding[vn] = {'_FillValue': None,
                        'zlib':True,
                        'complevel':9
                        }    

    output_ds_all['bin_thick_monthly'].values = (
            bin_thick_monthly
            )
    output_ds_all['bin_mass_monthly'].values = (
            bin_mass_monthly
            )

    return output_ds_all, encoding


def main(list_packed_vars):
    """
    create binned monthly mass change data product
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
    Returns
    -------
    binned_ds : netcdf Dataset
        updated binned netcdf containing binned monthly ice thickness and mass
    """

    if isinstance(list_packed_vars,list):
        # Unpack variables
        parser = getparser()
        args = parser.parse_args()
        count = list_packed_vars[0]
        glac_no = list_packed_vars[1]
        gcm_name = list_packed_vars[2]
        realization = list_packed_vars[3]
        if (gcm_name != pygem_prms.ref_gcm_name) and (args.scenario is None):
            scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
        elif not args.scenario is None:
            scenario = args.scenario
        
        # ===== LOAD GLACIERS =====
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

        for glac in range(main_glac_rgi.shape[0]):
            if glac == 0:
                print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
            reg_str = str(glacier_rgi_table.O1Region).zfill(2)
            rgiid = main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId']

            # get datapath to binned datasets produced from run_simulation.py
            output_sim_binned_fp = pygem_prms.output_sim_fp + reg_str + '/' + gcm_name + '/'
            if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                output_sim_binned_fp += scenario + '/'
            output_sim_binned_fp += 'binned/'
            # Create filepath if it does not exist
            if os.path.exists(output_sim_binned_fp) == False:
                os.makedirs(output_sim_binned_fp, exist_ok=True)
            # Number of simulations
            if pygem_prms.option_calibration == 'MCMC':
                sim_iters = pygem_prms.sim_iters
            else:
                sim_iters = 1
            # Netcdf filename
            if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
                # Filename
                netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba0' +
                            '_' +  str(sim_iters) + 'sets' + '_' + str(args.gcm_startyear) + '_' + str(args.gcm_endyear) + '_binned.nc')
            elif realization is not None:
                netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' + realization + '_' +
                                str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                '_' + str(sim_iters) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                str(args.gcm_endyear) + '_binned.nc')
            else:
                netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' +
                                str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                '_' + str(sim_iters) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                str(args.gcm_endyear) + '_binned.nc')

            # open dataset
            binned_ds = xr.open_dataset(output_sim_binned_fp + netcdf_fn)

            # calculate monthly change in mass
            bin_thick_monthly, bin_mass_monthly = get_binned_monthly(
                                                        binned_ds.bin_massbalclim_monthly.values, 
                                                        binned_ds.bin_massbalclim_annual.values, 
                                                        binned_ds.bin_mass_annual.values,
                                                        binned_ds.bin_thick_annual.values
                                                        )

            # update dataset to add monthly mass change
            output_ds_binned, encoding_binned = update_xrdataset(binned_ds, bin_thick_monthly, bin_mass_monthly)

            # close input ds before write
            binned_ds.close()

            # append to existing binned netcdf
            output_ds_binned.to_netcdf(output_sim_binned_fp + netcdf_fn, mode='a', encoding=encoding_binned, engine='netcdf4')

            # close datasets
            output_ds_binned.close()

    elif os.path.isfile(list_packed_vars):
        netcdf_fn = list_packed_vars
        # open dataset
        binned_ds = xr.open_dataset(netcdf_fn)

        # calculate monthly change in mass
        bin_thick_monthly, bin_mass_monthly = get_binned_monthly(
                                                    binned_ds.bin_massbalclim_monthly.values, 
                                                    binned_ds.bin_massbalclim_annual.values, 
                                                    binned_ds.bin_mass_annual.values,
                                                    binned_ds.bin_thick_annual.values
                                                    )

        # update dataset to add monthly mass change
        output_ds_binned, encoding_binned = update_xrdataset(binned_ds, bin_thick_monthly, bin_mass_monthly)

        # close input ds before write
        binned_ds.close()

        # append to existing binned netcdf
        output_ds_binned.to_netcdf(netcdf_fn, mode='a', encoding=encoding_binned, engine='netcdf4')

        # close datasets
        output_ds_binned.close()

    return


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    args = getparser().parse_args()

    if args.binned_simdir:
        # get list of sims
        simlist = glob.glob(args.binned_simdir+'*.nc')

        # Parallel processing
        if args.option_parallels:
            print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,simlist)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(simlist)):
                main(simlist[n])

    else:
        # RGI glacier number
        if args.rgi_glac_number_fn is not None:
            with open(args.rgi_glac_number_fn, 'rb') as f:
                glac_no = pickle.load(f)
        elif pygem_prms.glac_no is not None:
            glac_no = pygem_prms.glac_no
        else:
            main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                    rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                    rgi_glac_number=pygem_prms.rgi_glac_number, glac_no=pygem_prms.glac_no,
                    include_landterm=pygem_prms.include_landterm, include_laketerm=pygem_prms.include_laketerm, 
                    include_tidewater=pygem_prms.include_tidewater, 
                    min_glac_area_km2=pygem_prms.min_glac_area_km2)
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
            scenario = args.scenario
        elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
            gcm_list = [pygem_prms.ref_gcm_name]
            scenario = args.scenario
        else:
            with open(args.gcm_list_fn, 'r') as gcm_fn:
                gcm_list = gcm_fn.read().splitlines()
                scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
                print('Found %d gcms to process'%(len(gcm_list)))
    
        # Read realizations from argument parser
        if args.realization is not None:
            realizations = [args.realization]
        elif args.realization_list is not None:
            with open(args.realization_list, 'r') as real_fn:
                realizations = list(real_fn.read().splitlines())
                print('Found %d realizations to process'%(len(realizations)))
        else:
            realizations = None
        
        # Producing realization or realization list. Best to convert them into the same format!
        # Then pass this as a list or None.
        # If passing this through the list_packed_vars, then don't go back and get from arg parser again!
    
        # Loop through all GCMs
        for gcm_name in gcm_list:
            if args.scenario is None:
                print('Processing:', gcm_name)
            elif not args.scenario is None:
                print('Processing:', gcm_name, scenario)
            # Pack variables for multiprocessing
            list_packed_vars = []          
            if realizations is not None:
                for realization in realizations:
                    for count, glac_no_lst in enumerate(glac_no_lsts):
                        list_packed_vars.append([count, glac_no_lst, gcm_name, realization])
            else:
                for count, glac_no_lst in enumerate(glac_no_lsts):
                    list_packed_vars.append([count, glac_no_lst, gcm_name, realizations])
                    
            print('len list packed vars:', len(list_packed_vars))
            
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