#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20231026

@author: btobers mweather drounce

script to check for failed glaciers for a given simulation and export a pickle file containing a list of said glacier numbers to be reprocessed
"""

# imports
import os
import glob
import sys
import time
import pickle
import argparse
import xarray as xr
import numpy as np

try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
    import pygem

# Local libraries
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

def main(reg, simpath, gcm, scenario, bias_adj, gcm_bc_startyear, gcm_endyear):

    # define base directory
    base_dir = simpath + "/" + str(reg).zfill(2) + "/"


    # get all glaciers in region to see which fraction ran successfully
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                        rgi_regionsO2='all', rgi_glac_number='all', 
                                                        glac_no=None,
                                                        debug=True)

    glacno_list_all = list(main_glac_rgi_all['rgino_str'].values)

    # get list of glacier simulation files 
    sim_dir = base_dir + gcm  + '/' + scenario  + '/stats/'

    # check if gcm has given scenario
    assert os.path.isdir(sim_dir)

    # instantiate list of galcnos that are not in sim_dir
    failed_glacnos = []

    fps = glob.glob(sim_dir + '*_ba' + str(bias_adj) + '_*' + str(gcm_bc_startyear) + '_' + str(gcm_endyear) + '_all.nc')

    # get file ending
    file_ending = fps[-1][-31:]

    # Glaciers with successful runs to process
    glacno_ran = [x.split('/')[-1].split('_')[0] for x in fps]
    glacno_ran = [x.split('.')[0].zfill(2) + '.' + x[-5:] for x in glacno_ran]

    # print stats of successfully simualated glaciers
    main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all.apply(lambda x: x.rgino_str in glacno_ran, axis=1)]
    print(f'{gcm}, {scenario} glaciers successfully simulated:\n  - {main_glac_rgi.shape[0]} of {main_glac_rgi_all.shape[0]} glaciers ({np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,3)}%)')
    print(f'  - {np.round(main_glac_rgi.Area.sum(),0)} km2 of {np.round(main_glac_rgi_all.Area.sum(),0)} km2 ({np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,3)}%)')

    glacno_ran = ['{0:0.5f}'.format(float(x)) for x in glacno_ran]

    # loop through each glacier in batch list
    for i, glacno in enumerate(glacno_list_all):
        # gat glacier string and file name
        glacier_str = '{0:0.5f}'.format(float(glacno))  

        if glacier_str not in glacno_ran:
            failed_glacnos.append(glacier_str)
    return failed_glacnos


if __name__ == '__main__':

    # Set up CLI
    parser = argparse.ArgumentParser(
    description="""description: program for checking failed glacier simulations from the python glacier evolution model (PyGEM)\n\nexample call: $python check_failed_simulations.py -rgi_region01=1 -gcm_name=CanESM5 -scenrio=ssp585 -outpkl=/path/to/output/pickle/file.pkl""",
    formatter_class=argparse.RawTextHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-rgi_region01', type=int, help='Randoph Glacier Inventory 01 region', required=True)
    requiredNamed.add_argument('-gcm_name', type=str, help='Global Climate Model name (ex. CanESM5)', required=True)
    requiredNamed.add_argument('-scenario', type=str, help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)', required=True)
    parser.add_argument('-sim_path', type=str, default=None, help='PyGEM simulations filepath')
    parser.add_argument('-gcm_bc_startyear', type=int, default=None, help='Global Climate Model start year for simulations (ex. 2000)')
    parser.add_argument('-gcm_endyear', type=int, default=None, help='Global Circulation Model end year for simulations (ex. 2100)')
    parser.add_argument('-bias_adj', type=int, default=None, help='bias adjustment type (ex. ba1)')
    parser.add_argument('-outpkl', type=str, help='path to output pickle file containing list of failed glaciers')
    args = parser.parse_args()

    simpath = args.sim_path
    region = args.rgi_region01
    scenario = args.scenario
    gcm_name = args.gcm_name
    bias_adj = args.bias_adj
    gcm_bc_startyear = args.gcm_bc_startyear
    gcm_endyear = args.gcm_endyear
    outpath = args.outpkl

    if not simpath:
        simpath = pygem_prms.output_filepath + 'simulations/'

    if not region:
        region = pygem_prms.rgi_regionsO1

    if not bias_adj:
        bias_adj = pygem_prms.option_bias_adjustment

    if not gcm_bc_startyear:
        gcm_bc_startyear = pygem_prms.gcm_bc_startyear
    
    if not gcm_endyear:
        gcm_endyear = pygem_prms.gcm_endyear

    if not isinstance(region, list):
        region = [region]

    if not outpath.endswith('.pkl'):
        outpath = outpath.split('.')[0] + '.pkl'

    for reg in region:
        failed_glacs = main(reg, simpath, gcm_name, scenario, bias_adj, gcm_bc_startyear, gcm_endyear)
        if len(failed_glacs)>0:
            with open(outpath, 'wb') as f:
                pickle.dump(failed_glacs, f)
                print(f'List of failed glaciers for {gcm_name}, {scenario} exported to: {outpath}')
        else: 
            print(f'No glaciers failed from R{region}, for {gcm_name} {scenario}')