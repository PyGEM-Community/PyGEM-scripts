"""Process glacier directories from OGGM by calling script"""

# Built-in libraries
import argparse
import os
import sys
import time
import multiprocessing
import numpy as np
import logging

# Local libraries
try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving

from oggm import cfg

# logger
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    glacno (optional) : str
        glacier number to run 

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    parser.add_argument('-glacno', action='store', type=str, default=None,
                        help='glacier number used for model run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    return parser

def main(glac_nos):
    '''

    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    None.

    '''
    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_nos)

    # loop through list
    for nglac, glacno in enumerate(glac_nos):
        # log
        logging.info(f'preprocess_load_gdirs on glacier: {glacno}')
        # get terminus type from main_glac_rgi table
        glac_termtype = main_glac_rgi.loc[nglac,'TermType']
        
        if not glac_termtype in [1,5] or not pygem_prms.include_calving:
            is_tidewater = False
        else:
            is_tidewater = True
        try:
            if not is_tidewater or not pygem_prms.include_calving:
                gdir = single_flowline_glacier_directory(glacno, logging_level=pygem_prms.logging_level)
                gdir.is_tidewater = False
            else:
                gdir = single_flowline_glacier_directory_with_calving(glacno, logging_level=pygem_prms.logging_level)
                gdir.is_tidewater = True
        except Exception as err:
            print('preprocess_load_gdirs error:\t' + str(err))
    
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    glacno = args.glacno
    if glacno is None:
        glacno = pygem_prms.glac_no
    else: 
        glacno = [glacno]
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno,
            rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
            rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
            include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
    glac_nos = list(main_glac_rgi['rgino_str'].values)
    ### parallel processing ###
    if args.num_simultaneous_processes > 1:
        num_simultaneous_processes = int(np.min([len(glac_nos), args.num_simultaneous_processes]))
    else:
        num_simultaneous_processes = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_nos, n=num_simultaneous_processes, group_thousands=True)
    # print(glac_no_lsts[0])
    # print(len(glac_nos), sum([len(l) for l in glac_no_lsts]))

    print(f'Processing in parallel with {num_simultaneous_processes} cores...')
    with multiprocessing.Pool(num_simultaneous_processes) as p:
        p.map(main, glac_no_lsts)

    print('Total processing time:', time.time()-time_start, 's')