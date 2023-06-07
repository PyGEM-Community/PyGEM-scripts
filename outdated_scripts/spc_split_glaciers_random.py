"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
import os
# External libraries
import numpy as np
import pickle
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    n_batches (optional) : int
        number of nodes being used on the supercomputer
    ignore_regionname (optional) : int
        switch to ignore region name or not (1 ignore it, 0 use region)
    add_cal : int
        switch to add "Cal" to the batch filenames such that calibration and simulation can be run at same time
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by 
         regional variations)
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-n_batches', action='store', type=int, default=1,
                        help='number of nodes to split the glaciers amongst')
    parser.add_argument('-ignore_regionname', action='store', type=int, default=0,
                        help='switch to include the region name or not in the batch filenames')
    parser.add_argument('-add_cal', action='store', type=int, default=0,
                        help='switch to add "cal" to batch filenames')
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-startno', action='store', type=int, default=None,
                        help='starting number of rgi glaciers')
    parser.add_argument('-endno', action='store', type=int, default=None,
                        help='starting number of rgi glaciers')
    parser.add_argument('-regno', action='store', type=int, default=None,
                        help='regno for rgi glaciers')
    return parser


def glac_num_fromrange(int_low, int_high):
    """
    Generate list of glaciers for all numbers between two integers.

    Parameters
    ----------
    int_low : int64
        low value of range
    int_high : int64
        high value of range

    Returns
    -------
    y : list
        list of rgi glacier numbers
    """
    x = (np.arange(int_low, int_high+1)).tolist()
    y = [str(i).zfill(5) for i in x]
    return y


def split_list(lst, n=1, option_ordered=1):
    """
    Split list into batches for the supercomputer.
    
    Parameters
    ----------
    lst : list
        List that you want to split into separate batches
    n : int
        Number of batches to split glaciers into.
    
    Returns
    -------
    lst_batches : list
        list of n lists that have sequential values in each list
    """
    # If batches is more than list, then there will be one glacier in each batch
    if option_ordered == 1:
        if n > len(lst):
            n = len(lst)
        n_perlist_low = int(len(lst)/n)
        n_perlist_high = int(np.ceil(len(lst)/n))
        lst_copy = lst.copy()
        count = 0
        lst_batches = []
        for x in np.arange(n):
            count += 1
            if count <= len(lst) % n:
                lst_subset = lst_copy[0:n_perlist_high]
                lst_batches.append(lst_subset)
                [lst_copy.remove(i) for i in lst_subset]
            else:
                lst_subset = lst_copy[0:n_perlist_low]
                lst_batches.append(lst_subset)
                [lst_copy.remove(i) for i in lst_subset]
    
    else:
        if n > len(lst):
            n = len(lst)
    
        lst_batches = [[] for x in np.arange(n)]
        nbatch = 0
        for count, x in enumerate(lst):
            if count%n == 0:
                nbatch = 0
    
            lst_batches[nbatch].append(x)
            
            nbatch += 1
            
    return lst_batches    
 

if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()   
        
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#    regions = [11]
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=regions, rgi_regionsO2='all', rgi_glac_number='all',
                                                          include_landterm=True,
                                                          include_laketerm=True, 
                                                          include_tidewater=False)
    
    # Random indices
    rand_idxs_all = np.arange(main_glac_rgi_all.shape[0])
    np.random.shuffle(rand_idxs_all)
    
    nglaciers = 2000
    rand_idxs = rand_idxs_all[:nglaciers]
    glacno_str = [x.split('-')[1] for x in main_glac_rgi_all.RGIId.values[rand_idxs]]
    
    # Split list of glacier numbers
    rgi_glac_number_batches = split_list(glacno_str, n=args.n_batches, option_ordered=args.option_ordered)

    # Export new lists
    for n in range(len(rgi_glac_number_batches)):
            
        # add batch number and .pkl
        batch_fn = 'Cal_fullsim_batch_' + str(n) + '.pkl'
            
        print('Batch', n, ':\n', batch_fn, '\n')
        with open(batch_fn, 'wb') as f:
            pickle.dump(rgi_glac_number_batches[n], f)