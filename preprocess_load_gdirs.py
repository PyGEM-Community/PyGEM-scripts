"""Process glacier directories from OGGM by calling script"""

# Built-in libraries
import argparse
import os
import sys

# Local libraries
try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving

from oggm import cfg


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
    return parser


def main(glacno, is_tidewater=False):
    '''

    Parameters
    ----------
    glacno : str
        glacier number as a string.
    is_tidewater : boolean, optional
        switch for if the glacier is a tidewater glacier or not. The default is False.

    Returns
    -------
    None.

    '''
    if not is_tidewater or not pygem_prms.include_calving:
        gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
        gdir.is_tidewater = False
    else:
        gdir = single_flowline_glacier_directory_with_calving(glacno, logging_level='CRITICAL')
        gdir.is_tidewater = True
        
    
if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=pygem_prms.glac_no,
            rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
            rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
            include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
    glac_nos = list(main_glac_rgi['rgino_str'].values)
    
    for nglac, glac_no in enumerate(glac_nos):
        print(glac_no)
        
        glac_termtype = main_glac_rgi.loc[nglac,'TermType']
        
        if not glac_termtype in [1,5] or not pygem_prms.include_calving:
            glac_tidewater = False
        else:
            glac_tidewater = True
        
        main(glac_no, is_tidewater=glac_tidewater)

