""" Analyze simulation output - mass change, runoff, etc. """

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
import zipfile
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
# Script options
option_glacier_cs_plots_calving_bydeg = False    # Cross sectional plots for calving glaciers based on degrees
process_cs = True                               # Cross sectional plots by SSP
option_unzip_binned_data = False                 # Unzip data

#regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#regions = [1,3,4,5,7,9,17,19]
regions = [2]

#deg_groups = [1.5,2,2.7,3,4]
#deg_groups_bnds = [0.25, 0.5, 0.5, 0.5, 0.5]
#deg_group_colors = ['#4575b4','#74add1', '#fee090', '#fdae61', '#f46d43', '#d73027']
deg_groups = [1.5,2,3,4]
deg_groups_bnds = [0.25, 0.5, 0.5, 0.5]
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

netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v4/'
netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v6/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-runoff_fixed/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-nodebris/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_ssp119/'

temp_dev_fn = 'Global_mean_temp_deviation_2081_2100_rel_1850_1900.csv'

ds_marzeion2020_fn = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving_v3/Marzeion_etal_2020_results.nc'

analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
csv_fp = analysis_fp + '/csv/'
csv_fp_glacind = netcdf_fp_cmip5 + '_csv/'
csv_fp_glacind_land = netcdf_fp_cmip5_land + '_csv/'
pickle_fp = analysis_fp + '/pickle/'

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
# Colors list
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
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

if process_cs:
    
#    glac_nos = ['6.00340']
#    glac_nos = ['6.00483']
#    glac_nos = ['6.00471']
#    glac_nos = ['6.00478']
#    glac_nos = ['6.00465']
    glac_nos = ['2.14256', '2.14259', '2.14277',
       '2.14293', '2.14297', '2.14300',
       '2.14303', '2.14307', '2.14308',
       '2.14315', '2.14323', '2.14329',
       '2.14333', '2.14335', '2.14336',
       '2.14341', '2.14344', '2.14352',
       '2.14359', '2.14360', '2.14369',
       '2.14371', '2.18817', '2.18818']
#    glac_nos = ['2.14297']
    
    glac_name_dict = {'2.14297':'Emmons Glacier',
                      '2.14336':'Nisqually Glacier',
                      '2.14259':'Winthrop Glacier',
                      '2.14256':'Carbon Glacier',
                      '6.00340':'Sólheimajökull',
                      '6.00483':'Breiðamerkurjökull',
                      '6.00471':'Heinabergsjökull',
                      '6.00478':'Skálafellsjökull',
                      '6.00465':'Fláajökull',
                      '6.00026':'Tungnahryggsjökull (RGI60-06.00026)'}  
    
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_02/' # treated as clean ice
    fig_fp = netcdf_fp_cmip5 + 'figures/'

    fig_fp_ind = fig_fp + 'ind_glaciers/'
    if not os.path.exists(fig_fp_ind):
        os.makedirs(fig_fp_ind)

    cs_year = 2000
    vol_norm_endyear = 2100
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    startyear_idx = np.where(years == startyear)[0][0]
    cs_idx = np.where(years == cs_year)[0][0]  
        

    # Set up processing
    glac_zbed_all = {}
    glac_thick_all = {}
    glac_zsurf_all = {}
    glac_vol_all = {}
    for glac_no in glac_nos:
        
        print('\n\n', glac_no)

        gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
        
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        nfls = gdir.read_pickle('model_flowlines')
        
        
        x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx
        
        glac_idx = np.nonzero(nfls[0].thick)[0]
        xmax = np.ceil(x[glac_idx].max()/1000+0.5)*1000
                                
        glac_zbed_all[glac_no] = {}
        glac_thick_all[glac_no] = {}
        glac_zsurf_all[glac_no] = {}
        glac_vol_all[glac_no] = {}
        
        for rcp in rcps:
            
            glac_zbed_all[glac_no][rcp] = None
            glac_thick_all[glac_no][rcp] = None
            glac_zsurf_all[glac_no][rcp] = None
            glac_vol_all[glac_no][rcp] = None
            
            for gcm_name in gcm_names:
                
                ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/binned/' + gcm_name + '_' + rcp + '_binned/' 
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                ds_stats_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/stats/' + gcm_name + '_' + rcp + '_stats/' 
                for i in os.listdir(ds_stats_fp):
                    if i.startswith(glac_no):
                        ds_stats_fn = i
                
                if glac_no in ds_stats_fn and rcp in ds_stats_fn and gcm_name in ds_stats_fn:
                    ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)
                    ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)
    
                    thick = ds_binned.bin_thick_annual[0,:,:].values
                    zsurf_init = ds_binned.bin_surface_h_initial[0].values
                    zbed = zsurf_init - thick[:,cs_idx]
                    vol = ds_stats.glac_volume_annual[0,:].values
                    
                    if glac_zbed_all[glac_no][rcp] is None:
                        glac_thick_all[glac_no][rcp] = thick[np.newaxis,:,:]
                        glac_zbed_all[glac_no][rcp] = zbed[np.newaxis,:]
                        glac_zsurf_all[glac_no][rcp] = (zbed[:,np.newaxis] + thick)[np.newaxis,:,:]
                        glac_vol_all[glac_no][rcp] = vol[np.newaxis,:]

                    else:
                        glac_thick_all[glac_no][rcp] = np.concatenate((glac_thick_all[glac_no][rcp], thick[np.newaxis,:,:]), axis=0)
                        glac_zbed_all[glac_no][rcp] = np.concatenate((glac_zbed_all[glac_no][rcp], zbed[np.newaxis,:]), axis=0)
                        glac_zsurf_all[glac_no][rcp] = np.concatenate((glac_zsurf_all[glac_no][rcp], (zbed[:,np.newaxis] + thick)[np.newaxis,:,:]), axis=0)
                        glac_vol_all[glac_no][rcp] = np.concatenate((glac_vol_all[glac_no][rcp], vol[np.newaxis,:]), axis=0)
         
        #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,0.65])
        ax.patch.set_facecolor('none')
#        ax2 = fig.add_axes([0,0.67,1,0.35])
#        ax2.patch.set_facecolor('none')
        ax3 = fig.add_axes([0.67,0.32,0.3,0.3])
        ax3.patch.set_facecolor('none')
        
        normyear = 2000
        
        ymin, ymax, thick_max = None, None, None
        vol_med_all = []
        for nrcp, rcp in enumerate(rcps):
            zbed_med = np.median(glac_zbed_all[glac_no][rcp],axis=0)
            zbed_std = np.std(glac_zbed_all[glac_no][rcp], axis=0)
            
            thick_med = np.median(glac_thick_all[glac_no][rcp],axis=0)
            thick_std = np.std(glac_thick_all[glac_no][rcp], axis=0)
            
            zsurf_med = np.median(glac_zsurf_all[glac_no][rcp],axis=0)
            zsurf_std = np.std(glac_zsurf_all[glac_no][rcp], axis=0)
            
            vol_med = np.median(glac_vol_all[glac_no][rcp],axis=0)
            vol_std = np.std(glac_vol_all[glac_no][rcp], axis=0)
            
            normyear_idx = np.where(years == normyear)[0][0]
            endyear_idx = np.where(years == endyear)[0][0]
            
            if rcp in ['ssp585']:
                ax.fill_between(x[1:]/1000, zbed_med[1:]-20, zbed_med[1:], color='white', zorder=5+len(rcps))
                ax.plot(x/1000, zbed_med[np.arange(len(x))],
                        color='k', linestyle='-', linewidth=1, zorder=5+len(rcps), label='zbed')
                ax.plot(x/1000, zsurf_med[np.arange(len(x)),normyear_idx], 
                             color='k', linestyle=':', linewidth=0.5, zorder=4+len(rcps), label=str(normyear))
#                ax2.plot(x/1000, thick_med[np.arange(len(x)),normyear_idx], 
#                         color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
                zbed_last = zbed_med
                add_zbed = False
                
            ax.plot(x/1000, zsurf_med[np.arange(len(x)),endyear_idx], 
                         color=rcp_colordict[rcp], linestyle='-', linewidth=0.5, zorder=4+(len(rcps)-nrcp), label=str(endyear))
            
#            ax2.plot(x/1000, thick_med[np.arange(len(x)),endyear_idx],
#                     color=temp_colordict[deg_group], linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
            
            ax3.plot(years, vol_med / vol_med[normyear_idx], color=rcp_colordict[rcp], 
                     linewidth=0.5, zorder=4, label=None)

            if rcp in ['ssp245', 'ssp585']:
                ax3.fill_between(years, 
                                 (vol_med + vol_std)/vol_med[normyear_idx], 
                                 (vol_med - vol_std)/vol_med[normyear_idx],
                                 alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
                
            # Record median value for printing
            vol_med_all.append(vol_med[normyear_idx])
            
            # ymin and ymax for bounds
            if ymin is None:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
                ymax = np.ceil(zsurf_med[:,endyear_idx].max()/100)*100
            if np.floor(zbed_med.min()/100)*100 < ymin:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
            if np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100 > ymax:
                ymax = np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100
            # thickness max for bounds  
            if thick_max is None:
                thick_max = np.ceil(thick_med.max()/10)*10
            if np.ceil(thick_med.max()/10)*10 > thick_max:
                thick_max = np.ceil(thick_med.max()/10)*10
        
        if ymin < 0:
            water_idx = np.where(zbed_med < 0)[0]
            # Add water level
            ax.plot(x[water_idx]/1000, np.zeros(x[water_idx].shape), color='aquamarine', linewidth=1)
        
        if xmax/1000 > 25:
            x_major, x_minor = 10, 2
        elif xmax/1000 > 15:
            x_major, x_minor = 5, 1
        else:
            x_major, x_minor = 2, 0.5
        
        y_major, y_minor = 500,100
        
        if thick_max > 200:
            thick_major, thick_minor = 100, 20
        else:
            thick_major, thick_minor = 50, 10
            
            
        # ----- GLACIER SPECIFIC PLOTS -----
        plot_legend = True
        add_glac_name = True
        plot_sealevel = True
        leg_label = None
        if glac_no in ['6.00340']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -50, 1800
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00483']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -400, 2490
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00471']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -400, 1800
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00478']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -200, 2100
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00465']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -300, 2100
            y_major, y_minor = 500, 100
            plot_legend = True
            
        
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0,xmax/1000)
#        ax2.set_xlim(0,xmax/1000)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor)) 
#        ax2.set_ylim(0,thick_max)
#        ax2.yaxis.set_major_locator(MultipleLocator(thick_major))
##        ax2.yaxis.set_minor_locator(MultipleLocator(thick_minor))
#        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
#        ax2.get_xaxis().set_visible(False)
            
        ax.set_ylabel('Elevation (m a.s.l.)')
        ax.set_xlabel('Distance along flowline (km)')
#        ax2.set_ylabel('Ice thickness (m)', labelpad=10)
##        ax2.yaxis.set_label_position('right')
##        ax2.yaxis.tick_right()
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax.spines['top'].set_visible(False)
                
        if glac_no in glac_name_dict.keys():
            glac_name_text = glac_name_dict[glac_no]
        else:
             glac_name_text = glac_no
        
        if add_glac_name:
            ax.text(0.98, 1.02, glac_name_text, size=10, horizontalalignment='right', 
                    verticalalignment='bottom', transform=ax.transAxes)
        
        if not leg_label is None:
            ax.text(0.02, 0.98, leg_label, weight='bold', size=10, horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes)    
        
        ax3.set_ylabel('Mass (-)')
        ax3.set_xlim(normyear, vol_norm_endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
        ax3.set_ylim(0,1.1)
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax3.tick_params(axis='both', which='major', direction='inout', right=True)
        ax3.tick_params(axis='both', which='minor', direction='in', right=True)
        vol_norm_gt = np.median(vol_med_all) * pygem_prms.density_ice / 1e12
        
        if vol_norm_endyear > endyear:
            ax3.axvline(endyear, color='k', linewidth=0.5, linestyle='--', zorder=4)
        
        if vol_norm_gt > 10:
            vol_norm_gt_str = str(int(np.round(vol_norm_gt,0))) + ' Gt'
        elif vol_norm_gt > 1:
            vol_norm_gt_str = str(np.round(vol_norm_gt,1)) + ' Gt'
        else:
            vol_norm_gt_str = str(np.round(vol_norm_gt,2)) + ' Gt'
        ax3.text(0.95, 0.95, vol_norm_gt_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax3.transAxes)
        
        # Legend
        if plot_legend:
            leg_lines = []
            leg_labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
            for nrcp, rcp in enumerate(rcps):
                line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=1)
                leg_lines.append(line)
            
            # add years
            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=1)
            leg_lines.append(line)
            leg_labels.append(str(normyear))
            line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
            leg_lines.append(line)
            leg_labels.append(str(endyear))
            if plot_sealevel:
                line = Line2D([0,1],[0,1], color='aquamarine', linewidth=1)
                leg_lines.append(line)
                leg_labels.append('Sea level')
            line = Line2D([0,1],[0,1], color='k', linewidth=1)
            leg_lines.append(line)
            leg_labels.append('Bed')
            
            ax.legend(leg_lines, leg_labels, loc=(0.02,0.02), fontsize=8, labelspacing=0.25, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.5, frameon=False)
        
        # Save figure
        fig_fn = (glac_no + '_profile_' + str(endyear) + '_ssps.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)
        
#        assert 1==0, 'here'

        
#%%
if option_glacier_cs_plots_calving_bydeg:
#    glac_nos = ['6.00340']
#    glac_nos = ['6.00483']
#    glac_nos = ['6.00471']
#    glac_nos = ['6.00478']
#    glac_nos = ['6.00465']
    glac_nos = ['18.02397']
    
    glac_name_dict = {'6.00340':'Sólheimajökull',
                      '6.00483':'Breiðamerkurjökull',
                      '6.00471':'Heinabergsjökull',
                      '6.00478':'Skálafellsjökull',
                      '6.00465':'Fláajökull',
                      '18.02397':'Franz Josef'} 

#    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-tolly/' # treated as clean ice
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind/' # treated as clean ice
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-nz/' # treated as clean ice
    fig_fp = netcdf_fp_cmip5 + 'figures/'
    
    fig_fp_ind = fig_fp + 'ind_glaciers/'
    if not os.path.exists(fig_fp_ind):
        os.makedirs(fig_fp_ind)

    cs_year = 2000
    vol_norm_endyear = 2100
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    startyear_idx = np.where(years == startyear)[0][0]
    cs_idx = np.where(years == cs_year)[0][0]
        
    # MULTI-GCM STATISTICS by degree
    # Set up temps
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)  

    # Set up processing
    glac_zbed_all = {}
    glac_thick_all = {}
    glac_zsurf_all = {}
    glac_vol_all = {}
    glac_multigcm_zbed = {}
    glac_multigcm_thick = {}
    glac_multigcm_zsurf = {}
    glac_multigcm_vol = {}
    for glac_no in glac_nos:
        
        print('\n\n', glac_no)

        gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
        
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        nfls = gdir.read_pickle('model_flowlines')
        
#        tasks.init_present_time_glacier(gdir) # adds bins below
#        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
#
#        try:
#            nfls = gdir.read_pickle('model_flowlines')
#        except FileNotFoundError as e:
#            if 'model_flowlines.pkl' in str(e):
#                tasks.compute_downstream_line(gdir)
#                tasks.compute_downstream_bedshape(gdir)
#                tasks.init_present_time_glacier(gdir) # adds bins below
#                nfls = gdir.read_pickle('model_flowlines')
#            else:
#                raise
        
        
        
        x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx
        
        glac_idx = np.nonzero(nfls[0].thick)[0]
        xmax = np.ceil(x[glac_idx].max()/1000+0.5)*1000
        
#        vol_m3_init = ds_binned.bin_volume_annual[0,:,0].values
#        thick_init = ds_binned.bin_thick_annual[0,:,0].values
#        widths_m = nfls[0].widths_m
#        lengths_m = vol_m3_init / thick_init / widths_m
                                
                                
        glac_zbed_all[glac_no] = {}
        glac_thick_all[glac_no] = {}
        glac_zsurf_all[glac_no] = {}
        glac_vol_all[glac_no] = {}
        
        for rcp in rcps:
            
            glac_zbed_all[glac_no][rcp] = {}
            glac_thick_all[glac_no][rcp] = {}
            glac_zsurf_all[glac_no][rcp] = {}
            glac_vol_all[glac_no][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/binned/' + gcm_name + '_' + rcp + '_binned/' 
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                ds_stats_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/stats/' + gcm_name + '_' + rcp + '_stats/' 
                for i in os.listdir(ds_stats_fp):
                    if i.startswith(glac_no):
                        ds_stats_fn = i
                
                if glac_no in ds_stats_fn and rcp in ds_stats_fn and gcm_name in ds_stats_fn:
                    ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)
                    ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)
    
                    thick = ds_binned.bin_thick_annual[0,:,:].values
                    zsurf_init = ds_binned.bin_surface_h_initial[0].values
                    zbed = zsurf_init - thick[:,cs_idx]
                    vol = ds_stats.glac_volume_annual[0,:].values
                    
                    glac_thick_all[glac_no][rcp][gcm_name] = thick
                    glac_zbed_all[glac_no][rcp][gcm_name] = zbed
                    glac_zsurf_all[glac_no][rcp][gcm_name] = zbed[:,np.newaxis] + thick
                    glac_vol_all[glac_no][rcp][gcm_name] = vol

                else:
                    glac_thick_all[glac_no][rcp][gcm_name] = None
                    glac_zbed_all[glac_no][rcp][gcm_name] = None
                    glac_zsurf_all[glac_no][rcp][gcm_name] = None
                    glac_vol_all[glac_no][rcp][gcm_name] = None
                    
            #%%
        # Set up regions
        glac_multigcm_zbed[glac_no] = {}
        glac_multigcm_thick[glac_no] = {}
        glac_multigcm_zsurf[glac_no] = {}
        glac_multigcm_vol[glac_no] = {}
         
        for deg_group in deg_groups:
            
            gcm_rcps_list = temp_gcmrcp_dict[deg_group]
            
            ngcm = 0
            for rcp_gcm_name in gcm_rcps_list:
                
#                print('\n', glac_no, deg_group, rcp_gcm_name)
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                
                if rcp in rcps:
                
                    glac_zbed_gcm = glac_zbed_all[glac_no][rcp][gcm_name]
                    glac_thick_gcm = glac_thick_all[glac_no][rcp][gcm_name]
                    glac_zsurf_gcm = glac_zsurf_all[glac_no][rcp][gcm_name]
                    glac_vol_gcm = glac_vol_all[glac_no][rcp][gcm_name]
    
                    if not glac_vol_gcm is None:
                        if ngcm == 0:
                            glac_zbed_gcm_all = glac_zbed_gcm 
                            glac_thick_gcm_all = glac_thick_gcm[np.newaxis,:,:]
                            glac_zsurf_gcm_all = glac_zsurf_gcm[np.newaxis,:,:]
                            glac_vol_gcm_all = glac_vol_gcm[np.newaxis,:]
                        else:
                            glac_zbed_gcm_all = np.vstack((glac_zbed_gcm_all, glac_zbed_gcm))
                            glac_thick_gcm_all = np.vstack((glac_thick_gcm_all, glac_thick_gcm[np.newaxis,:,:]))
                            glac_zsurf_gcm_all = np.vstack((glac_zsurf_gcm_all, glac_zsurf_gcm[np.newaxis,:,:]))
                            glac_vol_gcm_all = np.vstack((glac_vol_gcm_all, glac_vol_gcm[np.newaxis,:]))
                        ngcm += 1

            glac_multigcm_zbed[glac_no][deg_group] = glac_zbed_gcm_all
            glac_multigcm_thick[glac_no][deg_group] = glac_thick_gcm_all
            glac_multigcm_zsurf[glac_no][deg_group] = glac_zsurf_gcm_all
            glac_multigcm_vol[glac_no][deg_group] = glac_vol_gcm_all
                
        #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        temps_plot_mad = [1.5,4]
        zbed_deg_group = [deg_groups[0]]
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,0.65])
        ax.patch.set_facecolor('none')
#        ax2 = fig.add_axes([0,0.67,1,0.35])
#        ax2.patch.set_facecolor('none')
        ax3 = fig.add_axes([0.67,0.32,0.3,0.3])
        ax3.patch.set_facecolor('none')
        
        ymin, ymax, thick_max = None, None, None
        vol_med_all = []
        for ngroup, deg_group in enumerate(deg_groups):
            zbed_med = np.median(glac_multigcm_zbed[glac_no][deg_group],axis=0)
            zbed_std = np.std(glac_multigcm_zbed[glac_no][deg_group], axis=0)
            
            thick_med = np.median(glac_multigcm_thick[glac_no][deg_group],axis=0)
            thick_std = np.std(glac_multigcm_thick[glac_no][deg_group], axis=0)
            
            zsurf_med = np.median(glac_multigcm_zsurf[glac_no][deg_group],axis=0)
            zsurf_std = np.std(glac_multigcm_zsurf[glac_no][deg_group], axis=0)
            
            vol_med = np.median(glac_multigcm_vol[glac_no][deg_group],axis=0)
            vol_std = np.std(glac_multigcm_vol[glac_no][deg_group], axis=0)
            
            normyear_idx = np.where(years == normyear)[0][0]
            endyear_idx = np.where(years == endyear)[0][0]
            
            if deg_group in zbed_deg_group:
                ax.fill_between(x[1:]/1000, zbed_med[1:]-20, zbed_med[1:], color='white', zorder=5+len(deg_groups))
                ax.plot(x/1000, zbed_med[np.arange(len(x))],
                        color='k', linestyle='-', linewidth=1, zorder=5+len(deg_groups), label='zbed')
                ax.plot(x/1000, zsurf_med[np.arange(len(x)),normyear_idx], 
                             color='k', linestyle=':', linewidth=0.5, zorder=4+len(deg_groups), label=str(normyear))
#                ax2.plot(x/1000, thick_med[np.arange(len(x)),normyear_idx], 
#                         color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
                zbed_last = zbed_med
                add_zbed = False
                
            ax.plot(x/1000, zsurf_med[np.arange(len(x)),endyear_idx], 
                         color=temp_colordict[deg_group], linestyle='-', linewidth=0.5, zorder=4+(len(deg_groups)-ngroup), label=str(endyear))
            
#            ax2.plot(x/1000, thick_med[np.arange(len(x)),endyear_idx],
#                     color=temp_colordict[deg_group], linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
            
            ax3.plot(years, vol_med / vol_med[normyear_idx], color=temp_colordict[deg_group], 
                     linewidth=0.5, zorder=4, label=None)

            if deg_group in temps_plot_mad:
                ax3.fill_between(years, 
                                 (vol_med + vol_std)/vol_med[normyear_idx], 
                                 (vol_med - vol_std)/vol_med[normyear_idx],
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
                
            # Record median value for printing
            vol_med_all.append(vol_med[normyear_idx])
            
            # ymin and ymax for bounds
            if ymin is None:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
                ymax = np.ceil(zsurf_med[:,endyear_idx].max()/100)*100
            if np.floor(zbed_med.min()/100)*100 < ymin:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
            if np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100 > ymax:
                ymax = np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100
            # thickness max for bounds  
            if thick_max is None:
                thick_max = np.ceil(thick_med.max()/10)*10
            if np.ceil(thick_med.max()/10)*10 > thick_max:
                thick_max = np.ceil(thick_med.max()/10)*10
        
        if ymin < 0:
            water_idx = np.where(zbed_med < 0)[0]
            # Add water level
            ax.plot(x[water_idx]/1000, np.zeros(x[water_idx].shape), color='aquamarine', linewidth=1)
        
        if xmax/1000 > 25:
            x_major, x_minor = 10, 2
        elif xmax/1000 > 15:
            x_major, x_minor = 5, 1
        else:
            x_major, x_minor = 2, 0.5
        
        y_major, y_minor = 500,100
        
        if thick_max > 200:
            thick_major, thick_minor = 100, 20
        else:
            thick_major, thick_minor = 50, 10
            
            
        # ----- GLACIER SPECIFIC PLOTS -----
        plot_legend = True
        add_glac_name = True
        plot_sealevel = True
        leg_label = None
        if glac_no in ['6.00340']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -50, 1800
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00483']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -400, 2490
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00471']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -400, 1800
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00478']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -200, 2100
            y_major, y_minor = 500, 100
            plot_legend = True
        elif glac_no in ['6.00465']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -300, 2100
            y_major, y_minor = 500, 100
            plot_legend = True
            
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0,xmax/1000)
#        ax2.set_xlim(0,xmax/1000)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor)) 
#        ax2.set_ylim(0,thick_max)
#        ax2.yaxis.set_major_locator(MultipleLocator(thick_major))
##        ax2.yaxis.set_minor_locator(MultipleLocator(thick_minor))
#        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
#        ax2.get_xaxis().set_visible(False)
            
        ax.set_ylabel('Elevation (m a.s.l.)')
        ax.set_xlabel('Distance along flowline (km)')
#        ax2.set_ylabel('Ice thickness (m)', labelpad=10)
##        ax2.yaxis.set_label_position('right')
##        ax2.yaxis.tick_right()
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax.spines['top'].set_visible(False)
                
        if glac_no in glac_name_dict.keys():
            glac_name_text = glac_name_dict[glac_no]
        else:
             glac_name_text = glac_no
        
        if add_glac_name:
            ax.text(0.98, 1.02, glac_name_text, size=10, horizontalalignment='right', 
                    verticalalignment='bottom', transform=ax.transAxes)
        
        if not leg_label is None:
            ax.text(0.02, 0.98, leg_label, weight='bold', size=10, horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes)    
        
        ax3.set_ylabel('Mass (-)')
        ax3.set_xlim(normyear, vol_norm_endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
        ax3.set_ylim(0,1.1)
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax3.tick_params(axis='both', which='major', direction='inout', right=True)
        ax3.tick_params(axis='both', which='minor', direction='in', right=True)
        vol_norm_gt = np.median(vol_med_all) * pygem_prms.density_ice / 1e12
        
        if vol_norm_endyear > endyear:
            ax3.axvline(endyear, color='k', linewidth=0.5, linestyle='--', zorder=4)
        
        if vol_norm_gt > 10:
            vol_norm_gt_str = str(int(np.round(vol_norm_gt,0))) + ' Gt'
        elif vol_norm_gt > 1:
            vol_norm_gt_str = str(np.round(vol_norm_gt,1)) + ' Gt'
        else:
            vol_norm_gt_str = str(np.round(vol_norm_gt,2)) + ' Gt'
        ax3.text(0.95, 0.95, vol_norm_gt_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax3.transAxes)
        
        # Legend
        if plot_legend:
            leg_lines = []
            leg_labels = []
            deg_labels = ['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C'
                          for x in np.arange(len(deg_groups))]
            for ngroup, deg_group in enumerate(deg_groups):
                line = Line2D([0,1],[0,1], color=temp_colordict[deg_group], linewidth=1)
                leg_lines.append(line)
                leg_labels.append(deg_labels[ngroup])
            
            # add years
#            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=0)
#            leg_lines.append(line)
#            leg_labels.append('')
            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=1)
            leg_lines.append(line)
            leg_labels.append(str(normyear))
            line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
            leg_lines.append(line)
            leg_labels.append(str(endyear))
            if plot_sealevel:
                line = Line2D([0,1],[0,1], color='aquamarine', linewidth=1)
                leg_lines.append(line)
                leg_labels.append('Sea level')
            line = Line2D([0,1],[0,1], color='k', linewidth=1)
            leg_lines.append(line)
            leg_labels.append('Bed')
            
            ax.legend(leg_lines, leg_labels, loc=(0.02,0.02), fontsize=8, labelspacing=0.25, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.5, frameon=False)
        
        # Save figure
        fig_fn = (glac_no + '_profile_' + str(endyear) + '_bydeg.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)
        
        
if option_unzip_binned_data:
    
    # Zipped filepath
    zipped_fp_base = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_02/'
    
    for reg in regions:
#        zipped_fp = zipped_fp_base + str(reg).zfill(2) + '/' + 'binned/'
#        for i in os.listdir(zipped_fp):
#            if i.endswith('.zip'):
#                unzipped_fp = zipped_fp + i.replace('.zip','/')
#                if not os.path.exists(unzipped_fp):
#                    os.makedirs(unzipped_fp)
#                with zipfile.ZipFile(zipped_fp + i, 'r') as zip_ref:
#                    zip_ref.extractall(unzipped_fp)
        zipped_fp = zipped_fp_base + str(reg).zfill(2) + '/' + 'stats/'
        for i in os.listdir(zipped_fp):
            if i.endswith('.zip'):
                unzipped_fp = zipped_fp + i.replace('.zip','/')
                if not os.path.exists(unzipped_fp):
                    os.makedirs(unzipped_fp)
                with zipfile.ZipFile(zipped_fp + i, 'r') as zip_ref:
                    zip_ref.extractall(unzipped_fp)