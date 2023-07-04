#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:57:06 2022

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

process_cs = True

rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}


if process_cs:
    
    glac_nos = ['6.00340']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/Output/simulations/'
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

    
    glac_name_dict = {'1.00570':'Gulkana',
                      '6.00340':'Solheimajokull'}
        

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
                
                ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                ds_stats_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
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
        if glac_no in ['1.00570']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = 1000, 2500
            y_major, y_minor = 500, 100
            thick_max = 700
            
        
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
        
        assert 1==0, 'here'
        
        # Save figure
        fig_fn = (glac_no + '_profile_' + str(endyear) + '_ssps.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)