#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:07:36 2022

@author: drounce
"""
import os

import numpy as np
import pandas as pd
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


wgms_fp = '/Users/drounce/Documents/HiMAT/WGMS/DOI-WGMS-FoG-2021-05/'
wgms_eee_fn = 'WGMS-FoG-2021-05-EEE-MASS-BALANCE-POINT.csv'
wgms_ee_fn = 'WGMS-FoG-2021-05-EE-MASS-BALANCE.csv'
wgms_e_fn = 'WGMS-FoG-2021-05-E-MASS-BALANCE-OVERVIEW.csv'
wgms_id_fn = 'WGMS-FoG-2021-05-AA-GLACIER-ID-LUT.csv'

#%%
# Load data 
wgms_e_df = pd.read_csv(wgms_fp + wgms_e_fn, encoding='unicode_escape')
wgms_ee_df_raw = pd.read_csv(wgms_fp + wgms_ee_fn, encoding='unicode_escape')
wgms_eee_df_raw = pd.read_csv(wgms_fp + wgms_eee_fn, encoding='unicode_escape')
wgms_id_df = pd.read_csv(wgms_fp + wgms_id_fn, encoding='unicode_escape')

# Map dictionary
wgms_id_dict = dict(zip(wgms_id_df.WGMS_ID, wgms_id_df.RGI_ID))
wgms_ee_df_raw['rgiid_raw'] = wgms_ee_df_raw.WGMS_ID.map(wgms_id_dict)
wgms_ee_df_raw = wgms_ee_df_raw.dropna(subset=['rgiid_raw'])
wgms_eee_df_raw['rgiid_raw'] = wgms_eee_df_raw.WGMS_ID.map(wgms_id_dict)
wgms_eee_df_raw = wgms_eee_df_raw.dropna(subset=['rgiid_raw'])

# Link RGIv5.0 with RGIv6.0
rgi60_fp = pygem_prms.main_directory +  '/../RGI/rgi60/00_rgi60_attribs/'
rgi50_fp = pygem_prms.main_directory +  '/../RGI/00_rgi50_attribs/'

# Process each region
regions_str = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
rgi60_df = None
rgi50_df = None
for reg_str in regions_str:
    # RGI60 data
    for i in os.listdir(rgi60_fp):
        if i.startswith(reg_str) and i.endswith('.csv'):
            rgi60_df_reg = pd.read_csv(rgi60_fp + i, encoding='unicode_escape')
    # append datasets
    if rgi60_df is None:
        rgi60_df = rgi60_df_reg
    else:
        rgi60_df = pd.concat([rgi60_df, rgi60_df_reg], axis=0)

    # RGI50 data
    for i in os.listdir(rgi50_fp):
        if i.startswith(reg_str) and i.endswith('.csv'):
            rgi50_df_reg = pd.read_csv(rgi50_fp + i, encoding='unicode_escape')
    # append datasets
    if rgi50_df is None:
        rgi50_df = rgi50_df_reg
    else:
        rgi50_df = pd.concat([rgi50_df, rgi50_df_reg], axis=0)
    
# Merge based on GLIMSID
glims_rgi50_dict = dict(zip(rgi50_df.GLIMSId, rgi50_df.RGIId))
rgi60_df['RGIId_50'] = rgi60_df.GLIMSId.map(glims_rgi50_dict)
rgi60_df_4dict = rgi60_df.dropna(subset=['RGIId_50'])
rgi50_rgi60_dict = dict(zip(rgi60_df_4dict.RGIId_50, rgi60_df_4dict.RGIId))
rgi60_self_dict = dict(zip(rgi60_df.RGIId, rgi60_df.RGIId))
rgi50_rgi60_dict.update(rgi60_self_dict)

# Add RGIId for version 6 to WGMS
wgms_ee_df_raw['rgiid'] = wgms_ee_df_raw.rgiid_raw.map(rgi50_rgi60_dict)
wgms_eee_df_raw['rgiid'] = wgms_eee_df_raw.rgiid_raw.map(rgi50_rgi60_dict)

# Drop points without data
wgms_ee_df = wgms_ee_df_raw.dropna(subset=['rgiid'])
wgms_eee_df = wgms_eee_df_raw.dropna(subset=['rgiid']).copy()

assert 1==0, 'here'

#%%
# Load glaciers
rgiids_unique = list(wgms_eee_df['rgiid'].unique())
glac_no = [x.split('-')[1] for x in rgiids_unique]

main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    
#%% Number of measurements in accumulation area
rgi60_zmed_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Zmed))
wgms_eee_df['Zmed'] = wgms_eee_df.rgiid.map(rgi60_zmed_dict)

wgms_eee_df_annual = wgms_eee_df.loc[wgms_eee_df['BALANCE_CODE'] == 'BA']
wgms_eee_df_annual_acc = wgms_eee_df_annual.loc[wgms_eee_df_annual['POINT_BALANCE'] > 0]
wgms_eee_df_annual_abl = wgms_eee_df_annual.loc[wgms_eee_df_annual['POINT_BALANCE'] < 0]

wgms_eee_df_summer = wgms_eee_df.loc[wgms_eee_df['BALANCE_CODE'] == 'BS']
wgms_eee_df_winter = wgms_eee_df.loc[wgms_eee_df['BALANCE_CODE'] == 'BW']

wgms_eee_df_acc = wgms_eee_df.loc[wgms_eee_df['POINT_ELEVATION'] > wgms_eee_df['Zmed']]

print('Accumulation points (based on zmed):', wgms_eee_df_acc.shape[0], 
      '\nTotal data points:', wgms_eee_df.shape[0], 
      '\nFraction in accumulation:', wgms_eee_df_acc.shape[0]/wgms_eee_df.shape[0])


#%% Arctic versus other regions
wgms_eee_df['region'] = [int(x.split('-')[1].split('.')[0]) for x in wgms_eee_df.rgiid.values]

arctic_count = 0
for reg in [1,3,4,5,19]:
    wgms_eee_df_reg = wgms_eee_df.loc[wgms_eee_df['region'] == reg]
    arctic_count += wgms_eee_df_reg.shape[0]

print(arctic_count / wgms_eee_df.shape[0], 'from arctic')










