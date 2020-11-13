import os
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests

import nibabel as nib 
import nilearn.plotting as plotting
from nilearn import datasets
from nilearn import surface

def euler_effect_on_mf(mf_df, roi_ids): 
    euler_effect_mf_df = pd.DataFrame(index=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ C(SEX) + age + euler_mean_bh".format(roi),
                            data=mf_df).fit()
        t = fitmod.tvalues[3]
        p = fitmod.pvalues[3]
        euler_effect_mf_df.loc[roi,'euler_p'] = p
        euler_effect_mf_df.loc[roi,'euler_t'] = t
    
    fdrs = multipletests(euler_effect_mf_df['euler_p'].values,method='fdr_bh')
    euler_effect_mf_df.loc[:,'euler_FDR'] = fdrs[1]
    
    return euler_effect_mf_df

def scd_effect_on_mf(mf_df, roi_ids, include_euler=False):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic variables, 
    this function computes the effect of sex chromosome dosage on the morphometric 
    feature of interest at each ROI. Additionally, correction for multiple comparison
    is done, and a dataframe with t & p values for each ROI is returned.
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    scd_effect_mf_df = pd.DataFrame(index=roi_ids)
    for roi in roi_ids:
        if include_euler == True:
            fitmod = smf.ols("Q('{0}') ~ age + SCdose + euler_mean_bh".format(roi),
                             data=mf_df).fit()
        else:
            fitmod = smf.ols("Q('{0}') ~ age + SCdose".format(roi),
                             data=mf_df).fit()
        t = fitmod.tvalues[2]
        p = fitmod.pvalues[2]
        scd_effect_mf_df.loc[roi,'SCdose_p'] = p
        scd_effect_mf_df.loc[roi,'SCdose_t'] = t
    
    fdrs = multipletests(scd_effect_mf_df['SCdose_p'].values,method='fdr_bh')
    scd_effect_mf_df.loc[:,'SCdose_FDR'] = fdrs[1]
    
    return scd_effect_mf_df
    
def compute_mf_norm_corr_matrix(mf_df, roi_ids):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic variables, 
    this function first regresses out the effect of age, sex, and age*sex on the feature of interest. 
    It subsequently computes a correlation matrix for that feature only in individuals
    with a SC dosage of 2 (either XX or XY).
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    mf_norm_sex_age_residuals = pd.DataFrame(index=mf_df[mf_df.SCdose==2].index, columns=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ C(SEX) + age + C(SEX) * age".format(roi),
                         data=mf_df[mf_df.SCdose==2]).fit()
        mf_norm_sex_age_residuals.loc[:,roi] = fitmod.resid
        
    mf_norm_corr_matrix = mf_norm_sex_age_residuals.corr() # rmap_a
    return mf_norm_corr_matrix

def compute_rmap_b_mf(mf_norm_corr_matrix, scd_effect_mf_df, roi_ids):
    r_map_b_mf = []
    for i, roi in enumerate(roi_ids): 
        r,p = stats.pearsonr(mf_norm_corr_matrix[roi], scd_effect_mf_df['SCdose_t'])
        r_map_b_mf.append(r)
    return r_map_b_mf
        
def compute_scd_effect_global_mf_coupling(mf_df, roi_ids):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic 
    variables, this function first regresses out the effect of SC dosage, age, 
    and age*SC dosage on the morphometric feature of interest. For each subject, 
    the average cortical value for the morphometric feature is returned. 
    It then computes the effect of SC dosage on global anatomical coupling and
    corrects for multiple comparison. A dataframe with t & p values for each ROI 
    is returned.
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    mf_norm_age_scd_resid_df = pd.DataFrame(index=mf_df.index, columns=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ age + SCdose + age * SCdose".format(roi),
                         data=mf_df).fit()
        mf_norm_age_scd_resid_df.loc[:,roi] = fitmod.resid
        
    for var in ['age', 'SEX', 'SCdose']: 
        mf_norm_age_scd_resid_df[var] = mf_df[var]
        
    for sub in mf_norm_age_scd_resid_df.index: 
        avg_mf = np.mean(mf_norm_age_scd_resid_df.loc[sub, roi_ids])
        mf_norm_age_scd_resid_df.loc[sub, 'Avg_MF'] = avg_mf
        
    # examine SCD effects on the global anatomical coupling of each cortical region
    scd_global_anat_coupling_df = pd.DataFrame(index=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ Avg_MF + SCdose + Avg_MF * SCdose".format(roi),
                         data=mf_norm_age_scd_resid_df).fit()
        t = fitmod.tvalues[3]
        p = fitmod.pvalues[3]
        scd_global_anat_coupling_df.loc[roi,'SCdose_global_p'] = p
        scd_global_anat_coupling_df.loc[roi,'SCdose_global_t'] = t
    
    fdrs = multipletests(scd_global_anat_coupling_df['SCdose_global_p'].values,method='fdr_bh')
    scd_global_anat_coupling_df.loc[:,'SCdose_global_FDR'] = fdrs[1]

    return mf_norm_age_scd_resid_df, scd_global_anat_coupling_df
    
def compute_scd_mf_coupling_effect(mf_norm_age_scd_resid_df, roi_ids):
    scd_mf_coupling_effect_df = pd.DataFrame(index=roi_ids, columns=roi_ids)
    for i, roi in enumerate(roi_ids): 
        for j, roi2 in enumerate(roi_ids): 
            fitmod = smf.ols("Q('{0}') ~ Q('{1}') + SCdose + Q('{1}') * SCdose".format(roi, roi2),
                         data=mf_norm_age_scd_resid_df).fit()
            t = fitmod.tvalues[3]
            p = fitmod.pvalues[3]
            scd_mf_coupling_effect_df.loc[roi, roi2] = t
            
    scd_mf_coupling_effect_sym_df = pd.DataFrame(index=roi_ids, columns=roi_ids, dtype="float32")
    for i, roi in enumerate(roi_ids): 
        for j, roi2 in enumerate(roi_ids): 
            val1 = scd_mf_coupling_effect_df.loc[roi, roi2]
            val2 = scd_mf_coupling_effect_df.loc[roi2, roi]
            avg = np.mean([val1, val2])
            scd_mf_coupling_effect_sym_df.loc[roi, roi2] = avg
            scd_mf_coupling_effect_sym_df.loc[roi2, roi] = avg
            
    return scd_mf_coupling_effect_sym_df
    
    
def compute_r_map_c_mf(scd_mf_coupling_effect_sym_df, scd_effect_mf_df, roi_ids):
    r_map_c_mf = []
    for i, roi in enumerate(roi_ids):
        r,p = stats.pearsonr(scd_mf_coupling_effect_sym_df[roi], scd_effect_mf_df['SCdose_t'])
        r_map_c_mf.append(r)
    return r_map_c_mf
    
def create_surf_stat_img(stat_data, parc308_img):
        
    stat_img_data = np.zeros_like(parc308_img.get_fdata())
    for i,roi in enumerate(roi_ids): 
        stat_img_data[parc308_data==(i+41)] = stat_data[i]
    
    stat_img = nib.Nifti1Image(stat_img_data, 
                               affine=parc308_img.affine, 
                               header=parc308_img.header)
    return stat_img
    
def plot_corr_matrix(corr_matrix_df, xlabel="308 Regions", ylabel="308 Regions", title="Coupling Change Matrix", cmap="RdBu_r"):
    plt.figure(figsize=(6,5))
    sns.heatmap(np.array(corr_matrix_df), cmap=cmap, vmax=4, vmin=-4)
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.xlabel(xlabel, fontsize=14)
    plt.yticks([])
    plt.ylabel(ylabel, fontsize=14)
