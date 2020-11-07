# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    return {k: v.dropna() for k, v in CTG_features.drop(columns=extra_feature).apply(pd.to_numeric, errors='coerce').items()}

def nan2num_samp(CTG_features, extra_feature):
    """
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    nan_dict = pd.DataFrame({k: v for k, v in CTG_features.drop(columns=extra_feature).apply(pd.to_numeric, errors='coerce').items()})
    c_cdf = {}
    for key, value in nan_dict.items():
        prob_list = value.value_counts(normalize=True)
        clean_list = []
        for val in value.values:
            if np.isnan(val):
                clean_list.append(np.random.choice(list(prob_list.keys()), 1, p=list(prob_list.values))[0])
            else:
                clean_list.append(val)
        c_cdf[key] = clean_list
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    d_summary = {}
    for k, v in c_feat.items():
        d_summary[k] = {"min": v.min(), "Q1": v.quantile(q=0.25),
                        "median": v.median(), "Q3": v.quantile(q=0.75), "max": v.max()}
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for k,v in c_feat.items():
        IQR_1_5 = 1.5*(d_summary[k]["Q3"]-d_summary[k]["Q1"])
        c_no_outlier[k]=pd.Series([], dtype='float64')
        for ind,val in v.iteritems():
            if d_summary[k]["Q1"]-IQR_1_5< val <d_summary[k]["Q3"]+IQR_1_5:
                c_no_outlier[k] = c_no_outlier[k].add(pd.Series(val, index=[ind]),fill_value=0)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    return c_cdf[feature].where(lambda x: x < thresh).dropna()


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    feat1 = CTG_features[x]
    feat2 = CTG_features[y]
    mean1 = np.mean(feat1)
    mean2 = np.mean(feat2)
    min1 = feat1.min()
    min2 = feat2.min()
    max1 = feat1.max()
    max2 = feat2.max()

    if mode == 'standard':
        std1 = np.std(feat1)
        std2 = np.std(feat2)
        feat1 = (feat1 - mean1) / std1
        feat2 = (feat2 - mean2) / std2

    elif mode == 'MinMax':
        feat1 = (feat1 - min1) / (min1 - max1)
        feat2 = (feat2 - min2) / (min2 - max2)

    elif mode == 'mean':
        feat1 = (feat1 - mean1) / (min1 - max1)
        feat2 = (feat2 - mean2) / (min2 - max2)

    nsd_res = [feat1, feat2]

    if flag:
        plt.hist(feat1, bins=100)
        plt.hist(feat2, bins=100)
        plt.title(f"Mode: {mode}")
        plt.legend([x, y], loc='upper right')
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
