import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
import os
import dataframe_image as dfi
from pandas.plotting import table
import plotly.figure_factory as ff
import plotly.graph_objects as go
from CorrelationAnalysisFunctions import sort_metric_scores, compute_correlation, get_original_correlation_table




# Import test results
mean_scores_C = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_C.csv")
mean_scores_G = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_G.csv")
mean_scores_H = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_H.csv")
mean_scores_R = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_R.csv")



# Import Toussaint 
Toussaint_original_C = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_C.csv')  #original
Toussaint_original_G = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_G.csv')
Toussaint_original_H = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_H.csv')
Toussaint_original_R = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_R.csv')

# Import LHL
LonguetHigginsLee_original_C = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_C.csv')  #original
LonguetHigginsLee_original_G = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_G.csv')
LonguetHigginsLee_original_H = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_H.csv')
LonguetHigginsLee_original_R = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_R.csv')

# Import Pressing
Pressing_original_C = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_C.csv') #original
Pressing_original_G = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_G.csv')
Pressing_original_H = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_H.csv')
Pressing_original_R = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_R.csv')

# Import WNBD
WNBD_original_C = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_C.csv') #original
WNBD_original_G = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_G.csv')
WNBD_original_H = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_H.csv')
WNBD_original_R = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_R.csv')

# Import Information Entropy
IOI_InformationEntropy_original_C = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_C.csv")
IOI_InformationEntropy_original_G = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_G.csv")
IOI_InformationEntropy_original_H = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_H.csv")
IOI_InformationEntropy_original_R = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_R.csv")

# Import Tallest Bin
IOI_TallestBin_original_C = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_C.csv")
IOI_TallestBin_original_G = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_G.csv")
IOI_TallestBin_original_H = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_H.csv")
IOI_TallestBin_original_R = pd.read_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_R.csv")

# Import Off-Beatness
OffBeatness_original_C = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_C.csv')   #original
OffBeatness_original_G = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_G.csv')
OffBeatness_original_H = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_H.csv')
OffBeatness_original_R = pd.read_csv('./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_R.csv')


# Create metric comparison list
Toussaint_original = [Toussaint_original_C, Toussaint_original_G, Toussaint_original_H, Toussaint_original_R]
LonguetHigginsLee_original = [LonguetHigginsLee_original_C, LonguetHigginsLee_original_G, LonguetHigginsLee_original_H, LonguetHigginsLee_original_R]
Pressing_original = [Pressing_original_C, Pressing_original_G, Pressing_original_H, Pressing_original_R]
WNBD_original = [WNBD_original_C, WNBD_original_G, WNBD_original_H, WNBD_original_R]
IOI_InformationEntropy_original = [IOI_InformationEntropy_original_C, IOI_InformationEntropy_original_G, IOI_InformationEntropy_original_H, IOI_InformationEntropy_original_R]
OffBeatness_original = [OffBeatness_original_C, OffBeatness_original_G, OffBeatness_original_H, OffBeatness_original_R]

mean_scores = [mean_scores_C, mean_scores_G, mean_scores_H, mean_scores_R]
# mean_scores_VelModeSTD = [mean_scores_VelModeSTD_C, mean_scores_VelModeSTD_G, mean_scores_VelModeSTD_H, mean_scores_VelModeSTD_R]
# mean_scores_TestSTD = [mean_scores_TestSTD_C, mean_scores_TestSTD_G, mean_scores_TestSTD_H, mean_scores_TestSTD_R]
# mean_scores_UserSTD = [mean_scores_UserSTD_C, mean_scores_UserSTD_G, mean_scores_UserSTD_H, mean_scores_UserSTD_R]


# Sort metrics
sorted_Toussaint = sort_metric_scores(Toussaint_original)
sorted_LonguetHigginsLee = sort_metric_scores(LonguetHigginsLee_original)
sorted_Pressing = sort_metric_scores(Pressing_original)
sorted_WNBD = sort_metric_scores(WNBD_original)
sorted_IOI_InformationEntropy = sort_metric_scores(IOI_InformationEntropy_original)
sorted_OffBeatness = sort_metric_scores(OffBeatness_original)

# Def lists for the for loop
sorted_metrics = [ [sorted_Toussaint], [sorted_LonguetHigginsLee],
            [sorted_Pressing], [sorted_WNBD], 
            [sorted_IOI_InformationEntropy], [sorted_OffBeatness] ]
metric_names = ['Toussaint', 'LHL', 'Pressing', 'WNBD', 'IOI Entropy', 'Off-Beatness']


# Init lists and dfs
pearson_metric_coefficients = [[]] * len(sorted_metrics)
spearman_metric_coefficients = [[]] * len(sorted_metrics)


# metric_coefficients_VelModeSTD = [[]] * len(sorted_metrics)
# metric_coefficients_TestSTD = [[]] * len(sorted_metrics)
# metric_coefficients_UserSTD = [[]] * len(sorted_metrics)

# For each metric, compute correlation with test results
for i in range(len(sorted_metrics)):

    # Def metric
    sorted_metric_original = sorted_metrics[i][0]


    # Compute Pearson coefficient and get Pearson Table
    pearson_metric_coefficients[i] = compute_correlation(sorted_metric_original, mean_scores, 'Pearson')
    spearman_metric_coefficients[i] = compute_correlation(sorted_metric_original, mean_scores, 'Spearman')

    
    # metric_coefficients_VelModeSTD[i] = compute_correlation(sorted_metric_original, mean_scores_VelModeSTD, 'Pearson')
    # metric_coefficients_TestSTD[i] = compute_correlation(sorted_metric_original, mean_scores_TestSTD, 'Pearson')
    # metric_coefficients_UserSTD[i] = compute_correlation(sorted_metric_original, mean_scores_UserSTD, 'Pearson')


# Get table with only original metrics correlations (change here the metric coefficients and the folder name)
get_original_correlation_table(pearson_metric_coefficients, metric_names, 'Pearson', 'Raw')
get_original_correlation_table(spearman_metric_coefficients, metric_names, 'Spearman', 'Raw')

# get_original_correlation_table(metric_coefficients_VelModeSTD, metric_names, 'Pearson', 'VelModeSTD')
# get_original_correlation_table(metric_coefficients_TestSTD, metric_names, 'Pearson', 'TestVerSTD')
# get_original_correlation_table(metric_coefficients_UserSTD, metric_names, 'Pearson', 'UserSTD')



