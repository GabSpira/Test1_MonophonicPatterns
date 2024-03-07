import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from numpy import random
import random
import matplotlib as mpl
import matplotlib.lines as mlines
from CorrelationAnalysisFunctions import tot_Scatter, linear_regression_model, BoxPlot_scoresDistribution


# Seed
random.seed(0)

# Import test results
scores_C = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_C.csv")
scores_G = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_G.csv")
scores_H = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_H.csv")
scores_R = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_R.csv")

scores = [scores_C, scores_G, scores_H, scores_R]


# Import mean test results
mean_scores_C = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_C.csv")
mean_scores_G = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_G.csv")
mean_scores_H = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_H.csv")
mean_scores_R = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_R.csv")

mean_scores = [mean_scores_C, mean_scores_G, mean_scores_H, mean_scores_R]

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
OffBeatness_original = [OffBeatness_original_C, OffBeatness_original_G, OffBeatness_original_H, OffBeatness_original_R]
IOI_InformationEntropy_original = [IOI_InformationEntropy_original_C, IOI_InformationEntropy_original_G, IOI_InformationEntropy_original_H, IOI_InformationEntropy_original_R]



# Function to sort metrics
def sort_metric(metric_velocity):
    sorted_metric = []
    for mode in metric_velocity:
        sorted_metric.append( mode.sort_values('Scores') )
    return sorted_metric

# SOrt metrics
sorted_Toussaint_original = sort_metric(Toussaint_original)
sorted_LonguetHigginsLee_original = sort_metric(LonguetHigginsLee_original)
sorted_Pressing_original = sort_metric(Pressing_original)
sorted_WNBD_original = sort_metric(WNBD_original)
sorted_OffBeatness_original = sort_metric(OffBeatness_original)
sorted_IOIEntropy_original = sort_metric(IOI_InformationEntropy_original)


# Function to sort test results
def sort_test(mean_scores, sorted_metric_velocity):
    for i in range(len(sorted_metric_velocity)):
        # mean_scores[i] = mean_scores[i]['Mean Results'].reindex(sorted_metric_velocity[i]['Pattern'])
        mean_scores[i] = mean_scores[i].set_index('Pattern')
        mean_scores[i] = mean_scores[i].reindex(index=sorted_metric_velocity[i]['Pattern'])
        mean_scores[i] = mean_scores[i].reset_index()
        # print(sorted_metric_velocity[i]['Pattern'])
    return mean_scores


# Get linear correlation models of each couple of metrics and velocity mode test scores
for sorted_metric in [[sorted_Toussaint_original, 'Toussaint'], [sorted_LonguetHigginsLee_original, 'Longuet-Higgins&Lee'], 
               [sorted_Pressing_original, 'Pressing'], [sorted_WNBD_original, 'WNBD'], 
               [sorted_IOIEntropy_original, 'IOI_Entropy'], [sorted_OffBeatness_original, 'Off-Beatness']]:
    
    metric_modes = sorted_metric[0]    # list of 4 singular velocity mode df (2 columns)
    metric_name = sorted_metric[1]

    # Sort mean scores according to the metric
    sorted_mean_scores = sort_test(mean_scores, sorted_metric[0])
    
    for metric_mode in metric_modes:

        if (metric_mode.loc[0,['Pattern']].str[-1].values == 'C'): 
            vel_index , velocity_name = 0, 'Constant' 
        if (metric_mode.loc[0,['Pattern']].str[-1].values == 'G'): 
            vel_index , velocity_name = 1, 'Hierarchy'
        if (metric_mode.loc[0,['Pattern']].str[-1].values == 'H'):
            vel_index , velocity_name = 2, 'Human' 
        if (metric_mode.loc[0,['Pattern']].str[-1].values == 'R'): 
            vel_index , velocity_name = 3, 'Random'

        metric_mode = metric_mode.reset_index()
        ordered_columns = metric_mode['Pattern']
        resorted_scores = scores[vel_index][ordered_columns]
        std = resorted_scores.std()
        linear_regression_model(metric_mode[['Scores']], sorted_mean_scores[vel_index]['Mean Results'].values, std, metric_name, velocity_name)
        tot_Scatter(metric_mode[['Scores']], resorted_scores, metric_name, velocity_name)
        # BoxPlot_scoresDistribution(metric_mode, resorted_scores, metric_name, velocity_name)






