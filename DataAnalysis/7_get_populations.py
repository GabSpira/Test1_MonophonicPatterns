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
from CorrelationAnalysisFunctions import mean_shifts_table, plot_shifts_distributions, plot_shifts_distribution_single






# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. GET POPULATIONS -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #

# Run this script to get the graphs about how users perceived the several velocity modes with respect to constant # 

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- #






# Import mean test results
mean_scores_C = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_C.csv")
mean_scores_G = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_G.csv")
mean_scores_H = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_H.csv")
mean_scores_R = pd.read_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_R.csv")

# Import test results
scores_C = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_C.csv")
scores_G = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_G.csv")
scores_H = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_H.csv")
scores_R = pd.read_csv("./DataAnalysis/test_scores/raw_scores/scores_R.csv")

# Import test results for test version
scores_test1 = pd.read_csv("./DataAnalysis/test_scores/raw_scores/subset/scores_subset1.csv")
scores_test2 = pd.read_csv("./DataAnalysis/test_scores/raw_scores/subset/scores_subset2.csv")
scores_test3 = pd.read_csv("./DataAnalysis/test_scores/raw_scores/subset/scores_subset3.csv")



# Obtain perceived complexity shifts (valued on the mean value of each pattern) in different velocity modes and their frequencies 
mean_scores_difference_C_G, counts_C_G = np.unique(round(mean_scores_G['Mean Results'] - mean_scores_C['Mean Results'],1), return_counts=True)
mean_scores_difference_C_H, counts_C_H = np.unique(round(mean_scores_H['Mean Results'] - mean_scores_C['Mean Results'],1), return_counts=True)
mean_scores_difference_C_R, counts_C_R = np.unique(round(mean_scores_R['Mean Results'] - mean_scores_C['Mean Results'],1), return_counts=True)

# Group items in list for plot loop
mean_scores_difference = [mean_scores_difference_C_G, mean_scores_difference_C_H, mean_scores_difference_C_R]
counts = [counts_C_G, counts_C_H, counts_C_R]

# Plot the distributions of the shifts among different velocity modes
plot_shifts_distributions(mean_scores_difference, counts, 'users_shifts_distributions_vs_C')

# Obtain mean shifts in perceived complexity among different velocity modes
mean_shifts = ([sum(mean_scores_difference[i]*counts[i])/sum(counts[i]) for i in range(len(counts))])
mean_shift_percentage = [str(round(mean_shifts[i],4) * 100) + '%' for i in range(len(mean_shifts))]

# Put them in a table
mean_shifts_table(mean_shift_percentage)




# Split test versions in 4 velocity modes
for letter in ['C', 'G', 'H', 'R']:

    # Constant, Gerarchia, Human, Random modes for each of the test versions  
    scores_test1_mode = scores_test1.iloc[:, [o.endswith(letter) for o in scores_test1.columns]]
    scores_test2_mode = scores_test2.iloc[:, [o.endswith(letter) for o in scores_test2.columns]]
    scores_test3_mode = scores_test3.iloc[:, [o.endswith(letter) for o in scores_test3.columns]]

    # Save mode scores (C, G, H, R)
    scores_test1_mode.to_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_subset1_{letter}.csv", index=False)
    scores_test2_mode.to_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_subset2_{letter}.csv", index=False)
    scores_test3_mode.to_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_subset3_{letter}.csv", index=False)

# Init dataframes with mean shifts in complexity perception for each user
meanShifts_forVelocityMode_forUser_test1 = pd.DataFrame(index = scores_test1.index, columns={'Shift G': [], 'Shift H': [], 'Shift R': [], 'version': []})
meanShifts_forVelocityMode_forUser_test2 = pd.DataFrame(index = scores_test2.index, columns={'Shift G': [], 'Shift H': [], 'Shift R': [], 'version': []})
meanShifts_forVelocityMode_forUser_test3 = pd.DataFrame(index = scores_test3.index, columns={'Shift G': [], 'Shift H': [], 'Shift R': [], 'version': []})





# Populate dataframes 
for test_version in ['subset1', 'subset2', 'subset3']:

    # Import velocity mode splitted scores for each test version
    scores_test_version_C = pd.read_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_{test_version}_C.csv")
    scores_test_version_G = pd.read_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_{test_version}_G.csv")
    scores_test_version_H = pd.read_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_{test_version}_H.csv")
    scores_test_version_R = pd.read_csv(f"./DataAnalysis/test_scores/raw_scores/subset/scores_{test_version}_R.csv")

    # Get shift in perceived complexity among different modes for each user
    for user in scores_test_version_C.index:
        
        # Handle unvalid answers
        if ( len(scores_test_version_C.loc[user].dropna().values) != len(scores_test_version_G.loc[user].dropna().values) ): 
            meanShifts_forVelocityMode_forUser_test1 = meanShifts_forVelocityMode_forUser_test1.drop([user])
            break

        # Get array of shifts perceived for each pattern in the current test version (for that user, for all velocity modes)
        shift_user_G_C = scores_test_version_G.loc[user].dropna().values - scores_test_version_C.loc[user].dropna().values
        shift_user_H_C = scores_test_version_H.loc[user].dropna().values - scores_test_version_C.loc[user].dropna().values
        shift_user_R_C = scores_test_version_R.loc[user].dropna().values - scores_test_version_C.loc[user].dropna().values
        
        # Get unique values and counts 
        shifts_values_user_G_C, counts_user_G_C = np.unique(shift_user_G_C, return_counts=True)
        shifts_values_user_H_C, counts_user_H_C = np.unique(shift_user_H_C, return_counts=True)
        shifts_values_user_R_C, counts_user_R_C = np.unique(shift_user_R_C, return_counts=True)

        # Weighted mean of the shifts (one for user)
        mean_shift_user_G_C = sum(shifts_values_user_G_C * counts_user_G_C) / sum(counts_user_G_C)
        mean_shift_user_H_C = sum(shifts_values_user_H_C * counts_user_H_C) / sum(counts_user_H_C) 
        mean_shift_user_R_C = sum(shifts_values_user_R_C * counts_user_R_C) / sum(counts_user_R_C)

        # Populate dataframe with the values of the mean shift the user (row) had evaluating the same patterns in different velocity modes
        if test_version=='subset1':
            meanShifts_forVelocityMode_forUser_test1.loc[user] = [mean_shift_user_G_C, mean_shift_user_H_C, mean_shift_user_R_C, 'subset1']
        if test_version=='subset2':
            meanShifts_forVelocityMode_forUser_test2.loc[user] = [mean_shift_user_G_C, mean_shift_user_H_C, mean_shift_user_R_C, 'subset2']
        if test_version=='subset3':
            meanShifts_forVelocityMode_forUser_test3.loc[user] = [mean_shift_user_G_C, mean_shift_user_H_C, mean_shift_user_R_C, 'subset3']



# Concat all users
meanShifts_forVelocityMode_forUser = pd.concat([meanShifts_forVelocityMode_forUser_test1, 
                                                meanShifts_forVelocityMode_forUser_test2, 
                                                meanShifts_forVelocityMode_forUser_test3], axis=0).reset_index(drop=True)

# Get mode-related pereption shift arrays, where each number is one user's mean shift in that mode
meanShifts_G = meanShifts_forVelocityMode_forUser['Shift G']
meanShifts_H = meanShifts_forVelocityMode_forUser['Shift H']
meanShifts_R = meanShifts_forVelocityMode_forUser['Shift R']

meanShifts = [meanShifts_G, meanShifts_H, meanShifts_R]

# Get values and counts for each user
shift_values_G, shift_counts_G = np.unique(meanShifts_G, return_counts=True)
shift_values_H, shift_counts_H = np.unique(meanShifts_H, return_counts=True)
shift_values_R, shift_counts_R = np.unique(meanShifts_R, return_counts=True)

# Put in list to plot
shift_values = [shift_values_G, shift_values_H, shift_values_R]
shift_counts = [shift_counts_G, shift_counts_H, shift_counts_R]

# Plot users distribution (each user has a perception shift for each mode)
plot_shifts_distribution_single(shift_values, shift_counts, 'users_shifts_distributions')












