import pandas as pd
import json
import dataframe_image as dfi
from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import re
from DistributionFunctions import get_subtest_results, get_scores_totSubset




# -.-.-.-.-.-.-.-.-.-. SAVE TEST SCORES .-.-.-.-.-.-.-.-.-.-.- #

# Run this script to save a dataframe with all the test results

# .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- #






def get_scores(df):

    # Keep only the columns of the scores in the 'scores' dataframe 
    scores = df.iloc[:, 7:127]
    scores.columns = scores.columns.str[-3:]
    scores = scores.reindex(sorted(scores.columns), axis=1)          # sort numerically the rhythms

    # Turn strings into numerical scores to use histplot
    vote_mapping = {
        'Very low': 1,
        'Low': 2,
        'Medium': 3,
        'High': 4,
        'Very high': 5
    }

    scores = scores.applymap(lambda x: vote_mapping.get(x, np.nan))

    # Create one dataframe for each type of velocity
    scores_C = scores.iloc[:, [o.endswith('C') for o in scores.columns]]
    scores_G = scores.iloc[:, [o.endswith('G') for o in scores.columns]]
    scores_H = scores.iloc[:, [o.endswith('H') for o in scores.columns]]
    scores_R = scores.iloc[:, [o.endswith('R') for o in scores.columns]]

    scores = [scores_C, scores_G, scores_H, scores_R]


    #mean values
    mean_C = scores_C.stack().values.mean()
    mean_G = scores_G.stack().values.mean()
    mean_R = scores_R.stack().values.mean()
    mean_H = scores_H.stack().values.mean() # mean(axis=0)
    # print('Mean scores of each mode (C,G,R,H):', (mean_C, mean_G, mean_R, mean_H))

    var_C = scores_C.stack().values.std()
    var_G = scores_G.stack().values.std()
    var_R = scores_R.stack().values.std()
    var_H = scores_H.stack().values.std()
    # print('Variances of each mode (C,G,R,H):', (var_C, var_G, var_R, var_H))

    return scores






# Load JSON
test1 = './output/exportedData1.json'
test2 = './output/exportedData2.json'
test3 = './output/exportedData3.json'

with open(test1, 'r', encoding='utf-8') as file:
    data1 = json.load(file)
with open(test2, 'r', encoding='utf-8') as file:
    data2 = json.load(file)
with open(test3, 'r', encoding='utf-8') as file:
    data3 = json.load(file)

# Get partial and total set of scores dataframes from json
df1 = get_subtest_results(data1)
df2 = get_subtest_results(data2)
df3 = get_subtest_results(data3)

df = pd.concat([df1, df2, df3], axis=0)
df.reset_index(inplace=True)

# Save partial and total set of scores dataframes
df.to_csv("./DataAnalysis/test_scores/results/results_tot.csv", index=False)
df1.to_csv("./DataAnalysis/test_scores/results/results_1.csv", index=False)
df2.to_csv("./DataAnalysis/test_scores/results/results_2.csv", index=False)
df3.to_csv("./DataAnalysis/test_scores/results/results_3.csv", index=False)

# Get scores dataframes for each velocity mode
scores = get_scores(df)   
scores_C = scores[0]
scores_G = scores[1]
scores_H = scores[2]
scores_R = scores[3]

# Save velocity mode scores dataframes
scores_C.to_csv("./DataAnalysis/test_scores/raw_scores/scores_C.csv", index=False)
scores_G.to_csv("./DataAnalysis/test_scores/raw_scores/scores_G.csv", index=False)
scores_H.to_csv("./DataAnalysis/test_scores/raw_scores/scores_H.csv", index=False)
scores_R.to_csv("./DataAnalysis/test_scores/raw_scores/scores_R.csv", index=False)

# Get mean scores dfs for each velocity mode
mean_scores_C = scores_C.mean(axis=0)
mean_scores_G = scores_G.mean(axis=0)
mean_scores_H = scores_H.mean(axis=0)
mean_scores_R = scores_R.mean(axis=0)

# Name columns
mean_scores_C = mean_scores_C.to_frame().reset_index().rename(columns={'index':'Pattern', 0:'Mean Results'})
mean_scores_G = mean_scores_G.to_frame().reset_index().rename(columns={'index':'Pattern', 0:'Mean Results'})
mean_scores_H = mean_scores_H.to_frame().reset_index().rename(columns={'index':'Pattern', 0:'Mean Results'})
mean_scores_R = mean_scores_R.to_frame().reset_index().rename(columns={'index':'Pattern', 0:'Mean Results'})

# Save mean scores dataframes
mean_scores_C.to_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_C.csv", index=False)
mean_scores_G.to_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_G.csv", index=False)
mean_scores_H.to_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_H.csv", index=False)
mean_scores_R.to_csv("./DataAnalysis/test_scores/raw_scores/mean/mean_scores_R.csv", index=False)

# Get scores dataframe for each subtest version
scores_subset1 = get_scores_totSubset(df1)
scores_subset2 = get_scores_totSubset(df2)
scores_subset3 = get_scores_totSubset(df3)

# Save test version subset scores dataframes
scores_subset1.to_csv("./DataAnalysis/test_scores/raw_scores/subset/scores_subset1.csv", index=False)
scores_subset2.to_csv("./DataAnalysis/test_scores/raw_scores/subset/scores_subset2.csv", index=False)
scores_subset3.to_csv("./DataAnalysis/test_scores/raw_scores/subset/scores_subset3.csv", index=False)
