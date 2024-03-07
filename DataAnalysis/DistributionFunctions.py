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





# ---------- FUNCTIONS TO CREATE AND MANIPULATE DATAFRAMES WITH TEST RESULTS ----------- #



def get_subtest_results(data):

    scores_data = []

    for result in data:
        user_id = {'id': result[0]}
        user_data = result[1]['userData']
        user_ratings = result[1]['userRatings']

        result_data = user_id.copy()
        result_data.update(user_data)

        for rating in user_ratings:
            rhythm = rating['rhythm']
            score = rating['score']
            result_data[rhythm] = score
        scores_data.append(result_data)

    df = pd.DataFrame.from_records(scores_data)
    
    # Exclude invalid answers
    df.replace('NA', np.nan, inplace=True)
    df = df.dropna()
    
    return df




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

    # Drop the last character from the column names so now each column indicates the rhythm without its type of velocity
    scores_C.columns = scores_C.columns.str[:2]
    scores_G.columns = scores_G.columns.str[:2]
    scores_H.columns = scores_H.columns.str[:2]
    scores_R.columns = scores_R.columns.str[:2]

    # Remove column names
    scores_C.columns = [''] * len(scores_C.columns)
    scores_G.columns = [''] * len(scores_G.columns)
    scores_H.columns = [''] * len(scores_H.columns)
    scores_R.columns = [''] * len(scores_R.columns)

    scores = [scores_C, scores_G, scores_H, scores_R]


    #mean values
    mean_C = scores_C.stack().values.mean()
    mean_G = scores_G.stack().values.mean()
    mean_R = scores_R.stack().values.mean()
    mean_H = scores_H.stack().values.mean() # mean(axis=0)
    print('Mean scores of each mode (C,G,R,H):', (mean_C, mean_G, mean_R, mean_H))

    var_C = scores_C.stack().values.std()
    var_G = scores_G.stack().values.std()
    var_R = scores_R.stack().values.std()
    var_H = scores_H.stack().values.std()
    print('Variances of each mode (C,G,R,H):', (var_C, var_G, var_R, var_H))

    return scores



def get_scores_totSubset(df):

    print(df.shape)
    print(df.columns)
    # Keep only the columns of the scores in the 'scores' dataframe 
    scores = df.iloc[:, 6:127]
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

    return scores


def get_counts_for_velocity(scores):
    
    scores_C = scores[0]
    scores_G = scores[1]
    scores_H = scores[2]
    scores_R = scores[3]
    
    # Get arrays from dataframes (for a score distribution, no info on the rhythms is needed)
    flatten_C = scores_C.values.flatten()
    flatten_G = scores_G.values.flatten()
    flatten_H = scores_H.values.flatten()
    flatten_R = scores_R.values.flatten()

    flatten_C = [round(k,3) for k in scores_C.values.flatten()]
    flatten_G = [round(k,3) for k in scores_G.values.flatten()]
    flatten_H = [round(k,3) for k in scores_H.values.flatten()]
    flatten_R = [round(k,3) for k in scores_R.values.flatten()]

    # Count the occurrences of each score in each velocity mode array
    values_C, counts_C = np.unique(flatten_C, return_counts=True)
    values_G, counts_G = np.unique(flatten_G, return_counts=True)
    values_H, counts_H = np.unique(flatten_H, return_counts=True)
    values_R, counts_R = np.unique(flatten_R, return_counts=True)
    # print(values_C)
    # print(counts_C)

    # Discard NaN values
    # values_C = values_C[:-1]
    # counts_C = counts_C[:-1]
    # values_G = values_G[:-1]
    # counts_G = counts_G[:-1]
    # values_H = values_H[:-1]
    # counts_H = counts_H[:-1]
    # values_R = values_R[:-1]
    # counts_R = counts_R[:-1]

    values = [values_C, values_G, values_H, values_R]
    counts = [counts_C, counts_G, counts_H, counts_R]

    return values, counts

def get_counts_for_velocity_tot(scores):
    
    scores_C = scores[0]
    scores_G = scores[1]
    scores_H = scores[2]
    scores_R = scores[3]
    
    # Get arrays from dataframes (for a score distribution, no info on the rhythms is needed)
    flatten_C = scores_C.values.flatten()
    flatten_G = scores_G.values.flatten()
    flatten_H = scores_H.values.flatten()
    flatten_R = scores_R.values.flatten()

    flatten_C = [round(k,3) for k in scores_C.values.flatten()]
    flatten_G = [round(k,3) for k in scores_G.values.flatten()]
    flatten_H = [round(k,3) for k in scores_H.values.flatten()]
    flatten_R = [round(k,3) for k in scores_R.values.flatten()]

    # Count the occurrences of each score in each velocity mode array
    values_C, counts_C = np.unique(flatten_C, return_counts=True)
    values_G, counts_G = np.unique(flatten_G, return_counts=True)
    values_H, counts_H = np.unique(flatten_H, return_counts=True)
    values_R, counts_R = np.unique(flatten_R, return_counts=True)
    # print(values_C)
    # print(counts_C)

    # Discard NaN values
    values_C = values_C[:-1]
    counts_C = counts_C[:-1]
    values_G = values_G[:-1]
    counts_G = counts_G[:-1]
    values_H = values_H[:-1]
    counts_H = counts_H[:-1]
    values_R = values_R[:-1]
    counts_R = counts_R[:-1]

    values = [values_C, values_G, values_H, values_R]
    counts = [counts_C, counts_G, counts_H, counts_R]

    return values, counts


def standardize_for_user(scores): 

    scores_C = scores[0].copy()
    scores_G = scores[1].copy()
    scores_H = scores[2].copy()
    scores_R = scores[3].copy()

    for i in range(scores_C.shape[0]):              # for each row, so each user
        
        std_C = np.std(scores_C.loc[i])
        std_G = np.std(scores_G.loc[i])
        std_H = np.std(scores_H.loc[i])
        std_R = np.std(scores_R.loc[i])

        mean_C = np.mean(scores_C.loc[i])
        mean_G = np.mean(scores_G.loc[i])
        mean_H = np.mean(scores_H.loc[i])
        mean_R = np.mean(scores_R.loc[i])

        scores_C.loc[i] = (scores_C.loc[i] - mean_C)/std_C
        scores_G.loc[i] = (scores_G.loc[i] - mean_G)/std_G
        scores_H.loc[i] = (scores_H.loc[i] - mean_H)/std_H
        scores_R.loc[i] = (scores_R.loc[i] - mean_R)/std_R

    scores_norm = [scores_C, scores_G, scores_H, scores_R]
    
    return scores_norm


def print_distribution(values, counts):

    counts_tot = []

    for i in range(len(values[0])):                           # for each value from 1 to 5
        tot_counts = 0
        counts_for_velocity = []
        for j in range(len(counts)):                          # for each velocity mode
            counts_for_velocity.append(counts[j][i])
            tot_counts = tot_counts + counts[j][i]
        counts_tot.append(tot_counts)
        print('counts for ', values[0][i], ': ', counts_for_velocity, tot_counts)
    print(counts_tot)
    return(counts_tot)



def standardize_for_user(scores):

    scores_C = scores[0]
    scores_G = scores[1]
    scores_H = scores[2]
    scores_R = scores[3]


    # standardize for user
    for i in range(len(scores_H.index)):
        
        H_mean_user = scores_H.loc[i].mean()
        H_std_user = scores_H.loc[i].std()

        scores_C.loc[i] = (scores_C.loc[i]-H_mean_user)/H_std_user

        R_mean_user = scores_R.loc[i].mean()
        R_std_user = scores_R.loc[i].std()

        R = (scores_C.loc[i].dropna()-R_mean_user)/R_std_user





# -------- FUNCTIONS TO PLOT RESULTS ---------- #

def plot_histogram(values, counts, name):

    values_C = values[0]
    values_G = values[1]
    values_H = values[2]
    values_R = values[3]

    counts_C = counts[0]
    counts_G = counts[1]
    counts_H = counts[2]
    counts_R = counts[3]

    # print(counts)

    x_label = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    

    # Background of the plot
    sns.set(style="whitegrid")

    # Plot 4 histograms next to each other
    plt.figure(figsize=(10, 4))
    print(counts_C)
    plt.bar(values_C-0.3, counts_C, color='#83a5de', edgecolor='#83a5de', alpha=0.6, align='center',  width=0.2)
    plt.bar(values_G-0.1, counts_G, color='#34bc4a', edgecolor='#34bc4a', alpha=0.5, width=0.2)
    plt.bar(values_R+0.1, counts_R, color='#ffde7f', edgecolor='#ffde7f', alpha=0.6, width=0.2)
    plt.bar(values_H+0.3, counts_H, color='#f74c7e', edgecolor='#f74c7e', alpha=0.6, width=0.2)

    # Legend settings
    colors = {'Constant':'#83a5de', 'Hierarchy':'#34bc4a', 'Random':'#ffde7f', 'Performed':'#f74c7e'}         
    values = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in values]
    plt.legend(handles, values, fontsize=15)

    # Plot settings
    title = 'Dataset - Scores Distribution - ' + name
    path = f'./DataAnalysis/distributions/Scores Distribution - {name}.png'
    # plt.suptitle(title, fontsize=13, fontweight='bold', y=0.96, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))
    plt.gca().set_xticks(range(1,6), x_label, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('')
    plt.tight_layout()
    # plt.show()    
    plt.savefig(path)


def plot_tot_histogram(counts_tot):


    # print(counts)

    x_label = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    

    # Background of the plot
    sns.set(style="whitegrid")

    # Plot 4 histograms next to each other
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(1,6), counts_tot, color='#83a5de', edgecolor='#83a5de', alpha=0.6, align='center',  width=0.9)


    # Legend settings

    # Plot settings
    title = 'Dataset - Scores Distribution - Total (no velocity)'
    path = './DataAnalysis/distributions/Scores Distribution - Total (no velocity).png'
    # plt.suptitle(title, fontsize=13, fontweight='bold', y=0.96, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))
    plt.gca().set_xticks(range(1,6), x_label, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('')
    plt.tight_layout()
    # plt.show()    
    plt.savefig(path)


