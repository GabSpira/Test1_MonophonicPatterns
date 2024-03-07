import pandas as pd
import json
import dataframe_image as dfi
from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import re
from DistributionFunctions import get_counts_for_velocity, get_counts_for_velocity_tot, get_scores, plot_histogram, print_distribution, plot_tot_histogram





# .-.-.-.-.-.-.-.-.-.-. COMPUTE SCORES DISTRIBUTION .-.-.-.-.-.-.-.-.-.-.-. #

# Run this code to analyze the distribution of the test scores with bar plots

# .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. #






# Read results dataframes
df = pd.read_csv("./DataAnalysis/test_scores/results/results_tot.csv")
df1 = pd.read_csv("./DataAnalysis/test_scores/results/results_1.csv")
df2 = pd.read_csv("./DataAnalysis/test_scores/results/results_2.csv")
df3 = pd.read_csv("./DataAnalysis/test_scores/results/results_3.csv")

# Get scores and counts dataframes for each velocity mode
scores = get_scores(df)                                   # Array with scores_C, scores_G, scores_H, scores_R
values, counts = get_counts_for_velocity_tot(scores)          # Arrays of values_C, values_G, values_H, values_R and counts_C, counts_G, counts_H, counts_R

# Get scores and counts for each subtest
scores1 = get_scores(df1)
values1, counts1 = get_counts_for_velocity(scores1)
scores2 = get_scores(df2)
values2, counts2 = get_counts_for_velocity(scores2)
scores3 = get_scores(df3)
values3, counts3 = get_counts_for_velocity(scores3)

# Plot distribution for each subtest set of scores
plot_histogram(values1, counts1, 'Subtest 1')
plot_histogram(values2, counts2, 'Subtest 2')
plot_histogram(values3, counts3, 'Subtest 3')

# Print and plot total dataset scores distribution
plot_histogram(values, counts, 'Total')
counts_tot = print_distribution(values, counts)
plot_tot_histogram(counts_tot)






