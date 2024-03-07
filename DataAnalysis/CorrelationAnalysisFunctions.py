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
import matplotlib as mpl
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from numpy import random


# Sort test results
def sort_test_results(test_results):

    # This function takes the list of 4 splitted-for-velocity-mode scores Series, 
    # The output is still a list with each of these is sorted and the pattern column becomes the index column 

    sorted_test_results = []
    for velocity_mode in test_results:

        # Remove index (Pattern code becomes index)
        velocity_mode = velocity_mode.set_index('Pattern')

        # Sort each mode by score
        velocity_mode = velocity_mode['Mean Results'].sort_values()

        sorted_test_results.append(velocity_mode)
    
    return(sorted_test_results)



# Sort test results
def sort_metric_scores(metric_scores):

    # This function takes the list of 4 splitted-for-velocity-mode scores Series, 
    # The output is still a list with each of these is sorted and the pattern column becomes the index column 

    sorted_metric_scores = []
    for velocity_mode in metric_scores:

        # Remove index (Pattern code becomes index)
        # velocity_mode = velocity_mode.set_index('Pattern')

        # Sort each mode by score
        velocity_mode = velocity_mode.sort_values(by=['Scores'])

        sorted_metric_scores.append(velocity_mode)
    
    return(sorted_metric_scores)


# Create correlation df for each metric
def compute_correlation(sorted_metric_scores, test_results, coefficient_type):

    # This function takes a list of 4 splitted-for-velocity-mode, sorted metric values
    # And returns a column with 4 Pearson coefficients representing the correlation btw metric and test results in each mode
    
    coefficient_scores = []

    # For each velocity mode subset (C, G, H, R)
    for i in range(len(sorted_metric_scores)):

        # Def the velocity mode subset of scores
        sorted_metric = sorted_metric_scores[i]
        test = test_results[i]

        # Sort test results according to the metric
        sorted_metric = sorted_metric.set_index('Pattern')
        test = test.set_index('Pattern')
        sorted_test = test.reindex(sorted_metric.index)

        # print((sorted_test.iloc[0]))
        # print(sorted_metric.iloc[0][0])
        # print(type(sorted_metric['Scores'].to_numpy()))
        # print((sorted_test['Mean Results'].to_numpy()))

        # a = scipy.stats.pearsonr(sorted_metric['Scores'].to_numpy(), sorted_test['Mean Results'].to_numpy())

        # Compute coefficients
        if coefficient_type=='Pearson': 
            coefficient = scipy.stats.pearsonr(sorted_metric['Scores'].values, sorted_test['Mean Results'].values)
        elif coefficient_type=='Spearman': coefficient = scipy.stats.spearmanr(sorted_metric['Scores'], sorted_test['Mean Results'])

        # Get list with coefficients for each velocity mode
        coefficient_scores.append(round(coefficient[0],5))

    return(coefficient_scores)


# Def tables style
styles = [
    dict(selector="tr:hover",
                props=[("background", "#D6EEEE")]),
    dict(selector="th.col_heading", props=[("color", "#fff"),
                            ("border", "3px solid #FFFFFF"),
                            ("padding", "12px 35px"),
                            #("border-collapse", "collapse"),
                            ("background", "#1D4477"),
                            ("font-size", "18px")
                            ]),
    dict(selector="th.row_heading", props=[("color", "#fff"),
                            ("border", "3px solid #FFFFFF"),
                            ("padding", "12px 35px"),
                            #("border-collapse", "collapse"),
                            ("background", "#1D4477"),
                            ("font-size", "15px")
                            ]),
    dict(selector="td", props=[("color", "#000000"),
                            ("border", "3px solid #FFFFFF"),
                            ("padding", "12px 35px"),
                            ('margin', '2px'),
                            # ("border-collapse", "collapse"),
                            ("font-size", "15px")   
                            ]),
    dict(selector="table", props=[                                   
                                    ("font-family" , 'Helvetica'),
                                    ("margin" , "25px auto"),
                                    ("border-collapse" , "collapse"),
                                    ("border" , "3px solid #FFFFFF"),
                                    ("border-bottom" , "2px solid #00cccc")                            
                                    ]),
    dict(selector="caption", props=[("caption-side", "left"), ("margin", "6px"), ("text-align", "right"), ("font-size", "120%"),
                                        ("font-weight", "bold")]),
    dict(selector="tr:nth-child(even)", props=[
        ("background-color", "#D9EFFA"),
    ]),
    dict(selector="tr:nth-child(odd)", props=[
        ("background-color", "#DEF4FF"),
    ]),
]

# Style function to highlight max value
def style_max(x, max_row, max_col):

    color = 'background-color: #BFE3F5'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    
    df1.iloc[max_row, max_col] = color
    return df1





# Table with metric correlations with test results (comparing original and proposed version)
def metric_table(coefficient_original, coefficient_velocity, metric_name, coefficient_type, folder):

    # This function creates a table with the correlation values between an objective metric 
    # (both in its original and velocity versions) and the subjective listening test results
    
    # Set columns
    metric_table = pd.DataFrame({'Original '+metric_name : coefficient_original, 'Velocity '+metric_name: coefficient_velocity})
    
    # Set rows
    metric_table = metric_table.rename(index={0:'Constant', 1:'Hierarchy', 2:'Human', 3:'Random'}) 
  
    print(metric_table)

    # Style table    
    table_styled = metric_table.style.set_caption(coefficient_type + ' Coefficient Comparison')

    table_styled.set_table_styles(styles)
    table_styled.apply(lambda row: ['font-weight: bold' if val == row.max() else '' for val in row], axis=1)
    
    # Underline max 
    max_index = metric_table.values.argmax()
    max_row, max_col = divmod(max_index, metric_table.shape[1])
    table_styled.apply(style_max, max_row=max_row, max_col=max_col, axis=None)

    # Save table
    path = './DataAnalysis/correlation_tables/'+ folder +'/' + coefficient_type + '_' + metric_name +'.png'
    dfi.export(table_styled, path)

    



# Create table
def get_original_correlation_table(metric_coefficients, metric_names, coefficient_type, description):

    # This function creates a table with the correlation values between an objective metric 
    # (both in its original and velocity versions) and the subjective listening test results
    
    # Set columns
    metric_table = pd.DataFrame({metric_names[0] : metric_coefficients[0], 
                                 metric_names[1] : metric_coefficients[1],
                                 metric_names[2] : metric_coefficients[2],
                                 metric_names[3] : metric_coefficients[3],
                                 metric_names[4] : metric_coefficients[4],
                                 metric_names[5] : metric_coefficients[5]})
                                #  metric_names[6] : metric_coefficients[6]})
    
    # Set rows
    metric_table = metric_table.rename(index={0:'Constant', 1:'Hierarchy', 2:'Random', 3:'Performed'}) 
  
    print(metric_table)

    # Style table    
    table_styled = metric_table.style.set_caption(coefficient_type + ' Correlation with Subjective Data in Different Velocity Modes')

    table_styled.set_table_styles(styles)
    # table_styled.apply(lambda row: ['font-weight: bold' if val == row.max() else '' for val in row], axis=1)
    
    # Underline max 
    # max_index = metric_table.values.argmax()
    # max_row, max_col = divmod(max_index, metric_table.shape[1])
    # table_styled.apply(style_max, max_row=max_row, max_col=max_col, axis=None)

    # Save table
    path = './DataAnalysis/correlation_tables/original/' + coefficient_type + '_OriginalMetrics_' + description +'3.png'
    dfi.export(table_styled, path)

    return table



# Function to create table with mean shifts in perceived complexity (valued on mean pattern scores for each velocity mode)
def mean_shifts_table(mean_shifts):

    # Column
    mean_shifts_table = pd.DataFrame({'Mean shift percentage': mean_shifts})

    # Rows
    mean_shifts_table = mean_shifts_table.rename(index={0: 'Hierarchy', 1:'Human', 2:'Random'}) 

    print(mean_shifts_table)

    # Style table    
    table_styled = mean_shifts_table.style.set_caption('Mean perceived complexity shift with respect to Constant mode')


    table_styled.set_table_styles(styles)
    table_styled.apply(lambda row: ['font-weight: bold' if val == row.max() else '' for val in row], axis=1)
    
    # Underline max 
    max_index = mean_shifts_table.values.argmax()
    max_row, max_col = divmod(max_index, mean_shifts_table.shape[1])
    table_styled.apply(style_max, max_row=max_row, max_col=max_col, axis=None)

    # Save table
    path = './DataAnalysis/populations/mean_shifts_table.png'
    dfi.export(table_styled, path)


# Function to plot shift distributions
def plot_shifts_distributions(mean_scores_difference, counts, name, folder):
    
    # Plot settings
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8),  gridspec_kw={'hspace': 0.45}, sharex='col')
    colors = ['#34bc4a','#f74c7e', '#ffde7f']  
    titles = ['Difference between Hierarchy and Constant', 'Difference between Human and Constant', 'Difference between Random and Constant'] 
    ax=[axes[0], axes[1], axes[2]]
    plt.suptitle('Distribution of users mean shifts in complexity perception between different velocity modes', fontsize=13, fontweight='bold', y=0.96, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))
    
    # Set global min and max for all axis
    min_tick, max_tick = 0,0
    for list in mean_scores_difference:
        if min(list) < min_tick:    min_tick = min(list)
        if max(list) > max_tick:    max_tick = max(list)

    # Plot 3 histograms
    for i in range(len(mean_scores_difference)):
        ax[i].axvline(x=0, color='red', linestyle='dotted')
        ax[i].bar(mean_scores_difference[i], counts[i], color=colors[i], edgecolor='black', alpha=0.7, align='center', width=0.09)
        ax[i].set_title(titles[i], fontdict={'fontsize': 10, 'fontweight': 'bold'})
        ax[i].set_ylabel("Frequency")
        ax[i].tick_params(labelbottom=True)
        ax[i].set_yticks(np.arange(0,15,2))

    # Save plot
    plt.savefig(f'./output/pattern_difference_inVelModes/{folder}/{name}.png')



def linear_regression_model(x, y, std, metric, vel_mode):

    # Fit model
    model = LinearRegression()
    model.fit(x, y)

    # Predict and compute mean error 
    predictions = model.predict(x)
    e = mean_absolute_error(y, predictions)

    # Plot 
    plt.figure(figsize=(10, 6))

    # Standard Deviations of mean subjective ratings
    for i in range(len((std))):
        plt.plot([x['Scores'][i], x['Scores'][i]], [y[i] - std[i]/2, y[i] + std[i]/2], color='#377CE4', linewidth=2, alpha=0.85)
    plt.plot(0,0,  color='#377CE4', label = 'Standard Deviation')

    # Plot linear regression line and subjective ratings
    plt.scatter(x, y, label='Mean Subjective Ratings', color = '#377CE4')
    plt.plot(x, predictions, color='#FF741E', linestyle='dashed', alpha=0.9, label='Linear Regression')

    # Plot settings
    plt.xlabel(f'{metric} Metric Scores', fontsize=16) 
    plt.ylabel('Subjective Ratings', fontsize=16) 
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.legend(fontsize=15)
    title = 'Linear Regression Model, ' + metric + ' Metric, Velocity Mode: ' + vel_mode
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.99, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))

    # Coefficient of Determination
    R2 = model.score(x, y)

    # Compute line parameters
    q = model.intercept_
    m = model.coef_
    print(f'You can get a coefficient of determination R^2 = {round(R2,4)} with the linear model: y = {round(m[0],4)}*x + {round(q,4)}')

    # Add equation and R^2 text box
    equation = f'y = {round(m[0], 4)}*x + {round(q, 4)}'
    r_squared = f'R^2 = {round(R2, 4)}'
    text_box = f'Equation: {equation}\n{r_squared}'
    plt.text(0.95, 0.05, text_box, transform=plt.gca().transAxes, fontsize=16,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='#edf4ff', alpha=1))

    # Save graph
    plt.tight_layout()
    path = f'./DataAnalysis/linear_regression_models/Linear_{metric}_{vel_mode}.png'
    plt.savefig(path)
    # plt.show()



def tot_Scatter(x, y_tot, metric, vel_mode):

    mean = y_tot.mean()

    # Fit model
    model = LinearRegression()
    model.fit(x, mean)

    # Predict and compute mean error 
    predictions = model.predict(x)

    # Plot 
    plt.figure(figsize=(10, 6))

    cmap = mpl.cm.Blues(np.linspace(0,1,20))
    cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

    # Plot Scatter of all subjective ratings
    for index, row in y_tot.iterrows():
        offset = index/80 * random.choice([-1, 1])
        distance_from_mean = np.sqrt(100/abs(mean - row))*1.5
        plt.scatter(x, row+offset, s=distance_from_mean, cmap=cmap, alpha=0.35, c=distance_from_mean)
    
    color_bar = plt.colorbar(label='Distance from mean value')
    color_bar.set_alpha(0.7)
    color_bar.draw_all()

    dot = mlines.Line2D([], [], color='#477fb6', marker='o', linestyle='None',
                            markersize=4, label='Subjective Ratings wrt Metric Scores')
     
    line, = plt.plot(x, predictions, color='#FF741E', linestyle='dashed', alpha=0.9, label='Linear Regression')

    plt.legend(handles=[dot, line])

    # Plot settings
    plt.xlabel(f'{metric} Metric Scores') 
    plt.ylabel('Subjective Ratings') 
    plt.grid(True)
    title = 'All subjective ratings with respect to ' + metric + ' Metric, Velocity Mode: ' + vel_mode
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.96, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))

    # Save graph
    path = f'./DataAnalysis/scatter/Scatter_{metric}_{vel_mode}_original.png'
    plt.savefig(path)
    # plt.show()




def BoxPlot_scoresDistribution(x, y_tot, metric, vel_mode):

    # Melt y_tot to long format
    y_melted = y_tot.melt(var_name='Rhythm', value_name='Subjective Rating')

    # Combine x with y_melted
    combined_data = pd.merge(x, y_melted, left_on='Pattern', right_on='Rhythm')

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_data, x='Pattern', y='Subjective Rating', color='#377CE4')
    # sns.violinplot(data=combined_data, x='Pattern', y='Subjective Rating', color='#377CE4')
    
    # Plot settings
    plt.xlabel('Rhythm') 
    plt.ylabel('Subjective Ratings') 
    # plt.grid(True)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    title = f'Boxplot of Subjective Ratings by Rhythm, {metric}, {vel_mode}'
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.96, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))

    # Save graph
    path = f'./DataAnalysis/boxplot/BoxPlot_{metric}_{vel_mode}'
    plt.savefig(path)
    # plt.show()


# Function to plot shift distributions
def plot_shifts_distributions(mean_scores_difference, counts, name):
    
    # Plot settings
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(16, 8),  gridspec_kw={'hspace': 0.45}, sharex='col')
    colors = ['#34bc4a','#f74c7e', '#ffde7f']  
    titles = ['Difference between Hierarchy and Constant', 'Difference between Human and Constant', 'Difference between Random and Constant'] 
    ax=[axes[0], axes[1], axes[2]]
    plt.suptitle('Distribution of users mean shifts in complexity perception between different velocity modes', fontsize=13, fontweight='bold', y=0.96, x=0.5, bbox=dict(boxstyle='square, pad=0.5', ec=(1., 0.5, 0.5), facecolor='#FFF7DA'))
    
    # # Set global min and max for all axis
    # min_tick, max_tick = 0,0
    # for list in mean_scores_difference:
    #     if min(list) < min_tick:    min_tick = min(list)
    #     if max(list) > max_tick:    max_tick = max(list)

    # Plot 3 histograms
    for i in range(len(mean_scores_difference)):
        ax[i].axvline(x=0, color='red', linestyle='dotted')
        ax[i].bar(mean_scores_difference[i], counts[i], color=colors[i], edgecolor='black', alpha=0.7, align='center', width=0.09)
        ax[i].set_title(titles[i], fontdict={'fontsize': 10, 'fontweight': 'bold'})
        ax[i].set_ylabel("Frequency")
        ax[i].tick_params(labelbottom=True)
        ax[i].set_yticks(np.arange(0,15,2))
        # plt.xlim([min_tick-0.1, max_tick+0.1])

    # Save plot
    plt.savefig(f'./DataAnalysis/populations/{name}.png')


# Function to plot one shift distributions at a time
def plot_shifts_distribution_single(mean_scores_difference, counts, name):
    
    subset = ''

    # Plot settings
    sns.set(style="whitegrid")
    plt.figure(figsize=(16,5))
    colors = ['#34bc4a','#f74c7e', '#ffde7f']  

    # Set global min and max for all axis
    min_tick, max_tick = 0,0
    for list in mean_scores_difference:
        if min(list) < min_tick:    min_tick = min(list)
        if max(list) > max_tick:    max_tick = max(list)

    for i in range(3):

        plt.cla()
        plt.axvline(x=0, color='red', linestyle='dotted')
        plt.bar(mean_scores_difference[i], counts[i], color=colors[i], edgecolor='black', alpha=0.7, align='center', width=0.09)
        # plt.set_title(titles[i], fontdict={'fontsize': 10, 'fontweight': 'bold'}) 
        plt.ylabel("Frequency", fontsize = 25)
        plt.xlabel("Shifts", fontsize = 25)
        plt.tick_params(labelbottom=True, labelsize = 22)
        plt.yticks(np.arange(0,15,2))
        plt.xlim([min_tick-0.1, max_tick+0.1])

        plt.tight_layout()

        if i == 0: subset = 'subset1'
        elif i == 1: subset = 'subset2'
        elif i == 2: subset = 'subset3'

        # Save plot
        # plt.show()
        plt.savefig(f'./DataAnalysis/populations/{name}_{subset}.png')
