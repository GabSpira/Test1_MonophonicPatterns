from Original_Complexity_Metric import Original_Complexity_Metric_Class
from ComplexityMetricsFunctions import get_pattern, get_pattern_with_velocities
from MIDIFunctions import get_velocities
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import pandas as pd
import pretty_midi


#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. COMPUTE METRICS ORIGINAL -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

# Run this script to save one dataframe with all 120 patterns (no velocity distinction) for each of the Original Metrics

#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#



# Metrical grid informations (duration of metric units in seconds)
duration = 2                       #bar
tactus = duration/4                #quarter note
tatum = tactus/4                   #sixteenth note
length = int(duration/tatum)       #16


# Init metric dictionaries (gonna convert them in df)
complexity_Toussaint_original = {}
complexity_LonguetHigginsLee_original = {}
complexity_Pressing_original = {}
complexity_WNBD_original = {}
complexity_IOI_InformationEntropy_original = {}
complexity_IOI_TallestBin_original = {}
complexity_OffBeatness_original = {}
    

# iterate over each pattern of the dataset
directory = './Stimuli/test1/all/'

for file in os.listdir(directory):

    # Read file
    f = os.path.join(directory, file)

    # Name of file for dict, and from this index and v mode
    name = file[:-4]    
    velocity_mode = name[5]   
    pattern_number = name[:2]
    index = pattern_number + velocity_mode

    # Create pretty midi object from read midi file
    pm = pretty_midi.PrettyMIDI(f)

    # Onsets, velocities, pattern and print info
    onsets_times = pm.get_onsets()
    onsets_indeces = (onsets_times/tatum).astype(int)
    onsets_velocities = get_velocities(onsets_times, pm)
    pattern = get_pattern(length, onsets_indeces)
    
    print('\n### PATTERN n° ', index, ' INFORMATION ###')
    print('This pattern has onsets at relative positions (in the 16-pulses grid): ', onsets_indeces)
    print('The pattern is:    ', pattern)

    # Instantiate a metric class
    metrics = Original_Complexity_Metric_Class(length, onsets_indeces)

    # # Toussaint Complexity
    Toussaint_original = metrics.getToussaintComplexity()

    # # Longuet-Higgins & Lee
    LonguetHigginsLee_original = metrics.getLonguetHigginsLeeComplexity()

    # # Pressing 
    Pressing_original = metrics.getPressingComplexity()

    # # Weighted Note to Beat Distance
    WNBD_original = metrics.getWeightedNotetoBeatDistance()

    # # IOI - Information Entropy
    IOI_InformationEntropy_original = metrics.getInformationEntropyComplexity()[1] #local

    # # IOI - Tallest Bin
    IOI_TallestBin_original = metrics.getTallestBinComplexity()[1] #global

    # # Toussaint Off-Beatness
    OffBeatness_original = metrics.getOffBeatnessComplexity()

    # get metric value
    if type(Toussaint_original) is tuple: Toussaint_original = Toussaint_original[0]

    # Add pattern and relative metric score to metric dictionary
    complexity_Toussaint_original[index] = Toussaint_original
    complexity_LonguetHigginsLee_original[index] = LonguetHigginsLee_original
    complexity_Pressing_original[index] = Pressing_original
    complexity_WNBD_original[index] = WNBD_original
    complexity_IOI_InformationEntropy_original[index] = IOI_InformationEntropy_original
    complexity_IOI_TallestBin_original[index] = IOI_TallestBin_original
    complexity_OffBeatness_original[index] = OffBeatness_original




# List of patterns name sorted for velocity mode and number
sorted_patterns = [sorted(f'{i:02d}{letter}' for i in range(1, 31)) for letter in ['C', 'G', 'H', 'R']]
sorted_patterns = [element for nestedlist in sorted_patterns for element in nestedlist]

# Create original metric dataframes with all 120 pattern
Toussaint_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_Toussaint_original[pattern] for pattern in sorted_patterns]})
LonguetHigginsLee_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_LonguetHigginsLee_original[pattern] for pattern in sorted_patterns]})
Pressing_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_Pressing_original[pattern] for pattern in sorted_patterns]})
WNBD_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_WNBD_original[pattern] for pattern in sorted_patterns]})
OffBeatness_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_OffBeatness_original[pattern] for pattern in sorted_patterns]})
IOI_InformationEntropy_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_IOI_InformationEntropy_original[pattern] for pattern in sorted_patterns]})
IOI_TallestBin_original_tot = pd.DataFrame({'Pattern': sorted_patterns, 'Scores': [complexity_IOI_TallestBin_original[pattern] for pattern in sorted_patterns]})

# Save dataframes
Toussaint_original_tot.to_csv("./DataAnalysis/metrics_scores/Toussaint_original_tot.csv", index=False)
LonguetHigginsLee_original_tot.to_csv("./DataAnalysis/metrics_scores/LonguetHigginsLee_original_tot.csv", index=False)
Pressing_original_tot.to_csv("./DataAnalysis/metrics_scores/Pressing_original_tot.csv", index=False)
WNBD_original_tot.to_csv("./DataAnalysis/metrics_scores/WNBD_original_tot.csv", index=False)
IOI_InformationEntropy_original_tot.to_csv("./DataAnalysis/metrics_scores/IOI_InformationEntropy_original_tot.csv", index= False)
IOI_TallestBin_original_tot.to_csv("./DataAnalysis/metrics_scores/IOI_TallestBin_original_tot.csv", index= False)
OffBeatness_original_tot.to_csv("./DataAnalysis/metrics_scores/OffBeatness_original_tot.csv", index=False)


