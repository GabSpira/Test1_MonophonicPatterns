import pandas as pd


#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. SPLIT ORIGINAL METRIC DATAFRAMES -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

# Run this script to save the splitted for velocity mode versions of each of the considered original metrics

#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#


# Import metric scores
Toussaint_original = pd.read_csv('./DataAnalysis/metrics_scores/Toussaint_original_tot.csv')
LonguetHigginsLee_original = pd.read_csv('./DataAnalysis/metrics_scores/LonguetHigginsLee_original_tot.csv')
Pressing_original = pd.read_csv('./DataAnalysis/metrics_scores/Pressing_original_tot.csv')
WNBD_original = pd.read_csv('./DataAnalysis/metrics_scores/WNBD_original_tot.csv')
OffBeatness_original = pd.read_csv('./DataAnalysis/metrics_scores/OffBeatness_original_tot.csv')
IOI_InformationEntropy_original = pd.read_csv('./DataAnalysis/metrics_scores/IOI_InformationEntropy_original_tot.csv')
IOI_TallestBin_original = pd.read_csv('./DataAnalysis/metrics_scores/IOI_TallestBin_original_tot.csv')




# Split Toussaint metric scores in velocity modes
Toussaint_original_C = Toussaint_original.loc[Toussaint_original['Pattern'].str[-1] == 'C']
Toussaint_original_G = Toussaint_original.loc[Toussaint_original['Pattern'].str[-1] == 'G']
Toussaint_original_H = Toussaint_original.loc[Toussaint_original['Pattern'].str[-1] == 'H']
Toussaint_original_R = Toussaint_original.loc[Toussaint_original['Pattern'].str[-1] == 'R']

# Split Longuet-Higgins & Lee metric scores in velocity modes
LonguetHigginsLee_original_C = LonguetHigginsLee_original.loc[LonguetHigginsLee_original['Pattern'].str[-1] == 'C']
LonguetHigginsLee_original_G = LonguetHigginsLee_original.loc[LonguetHigginsLee_original['Pattern'].str[-1] == 'G']
LonguetHigginsLee_original_H = LonguetHigginsLee_original.loc[LonguetHigginsLee_original['Pattern'].str[-1] == 'H']
LonguetHigginsLee_original_R = LonguetHigginsLee_original.loc[LonguetHigginsLee_original['Pattern'].str[-1] == 'R']

# Split Pressing metric scores in velocity modes
Pressing_original_C = Pressing_original.loc[Pressing_original['Pattern'].str[-1] == 'C']
Pressing_original_G = Pressing_original.loc[Pressing_original['Pattern'].str[-1] == 'G']
Pressing_original_H = Pressing_original.loc[Pressing_original['Pattern'].str[-1] == 'H']
Pressing_original_R = Pressing_original.loc[Pressing_original['Pattern'].str[-1] == 'R']

# Split WNBD metric scores in velocity modes
WNBD_original_C = WNBD_original.loc[WNBD_original['Pattern'].str[-1] == 'C']
WNBD_original_G = WNBD_original.loc[WNBD_original['Pattern'].str[-1] == 'G']
WNBD_original_H = WNBD_original.loc[WNBD_original['Pattern'].str[-1] == 'H']
WNBD_original_R = WNBD_original.loc[WNBD_original['Pattern'].str[-1] == 'R']

# Split IOI Information Entropy metric scores in velocity modes
IOI_InformationEntropy_original_C = IOI_InformationEntropy_original.loc[IOI_InformationEntropy_original['Pattern'].str[-1]=='C']
IOI_InformationEntropy_original_G = IOI_InformationEntropy_original.loc[IOI_InformationEntropy_original['Pattern'].str[-1]=='G']
IOI_InformationEntropy_original_H = IOI_InformationEntropy_original.loc[IOI_InformationEntropy_original['Pattern'].str[-1]=='H']
IOI_InformationEntropy_original_R = IOI_InformationEntropy_original.loc[IOI_InformationEntropy_original['Pattern'].str[-1]=='R']

# Split IOI Tallest Bin metric scores in velocity modes
IOI_TallestBin_original_C = IOI_TallestBin_original.loc[IOI_TallestBin_original['Pattern'].str[-1]=='C']
IOI_TallestBin_original_G = IOI_TallestBin_original.loc[IOI_TallestBin_original['Pattern'].str[-1]=='G']
IOI_TallestBin_original_H = IOI_TallestBin_original.loc[IOI_TallestBin_original['Pattern'].str[-1]=='H']
IOI_TallestBin_original_R = IOI_TallestBin_original.loc[IOI_TallestBin_original['Pattern'].str[-1]=='R']

# Split Off-Beatness metric scores in velocity modes
OffBeatness_original_C = OffBeatness_original.loc[OffBeatness_original['Pattern'].str[-1] == 'C']
OffBeatness_original_G = OffBeatness_original.loc[OffBeatness_original['Pattern'].str[-1] == 'G']
OffBeatness_original_H = OffBeatness_original.loc[OffBeatness_original['Pattern'].str[-1] == 'H']
OffBeatness_original_R = OffBeatness_original.loc[OffBeatness_original['Pattern'].str[-1] == 'R']




# Save 24 df of metrics
Toussaint_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_C.csv", index=False)
Toussaint_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_G.csv", index=False)
Toussaint_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_H.csv", index=False)
Toussaint_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Toussaint/Toussaint_original_R.csv", index=False)
LonguetHigginsLee_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_C.csv", index=False)
LonguetHigginsLee_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_G.csv", index=False)
LonguetHigginsLee_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_H.csv", index=False)
LonguetHigginsLee_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Longuet-Higgins&Lee/LonguetHigginsLee_original_R.csv", index=False)
Pressing_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_C.csv", index=False)
Pressing_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_G.csv", index=False)
Pressing_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_H.csv", index=False)
Pressing_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Pressing/Pressing_original_R.csv", index=False)
WNBD_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_C.csv", index=False)
WNBD_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_G.csv", index=False)
WNBD_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_H.csv", index=False)
WNBD_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/WNBD/WNBD_original_R.csv", index=False)
IOI_InformationEntropy_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_C.csv", index=False)
IOI_InformationEntropy_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_G.csv", index=False)
IOI_InformationEntropy_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_H.csv", index=False)
IOI_InformationEntropy_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_InformationEntropy/IOI_InformationEntropy_original_R.csv", index=False)
IOI_TallestBin_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_C.csv", index=False)
IOI_TallestBin_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_G.csv", index=False)
IOI_TallestBin_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_H.csv", index=False)
IOI_TallestBin_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/IOI_TallestBin/IOI_TallestBin_original_R.csv", index=False)
OffBeatness_original_C.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_C.csv", index=False)
OffBeatness_original_G.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_G.csv", index=False)
OffBeatness_original_H.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_H.csv", index=False)
OffBeatness_original_R.to_csv("./DataAnalysis/metrics_scores/SPLITTED/Off-Beatness/OffBeatness_original_R.csv", index=False)
