## IMPORTS ##
import numpy as np
import pandas as pd
import os


## READ THE DATA ##
possible_paths = [
    'C:/Users/lenne/OneDrive/Documenten/Master of Statistics and Data Science/2023-2024/Master thesis/Thesis_Sofia_Lennert/new_data',
    'C:/Users/anaso/Desktop/SOFIA MENDES/KU Leuven/Master Thesis/Thesis_Sofia_Lennert/new_data'
]
file = 'merged_data.csv'
path = next((f'{path}/{file}' for path in possible_paths if os.path.exists(f'{path}/{file}')), None)
data = pd.read_csv(path)


## PREPROCESS THE DATA ##

## 1. Number of Relapses

# Assume missing means 0 relapses
data['NRELAP'] = data['NRELAP'].fillna(0)
# Bin the number of relapses into 0, 1, 2, 3 and 4+ 
def bin_column(value):
    if value in [0, 1, 2, 3]:
        return str(value)
    else:
        return '4+'
data['NRELAP'] = data['NRELAP'].apply(bin_column)

## 2. RAND36 and SF12 
## NOTE: for the binary indicator we can only keep one (before and after - because this will be the same)

# Create a new column 'RAND36_M-before' and fill it with values from 'SF12_M-before'
data['M_R36-SF12-before'] = data['RAND36_M-before'].fillna(data['SF12_M-before'])
# Create a new column 'RAND36_P-before' and fill it with values from 'SF12_P-before'
data['P_R36-SF12-before'] = data['RAND36_P-before'].fillna(data['SF12_P-before'])
# Create a new column 'R36-SF12-before' with a binary indicator
data['R36-SF12-before_Ind'] = data.apply(lambda row: 1 if pd.notna(row['RAND36_M-before']) else (0 if pd.notna(row['SF12_M-before']) else np.nan), axis=1)
# Drop the original columns if needed
data = data.drop(['SF12_P-before','SF12_M-before','RAND36_P-before','RAND36_M-before'], axis=1)

# Create a new column 'RAND36_M-after' and fill it with values from 'SF12_M-after'
data['M_R36-SF12-after'] = data['RAND36_M-after'].fillna(data['SF12_M-after'])
# Create a new column 'RAND36_P-after' and fill it with values from 'SF12_P-after'
data['P_R36-SF12-after'] = data['RAND36_P-after'].fillna(data['SF12_P-after'])
# Create a new column 'R36-SF12' with a binary indicator
data['R36-SF12-after_Ind'] = data.apply(lambda row: 1 if pd.notna(row['RAND36_M-after']) else (0 if pd.notna(row['SF12_M-after']) else np.nan), axis=1)
# Drop the original columns if needed
data = data.drop(['SF12_P-after','SF12_M-after','RAND36_P-after','RAND36_M-after'], axis=1)


## EXPORT THE DATA ## 

# Create folder 'new_data' if doesn't exist
folder_name = 'new_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# Specify the path for the CSV file
csv_file_path = os.path.join(folder_name, 'merged_data_modeling.csv')
# Save the DataFrame to CSV
data.to_csv(csv_file_path, index=False)

print("Complete: Dataframe for modeling has been obtained")