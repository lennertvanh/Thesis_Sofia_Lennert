# Imports
import pandas as pd
import numpy as np
import os

possible_paths = [
    'C:/Users/lenne/OneDrive/Documenten/Master of Statistics and Data Science/2023-2024/Master thesis/MSOAC Placebo Data',
    'C:/Users/anaso/Desktop/SOFIA MENDES/KU Leuven/Master Thesis/MSOAC Placebo dataset/csv files'
]

################################################
############### Functional Tests ###############
################################################

# Load the data
file_name = 'ft.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
ftests = pd.read_csv(file_path)

# Filter columns based on missing percentage
missing_percentage_ftests = (ftests.isnull().sum() / len(ftests)) * 100
missing_ftests = pd.DataFrame({'Column Name': missing_percentage_ftests.index, 'Missing Percentage': missing_percentage_ftests.values})

# Set the threshold for missing percentage
threshold = 90
columns_to_drop = missing_ftests[missing_ftests['Missing Percentage'] >= threshold]['Column Name']

# Drop columns from the DataFrame
ftests = ftests.drop(columns=columns_to_drop)

# Remove redundant columns
ftests = ftests.drop(columns=['STUDYID', 'DOMAIN', 'FTTESTCD', 'FTORRES', 'FTORRESU', 'FTSTRESU'])

# Define the 'FTTEST' values with a numeric outcome
num_FTTEST_values = [
    'T25FW1-Time to Complete 25-Foot Walk',
    'NHPT01-Time to Complete 9-Hole Peg Test',
    'PASAT1-Total Correct',
    'SDMT01-Total Score'
]
filtered_rows = ftests[ftests['FTTEST'].isin(num_FTTEST_values)]

# Update FTSTRESN for rows where FTCAT is T25FW and FTSTRESN > 180
filtered_rows.loc[(filtered_rows['FTCAT'] == 'T25FW') & (filtered_rows['FTSTRESN'] > 180), 'FTSTRESN'] = 180

# Update FTSTRESN for rows where FTCAT is NHPT and FTSTRESN > 300
filtered_rows.loc[(filtered_rows['FTCAT'] == 'NHPT') & (filtered_rows['FTSTRESN'] > 300), 'FTSTRESN'] = 300

conditions = [
    (filtered_rows['FTDY'] <= 1),
    ((filtered_rows['FTDY'] > 1) & (filtered_rows['FTDY'] <= 730)),
    (filtered_rows['FTDY'] > 730)
]
# Define corresponding values for each condition
values = ['before', '2y', 'after_2y']

# Create the new column "FT_PERIOD"
filtered_rows['FT_PERIOD'] = np.select(conditions, values, default='NaN')
filtered_rows = filtered_rows.dropna(subset=['FTDY']) #Drop observations for which we don't have time of test - note that ~700 patients don't have the time recorded so they will only have NAs

# Replace 'PASAT' with 'PASAT_2S' when 'FTCAT' is 'PASAT' and 'FTSCAT' is 2
filtered_rows.loc[(filtered_rows['FTCAT'] == 'PASAT') & (filtered_rows['FTSCAT'] == '2 SECONDS'), 'FTCAT'] = 'PASAT_2s'

# Replace 'PASAT' with 'PASAT_3S' when 'FTCAT' is 'PASAT' and 'FTSCAT' is 3
filtered_rows.loc[(filtered_rows['FTCAT'] == 'PASAT') & (filtered_rows['FTSCAT'] == '3 SECONDS'), 'FTCAT'] = 'PASAT_3s'

med_df = filtered_rows.groupby(['USUBJID', 'FTCAT', 'FT_PERIOD']).agg({
    'FTSTRESN': np.median
}).reset_index()

# Pivot the table
pivot_df = med_df.pivot_table(index='USUBJID', columns=['FTCAT', 'FT_PERIOD'], values='FTSTRESN').reset_index()

# Flatten the multi-level column index
pivot_df.columns = [f'{cat}-{period}' if period != '' else cat for cat, period in pivot_df.columns]

# Define the desired order of FT_period for each category
ft_period_order = ['before', '2y', 'after_2y']

# Function to sort columns based on category and FT_period order
def custom_sort(column):
    if '-' in column:
        cat, period = column.split('-')
        return (cat, ft_period_order.index(period))
    else:
        return (column, 0)

# Sort the columns based on the custom order
column_order = ['USUBJID'] + sorted(
    [col for col in pivot_df.columns if col != 'USUBJID'],
    key=custom_sort
)

# Merge the new dataframe with the original dataframe on 'USUBJID'
result_df = pd.merge(filtered_rows[['USUBJID']], pivot_df, on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient
result_df = result_df.drop_duplicates(subset='USUBJID')

# Reorder the columns based on the desired order
result_df = result_df[column_order]

# Define the 'FTTEST' values with a categorical outcome
cat_FTTEST_values = [
    'T25FW1-Complete Two Successful Trials',
    'T25FW1-More Than Two Attempts',
    'NHPT01-More Than Two Attempts',
    'PASAT1-More Than One Attempt',
]
filtered_rows = ftests[ftests['FTTEST'].isin(cat_FTTEST_values)]
filtered_rows = filtered_rows[filtered_rows['FTTEST'] != 'T25FW1-Complete Two Successful Trials'] #always yes so don't use

conditions = [
    (filtered_rows['FTDY'] <= 1),
    (filtered_rows['FTDY'] > 1) 
]

# Define corresponding values for each condition
values = ['before', 'after']

# Create the new column "FT_PERIOD"
filtered_rows['FT_PERIOD'] = np.select(conditions, values, default='NaN')
filtered_rows = filtered_rows.dropna(subset=['FTDY']) 
columns_to_drop = ['FTGRPID', 'FTSEQ', 'FTSCAT', 'FTREPNUM', 'VISIT', 'VISITDY']
filtered_rows = filtered_rows.drop(columns=columns_to_drop)

filtered_rows['FTCAT'] = filtered_rows['FTCAT'].str[0]

# Create new columns based on conditions
for cat in filtered_rows['FTCAT'].unique():
    for period in filtered_rows['FT_PERIOD'].unique():
        col_name = f'{cat}-{period}'
        filtered_rows[col_name] = (
            filtered_rows[(filtered_rows['FTCAT'] == cat) & (filtered_rows['FT_PERIOD'] == period)]
            .groupby('USUBJID')['FTSTRESC']
            .transform(lambda x: 1 if 'Y' in x.values else 0)
        )

# Extract relevant columns
selected_columns = [f'{cat}-{period}' for cat in filtered_rows['FTCAT'].unique() for period in filtered_rows['FT_PERIOD'].unique()]

# Select only the relevant columns
result_df2 = filtered_rows[['USUBJID'] + selected_columns]

# Group by 'USUBJID' and aggregate using the maximum value for each column
result_df2 = result_df2.groupby('USUBJID').max().reset_index()

# Merge with result_df
merged_df = pd.merge(result_df, result_df2, on='USUBJID', how='outer')

# Export data
folder_name = 'new_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Specify the path for the CSV file
csv_file_path = os.path.join(folder_name, 'FT_agg.csv')

# Save the DataFrame to CSV
merged_df.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("FT_agg.csv has been created in the folder new_data")



###############################################
########### Ophthalmic Examinations ###########
###############################################

# Load the data
file_name = 'oe.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
opt = pd.read_csv(file_path)

missing_percentage_opt = (opt.isnull().sum() / len(opt)) * 100
missing_opt = pd.DataFrame({'Column Name': missing_percentage_opt.index, 'Missing Percentage': missing_percentage_opt.values})

# Set the threshold for missing percentage
threshold = 90

# Filter columns based on missing percentage
columns_to_drop = missing_opt[missing_opt['Missing Percentage'] >= threshold]['Column Name']

# Drop columns from the DataFrame
opt = opt.drop(columns=columns_to_drop)

# Remove redundant columns
opt = opt.drop(columns=['STUDYID', 'DOMAIN', 'OETESTCD', 'OELOC', 'OECAT', 'OEORRES'])

# SNELLEN EQUIVALENT SCORE #
SES_df = opt[opt['OETEST'] == 'Snellen Equivalent Score'].copy()  # Create a copy to avoid the warning

conditions = [
    (SES_df['OEDY'] <= 1),
    (SES_df['OEDY'] > 1)
]

# Define corresponding values for each condition
values = ['before', 'after']

# Create the new column "FT_PERIOD"
SES_df['OE_PERIOD'] = np.select(conditions, values, default='NaN')

SES_df['OESTRESC'] = SES_df['OESTRESC'].replace(0.2, 30)
SES_df['OESTRESN'] = SES_df['OESTRESN'].replace(0.2, 30)

# Function to fill missing values in OESTRESN based on OESTRESC
def fill_missing_oestresn(row):
    if pd.isna(row['OESTRESN']):
        # Extract the second number after "6/"
        oestresc_values = str(row['OESTRESC']).split('/')
        if len(oestresc_values) == 2:
            second_number = oestresc_values[1].strip()
            try:
                # Try converting the second number to float
                return float(second_number)
            except ValueError:
                # Handle the case where conversion to float fails
                return None
    return row['OESTRESN']

# Apply the function to fill missing values in OESTRESN
SES_df['OESTRESN'] = SES_df.apply(fill_missing_oestresn, axis=1)

# Pivot SES_df to create a new DataFrame with median values for each period and patient
grouped_df = SES_df.pivot_table(values='OESTRESN', index='USUBJID', columns='OE_PERIOD', aggfunc='median', fill_value=None).reset_index()

# Rename the columns to SES_after and SES_before
grouped_df.columns = ['USUBJID', 'SES_after', 'SES_before']

# Merge the new DataFrame with the original DataFrame based on 'USUBJID'
result_SES = pd.merge(SES_df[['USUBJID']], grouped_df, on='USUBJID', how='left')

# Drop duplicate rows to retain unique rows per patient and period
result_SES = result_SES.drop_duplicates(subset=['USUBJID'])

# Change the values in the SES_after and SES_before columns to 6 divided by the original values
result_SES[['SES_after', 'SES_before']] = 6 / result_SES[['SES_after', 'SES_before']]

# DECIMAL SCORE #
DC_rows = opt[opt['OETEST'] == 'Decimal Score']

result_DC = DC_rows.pivot_table(index='USUBJID', columns='OELAT', values='OESTRESN', aggfunc='min').reset_index()
result_DC.columns = ['USUBJID', 'DS_Left', 'DS_Right']

# Create a new column 'DS' with the minimum value between DS_L and DS_R
result_DC['DS'] = result_DC[['DS_Left', 'DS_Right']].min(axis=1)

# Update DS_L and DS_R based on the conditions
result_DC['DS_L'] = np.where(result_DC['DS_Left'] <= result_DC['DS_Right'], 1, 0)
result_DC['DS_R'] = np.where(result_DC['DS_Right'] <= result_DC['DS_Left'], 1, 0)

# Drop DS_Left and DS_Right columns
result_DC = result_DC.drop(['DS_Left', 'DS_Right'], axis=1)

# SLOAN LETTER EYE CHART #
SLEC_rows = opt[(opt['OETEST'] == 'Number of Letters Correct') & (opt['OEMETHOD'] == 'SLOAN LETTER EYE CHART 1.25%')].copy()
SLEC_rows = SLEC_rows[SLEC_rows['OELAT'] == 'B']

conditions = [
    (SLEC_rows['OEDY'] <= 1),
    (SLEC_rows['OEDY'] > 1)
]

# Define corresponding values for each condition
values = ['before', 'after']

# Create the new column "FT_PERIOD"
SLEC_rows['OE_PERIOD'] = np.select(conditions, values, default='NaN')
SLEC_rows = SLEC_rows.dropna(subset=['OEDY']) #Drop observations for which we don't have time of test!

grouped_df = SLEC_rows.pivot_table(values='OESTRESN', index='USUBJID', columns='OE_PERIOD', aggfunc='median', fill_value=None).reset_index() #min

# Rename the columns 
grouped_df.columns = ['USUBJID'] + [f"SLEC_{period}" for period in grouped_df.columns[1:]]

# Merge the new DataFrame with the original DataFrame on 'USUBJID'
result_SLEC = pd.merge(SLEC_rows[['USUBJID']], grouped_df, on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient and period
result_SLEC = result_SLEC.drop_duplicates(subset=['USUBJID'])

# Desired column order
desired_order = ['USUBJID', 'SLEC_before', 'SLEC_after']
result_SLEC = result_SLEC[desired_order]

# VISUAL ACUITY ASSESSMENT #
VAA_rows = opt[opt['OETEST'] == 'Visual Acuity Assessment']

result_VAA = VAA_rows.groupby('USUBJID')['OESTRESC'].apply(lambda x: 1 if 'ABNORMAL' in x.values else 0).reset_index()
result_VAA.columns = ['USUBJID', 'VAA']

# Merge individual dataframes from OE #
# Extract unique USUBJID values from the opt DataFrame
unique_usubjid = opt['USUBJID'].unique()

# Initialize an empty DataFrame with the unique USUBJID values
final_merged_df = pd.DataFrame({'USUBJID': unique_usubjid})

# List of result DataFrames
result_dfs = [result_DC, result_SLEC, result_SES, result_VAA]

# Iterate through result DataFrames and perform left merges
for result_df in result_dfs:
    final_merged_df = pd.merge(final_merged_df, result_df, on='USUBJID', how='left')

#Export data
folder_name = 'new_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Specify the path for the CSV file
csv_file_path = os.path.join(folder_name, 'OE_agg.csv')

# Save the DataFrame to CSV
final_merged_df.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("OE_agg.csv has been created in the folder new_data")



