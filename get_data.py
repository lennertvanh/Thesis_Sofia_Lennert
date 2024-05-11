# Imports
import pandas as pd
import numpy as np
import os

possible_paths = [
    'C:/Users/lenne/OneDrive/Documenten/Master of Statistics and Data Science/2023-2024/Master thesis/MSOAC Placebo Data',
    'C:/Users/anaso/Desktop/SOFIA MENDES/KU Leuven/Master Thesis/MSOAC Placebo dataset/csv files'
]

################################################
################# Demographics #################
################################################

# Load the data
file_name = 'dm.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
demographics = pd.read_csv(file_path)

# Drop initial columns
columns_to_drop = ['STUDYID','DOMAIN','SUBJID','RFSTDTC','RFENDTC','DTHDTC','DTHFL','SITEID','INVID','INVNAM','BRTHDTC','AGEU','ETHNIC','ARMCD','ARM','ACTARMCD','ACTARM','DMDTC','DMDY','DMENDY','DMDTC_TS','RFENDTC_TS','RFSTDTC_TS']
demographics = demographics.drop(columns_to_drop, axis=1)

# Convert RACE to 2 categories
demographics['RACE'] = demographics['RACE'].apply(lambda x: 'WHITE' if x == 'WHITE' else 'NON-WHITE' if pd.notnull(x) else np.nan)

# Convert COUNTRY to CONTINENT
continent_mapping = {
    'USA': 'North America',
    'POL': 'Europe',
    'CAN': 'North America',
    'UKR': 'Europe',
    'CZE': 'Europe',
    'IND': 'Asia',
    'RUS': 'Eurasia',
    'SRB': 'Europe',
    'DEU': 'Europe',
    'GBR': 'Europe',
    'NLD': 'Europe',
    'BGR': 'Europe',
    'HUN': 'Europe',
    'ROU': 'Europe',
    'GRC': 'Europe',
    'FRA': 'Europe',
    'NZL': 'Oceania',
    'BEL': 'Europe',
    'SWE': 'Europe',
    'MEX': 'North America',
    'EST': 'Europe',
    'ESP': 'Europe',
    'PER': 'South America',
    'GEO': 'Asia',
    'AUS': 'Oceania',
    'ISR': 'Asia',
    'CHE': 'Europe',
    'HRV': 'Europe',
    'TUR': 'Eurasia',
    'COL': 'South America',
    'LVA': 'Europe',
    'FIN': 'Europe',
    'IRL': 'Europe',
    'DNK': 'Europe',
    'CHL': 'South America'
}

demographics['CONTINENT'] = demographics['COUNTRY'].map(continent_mapping)
demographics['CONTINENT'] = demographics['CONTINENT'].str.upper()
demographics = demographics.drop('COUNTRY', axis=1)

# Export data to DM_agg.csv
folder_name = 'new_data'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

csv_file_path = os.path.join(folder_name, 'DM_agg.csv')
demographics.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("DM_agg.csv has been created in the folder new_data")



################################################
############### Clinical Events ################
################################################

# Load data
file_name = 'ce.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
clinical_events = pd.read_csv(file_path)
clinical_events = clinical_events.sort_values(by=['USUBJID','CESEQ'], ascending=True)

# Drop some initial columns
missing_percentage_ce = (clinical_events.isnull().sum() / len(clinical_events)) * 100
missing_clinical_events = pd.DataFrame({'Column Name': missing_percentage_ce.index, 'Missing Percentage': missing_percentage_ce.values})
missing_clinical_events = missing_clinical_events.sort_values(by='Missing Percentage', ascending=False)

columns_to_drop = missing_clinical_events[missing_clinical_events['Missing Percentage'] > 85]['Column Name'].tolist()
additional_columns_to_drop = ['STUDYID', 'DOMAIN']  # Add your column names here
columns_to_drop.extend(additional_columns_to_drop)
clinical_events.drop(columns=columns_to_drop, inplace=True)

# Remove cases where CEDECOD is "MULTIPLE SCLEROSIS"
ms_entries = ['Multiple Sclerosis', 'MULTIPLE SCLEROSIS', 'MULTIPLE SCLEROSIS AGGRAVATED', 'PROGRESSION OF MULTIPLE SCLEROSIS', 'MS-LIKE SYNDROME']
clinical_events = clinical_events[~clinical_events['CEMODIFY'].isin(ms_entries)]

# New set of columns to drop
columns_to_drop_2 = ['VISIT','VISITNUM','CEDECOD','CEENDY','CEDY','CEPRESP','CEBODSYS','MIDS','CESEQ']
clinical_events.drop(columns=columns_to_drop_2, inplace=True)

# Count total of relapses
clinical_events['TOTRELAP'] = clinical_events.groupby('USUBJID')['USUBJID'].transform('count')

# Obtain static dataframe
def aggregate_and_remove_duplicates(series):
    unique_values = set(series.dropna())  # Drop NaN and convert to set
    return '; '.join(map(str, unique_values))

clinical_events_aggregated = clinical_events.groupby('USUBJID').agg(aggregate_and_remove_duplicates).reset_index()
clinical_events_aggregated.replace('', np.nan, inplace=True)
clinical_events_aggregated.drop(columns=['CETERM','CEMODIFY','CESTDY','CESER','CEOCCUR'], inplace=True)

# Store only high relapse severity
def replace_words(row):
    if isinstance(row, str):
        if 'SEVERE' in row:
            return 'SEVERE'
        elif 'MODERATE' in row:
            return 'MODERATE'
        elif 'MILD' in row:
            return 'MILD'
        else:
            return row
    else:
        return row

clinical_events_aggregated['CESEV'] = clinical_events_aggregated['CESEV'].apply(replace_words)

# Clean Treatment variable
clinical_events_aggregated['CECONTRT'] = clinical_events_aggregated['CECONTRT'].replace('N; Y', 'Y')
clinical_events_aggregated['CECONTRT'] = clinical_events_aggregated['CECONTRT'].replace('Y; N', 'Y')

# Convert total relapses to numeric
clinical_events_aggregated['TOTRELAP'] = pd.to_numeric(clinical_events_aggregated['TOTRELAP'], errors='coerce')

# Export dataframe to CSV
folder_name = 'new_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

csv_file_path = os.path.join(folder_name, 'CE_agg.csv')
clinical_events_aggregated.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("CE_agg.csv has been created in the folder new_data")



################################################
########## Subject Disease Milestones ##########
################################################

# Load data
file_name = 'sm.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
milestones = pd.read_csv(file_path)
milestones = milestones.sort_values(by=['USUBJID', 'SMSEQ'], ascending=True)


# Drop some initial columns
columns_to_drop = ['STUDYID','DOMAIN','SMSEQ','SMENRF','MIDSTYPE']
milestones = milestones.drop(columns_to_drop, axis=1)

# Count number of relapses for each patient
milestones['Number'] = milestones['MIDS'].str.extract(r'(\d+)').astype(int)
milestones['NRELAP'] = milestones.groupby('USUBJID')['Number'].transform('max')
milestones = milestones.drop(['MIDS', 'Number', 'SMENDY'], axis=1)

# Keep the first row for each patient
milestones_final = milestones.groupby('USUBJID').first().reset_index()

# Export data to csv
folder_name = 'new_data'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

csv_file_path = os.path.join(folder_name, 'SM_agg.csv')
milestones_final.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("SM_agg.csv has been created in the folder new_data")



################################################
############### Medical History ################
################################################
print("Warning: MH dataset might take some time to run")

# Load data
file_name = 'mh.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
medical_history = pd.read_csv(file_path)

# Drop initial columns
missing_percentage_mh = (medical_history.isnull().sum() / len(medical_history)) * 100
missing_medical_history = pd.DataFrame({'Column Name': missing_percentage_mh.index, 'Missing Percentage': missing_percentage_mh.values})
missing_medical_history = missing_medical_history.sort_values(by='Missing Percentage', ascending=False)
columns_to_drop = missing_medical_history[missing_medical_history['Missing Percentage'] > 95]['Column Name'].tolist()
additional_columns_to_drop = ['STUDYID', 'DOMAIN', 'MHENTPT','MHLLT','MHENRTPT','MHHLGT','MHEVLINT'] #'MHSTDY',
columns_to_drop.extend(additional_columns_to_drop)
medical_history.drop(columns=columns_to_drop, inplace=True)

# Create indicators for 4 medical factors
# CARDIOVASCULAR
terms_to_check = ['CARDIOVASCULAR']# or 'ALLERGY', 'ALLERGIES'
pattern = '|'.join(terms_to_check)

for index, row in medical_history.iterrows():
    found_term = False  # Flag to track if either term is found in any column
    for column in ['MHTERM', 'MHDECOD', 'MHCAT', 'MHBODSYS']:
        if pd.notna(row[column]) and pd.Series(row[column]).str.contains(pattern, case=False, regex=True).any():
            found_term = True
            break  # Break the loop when a match is found in any column

    medical_history.at[index, 'MHNEWCAT'] = 'CARDIO' if found_term else pd.NA


# URINARY
terms_to_check = ['BLADDER', 'URINARY','URINATION']#'GENITOURINARY',
pattern = '|'.join(terms_to_check)

for index, row in medical_history.iterrows():
    found_term = False  # Flag to track if either term is found in any column
    for column in ['MHTERM', 'MHDECOD', 'MHCAT', 'MHBODSYS']:
        if pd.notna(row[column]) and pd.Series(row[column]).str.contains(pattern, case=False, regex=True).any():
            found_term = True
            break  # Break the loop when a match is found in any column

    if pd.isna(row['MHNEWCAT']):
        medical_history.at[index, 'MHNEWCAT'] = 'URINARY' if found_term else pd.NA


# MUSCULOSKELETAL
terms_to_check = ['MUSCULOSKELETAL']# or MUSCLE
pattern = '|'.join(terms_to_check)

for index, row in medical_history.iterrows():
    found_term = False  # Flag to track if either term is found in any column
    for column in ['MHTERM', 'MHDECOD', 'MHCAT', 'MHBODSYS','MHSOC']:
        if pd.notna(row[column]) and pd.Series(row[column]).str.contains(pattern, case=False, regex=True).any():
            found_term = True
            break  # Break the loop when a match is found in any column

    if pd.isna(row['MHNEWCAT']):
        medical_history.at[index, 'MHNEWCAT'] = 'MUSCKELET' if found_term else pd.NA


# FATIGUE
terms_to_check = ['FATIGUE']
pattern = '|'.join(terms_to_check)

for index, row in medical_history.iterrows():
    found_term = False  # Flag to track if either term is found in any column
    for column in ['MHTERM', 'MHDECOD', 'MHCAT', 'MHBODSYS','MHSOC']:
        if pd.notna(row[column]) and pd.Series(row[column]).str.contains(pattern, case=False, regex=True).any():
            found_term = True
            break  # Break the loop when a match is found in any column

    if pd.isna(row['MHNEWCAT']):
        medical_history.at[index, 'MHNEWCAT'] = 'FATIGUE' if found_term else pd.NA


# New set of columns to drop
columns_to_drop_2 = ['MHSEQ', 'VISIT', 'VISITNUM', 'MHSCAT', 'MHENRF', 'MHDY', 'MHSTDY', 'MHOCCUR', 'MHPRESP', 'MHSEV'] #, 
medical_history.drop(columns=columns_to_drop_2, inplace=True)

# Remove rows where MHDIAG is 'MS DIAGNOSIS'
medical_history['MHDIAGN'] = medical_history.loc[medical_history['MHCAT'].isin(['DIAGNOSIS', 'PRIMARY DIAGNOSIS']), 'MHTERM']

# Clean MHDIAG
medical_history['MHDIAGN'] = medical_history['MHDIAGN'].replace('RELAPSING-REMITTING', 'RRMS')
medical_history['MHDIAGN'] = medical_history['MHDIAGN'].replace('PRIMARY-PROGRESSIVE', 'PPMS')
medical_history['MHDIAGN'] = medical_history['MHDIAGN'].replace({'SPMS DIAGNOSIS','SECONDARY-PROGRESSIVE'}, 'SPMS')
medical_history['MHDIAGN'] = medical_history['MHDIAGN'].replace({'MS DIAGNOSIS CONFIRMED BY MRI','MS DIAGNOSIS','SUSPECTED ONSET OF MS','PROGRESSIVE RELAPSING'}, np.nan) #'MS'

# Obtain static dataframe
def aggregate_and_remove_duplicates(series):
    unique_values = set(series.dropna())  # Drop NaN and convert to set
    return '; '.join(map(str, unique_values))

medical_history_aggregated = medical_history.groupby('USUBJID').agg(aggregate_and_remove_duplicates).reset_index()
medical_history_aggregated.replace('', np.nan, inplace=True)
medical_history_aggregated['MHCAT'] = medical_history_aggregated['MHCAT'].replace('PRIMARY DIAGNOSIS', 'DIAGNOSIS')
medical_history_aggregated.drop(columns=['MHTERM','MHDECOD', 'MHCAT','MHBODSYS','MHSOC'], inplace=True)


# Create binary indicators for the 4 medical factors
def has_cardio(text):
    return 1 if pd.notna(text) and 'CARDIO' in text else 0

def has_urinary(text):
    return 1 if pd.notna(text) and 'URINARY' in text else 0

def has_muscles(text):
    return 1 if pd.notna(text) and 'MUSCKELET' in text else 0

def has_fatigue(text):
    return 1 if pd.notna(text) and 'FATIGUE' in text else 0

medical_history_aggregated['CARDIO'] = medical_history_aggregated['MHNEWCAT'].apply(has_cardio)
medical_history_aggregated['URINARY'] = medical_history_aggregated['MHNEWCAT'].apply(has_urinary)
medical_history_aggregated['MUSCKELET'] = medical_history_aggregated['MHNEWCAT'].apply(has_muscles)
medical_history_aggregated['FATIGUE'] = medical_history_aggregated['MHNEWCAT'].apply(has_fatigue)

medical_history_aggregated.drop(columns=['MHNEWCAT'], inplace=True)

# Keep 'Y' for MHCONTRT
medical_history_aggregated['MHCONTRT'] = medical_history_aggregated['MHCONTRT'].replace('N; Y', 'Y')
medical_history_aggregated['MHCONTRT'] = medical_history_aggregated['MHCONTRT'].replace('Y; N', 'Y')

# Export data to csv
folder_name = 'new_data'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

csv_file_path = os.path.join(folder_name, 'MH_agg.csv')
medical_history_aggregated.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("OE_agg.csv has been created in the folder new_data")


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
ftests.sort_values(by=['USUBJID', 'FTSEQ'], inplace=True)

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
opt.sort_values(by=['USUBJID', 'OESEQ'], inplace=True)

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
SLEC_rows = SLEC_rows[SLEC_rows['OELAT'] == 'BILATERAL']

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
final_merged_df = final_merged_df.sort_values(by='USUBJID')

# List of result DataFrames  # Excluse decimal score because not sure what it means
result_dfs = [result_SLEC, result_SES] #, result_VAA

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



################################################
################ Questionnaires ################
################################################

# Load data
file_name = 'qs.csv'
file_path = next(f'{path}/{file_name}' for path in possible_paths if os.path.exists(f'{path}/{file_name}'))
qs = pd.read_csv(file_path)

missing_percentage_qs = (qs.isnull().sum() / len(qs)) * 100
missing_qs = pd.DataFrame({'Column Name': missing_percentage_qs.index, 'Missing Percentage': missing_percentage_qs.values})

# Set the threshold for missing percentage
threshold = 80

# Filter columns based on missing percentage
columns_to_drop = missing_qs[missing_qs['Missing Percentage'] >= threshold]['Column Name']

# Drop columns from the DataFrame
qs = qs.drop(columns=columns_to_drop)

# Remove redundant columns
qs = qs.drop(columns=['STUDYID', 'DOMAIN']) #, 'QSTESTCD'
qs.sort_values(by=['USUBJID', 'QSSEQ'], inplace=True)

# BDI II #
BDI_rows = qs[qs['QSCAT'] == 'BDI-II']
BDI_rows = BDI_rows.copy()
BDI_rows.drop_duplicates(subset=['USUBJID', 'QSDY', 'QSTEST', 'QSSTRESN'], inplace=True)

# Convert QSSTRESN column to numeric in case it's not already
BDI_rows['QSSTRESN'] = pd.to_numeric(BDI_rows['QSSTRESN'], errors='coerce')

# Find the index of the maximum QSSTRESN within each group
max_idx = BDI_rows.groupby(['USUBJID', 'QSDY', 'QSTEST'])['QSSTRESN'].idxmax()

# Filter the DataFrame using the identified indices
BDI_rows = BDI_rows.loc[max_idx]

result_table = BDI_rows.groupby(['USUBJID', 'QSDY']).size().reset_index(name='Count')

# Step 1: Filter BDI_rows based on QSTEST
filtered_rows = BDI_rows[BDI_rows['QSTEST'] == 'BDI01-BDI Total Score'].copy()

# Step 2: Merge with result_table to get the 'Count' for each unique combination of 'USUBJID' and 'QSDY'
merged_df = pd.merge(filtered_rows, result_table[['USUBJID', 'QSDY', 'Count']], on=['USUBJID', 'QSDY'], how='left')

# Step 3: Calculate the new value for QSSTRESN
merged_df['QSSTRESN'] = merged_df['QSSTRESN'] / (3 * (merged_df['Count'] - 1))

# Step 4: Drop the 'Count' column 
merged_df.drop(columns=['Count'], inplace=True)

# Step 5: Merge the modified rows back into the original DataFrame
BDI_rows = pd.merge(BDI_rows, merged_df[['USUBJID', 'QSDY', 'QSSTRESN']], on=['USUBJID', 'QSDY'], how='left')

columns_to_drop = ['QSSEQ', 'QSSCAT', 'QSORRES', 'QSSTRESC', 'VISITNUM', 'QSEVLINT', 'QSSTRESN_x']
BDI_rows = BDI_rows[BDI_rows['QSTEST'] == 'BDI01-BDI Total Score'].drop(columns=columns_to_drop)
BDI_rows.rename(columns={'QSSTRESN_y': 'QSSTRESN'}, inplace=True)
BDI_rows = BDI_rows.sort_values(by='USUBJID')

conditions = [
    (BDI_rows['QSDY'] <= 1),
    (BDI_rows['QSDY'] > 1) 
]
# Define corresponding values for each condition
values = ['BDI-before', 'BDI-after']

# Create the new column "FT_PERIOD"
BDI_rows['QS_PERIOD'] = np.select(conditions, values, default='NaN')
BDI_rows = BDI_rows.dropna(subset=['QSDY']) #Drop observations for which we don't have time of test 

# Calculate the median of QSSTRESN for each 'QS_PERIOD' for each patient
median_df = BDI_rows.groupby(['USUBJID', 'QS_PERIOD']).agg({
    'QSSTRESN': 'median'
}).reset_index()

# Pivot the table
pivot_df = median_df.pivot_table(index='USUBJID', columns=['QS_PERIOD'], values='QSSTRESN').reset_index()

# Remove the name of the index and columns
pivot_df.index.name = None
pivot_df.columns.name = None

# Reorganize columns
desired_order = ['USUBJID', 'BDI-before', 'BDI-after']
result_BDI = pivot_df[desired_order]

# EDSS #
EDSS_rows = qs[qs['QSCAT'] == 'EDSS']
EDSS_df = EDSS_rows.copy()  # Create a copy to avoid the warning

conditions = [
    (EDSS_df['QSDY'] <= 1),
    ((EDSS_df['QSDY'] > 1) & (EDSS_df['QSDY'] <= 730)), 
    ((EDSS_df['QSDY'] > 730))
]
# Define corresponding values for each condition
values = ['before', '2y', 'after_2y'] 

# Create the new column "FT_PERIOD"
EDSS_df['QS_PERIOD'] = np.select(conditions, values, default='NaN')
EDSS_df = EDSS_df.dropna(subset=['QSDY']) #Drop observations for which we don't have time of test

# median EDSS for each period
grouped_df = EDSS_df.pivot_table(values='QSSTRESN', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()

# Rename the columns 
grouped_df.columns = ['USUBJID'] + [f"EDSS-{period}" for period in grouped_df.columns[1:]]

# Merge the new DataFrame with the original DataFrame on 'USUBJID'
result_EDSS = pd.merge(EDSS_df[['USUBJID']], grouped_df, on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient and period
result_EDSS = result_EDSS.drop_duplicates(subset=['USUBJID'])

# Reorganize columns
desired_order = ['USUBJID', 'EDSS-before', 'EDSS-2y', 'EDSS-after_2y']
result_EDSS = result_EDSS[desired_order]

# KFSS #
KFSS_qs = qs[qs['QSCAT'] == 'KFSS']
KFSS_qs = KFSS_qs.drop(columns=['QSSEQ','VISITNUM','VISIT'])
KFSS_qs = KFSS_qs.dropna(subset=['QSSTRESN'])

# Group data by patient ID and count total entries and missing values in column A
missing_data = KFSS_qs.groupby('USUBJID', as_index=False)['QSDY'].agg(total_entries='count', missing_values=lambda x: x.isnull().sum())

# Calculate percentage of missing values for each patient ID
missing_data['percentage_missing'] = (missing_data['missing_values'] / missing_data['total_entries']) * 100

usubjid_inf_percentage_missing = missing_data[missing_data['percentage_missing'] == np.inf]['USUBJID'].tolist()
KFSS_qs = KFSS_qs[~KFSS_qs['USUBJID'].isin(usubjid_inf_percentage_missing)]
KFSS_qs = KFSS_qs.drop(columns=['QSEVLINT'])

KFSS_qs = KFSS_qs[KFSS_qs['QSTEST'] != 'KFSS1-Other Functions']

#put as NA rows with values 9 or 99
KFSS_qs['QSSTRESN'] = KFSS_qs['QSSTRESN'].replace([9, 99], pd.NA)

KFSS_qs_1=KFSS_qs.copy()
KFSS_qs_2=KFSS_qs.copy()

KFSS_qs_1['QSSTRESN'] = KFSS_qs_1.groupby(['USUBJID', 'QSTEST', 'QSDY'])['QSSTRESN'].transform('mean')
KFSS_qs_1.drop_duplicates(subset=['USUBJID', 'QSTEST', 'QSDY'	, 'QSSTRESN'], inplace=True)

def set_scoremax(row):
    if row['QSTEST'] in ['KFSS1-Cerebellar Functions', 'KFSS1-Brain Stem Functions', 'KFSS1-Cerebral or Mental Functions']:
        return 5
    #elif row['QSTEST'] in ['KFSS1-Other Functions']:
        #return 1
    else:
        return 6

# Apply the function row-wise to set the values in column B
KFSS_qs_1['SCOREMAX'] = KFSS_qs_1.apply(set_scoremax, axis=1)
KFSS_qs_1['QSPERC'] = KFSS_qs_1['QSSTRESN'] / KFSS_qs_1['SCOREMAX'] 
KFSS_qs_1 = KFSS_qs_1.drop(columns=['QSCAT', 'QSSCAT', 'QSORRES', 'QSSTRESC', 'QSSTRESN', 'SCOREMAX'])

KFSS_qs_1 = KFSS_qs_1.copy()  # Create a copy to avoid the warning
conditions = [
    (KFSS_qs_1['QSDY'] <= 1),
    ((KFSS_qs_1['QSDY'] > 1) & (KFSS_qs_1['QSDY'] <= 730)),
    #((KFSS_qs['QSDY'] > 365) & (KFSS_qs['QSDY'] <= 730)),
    ((KFSS_qs_1['QSDY'] > 730)) #& (KFSS_qs['QSDY'] <= 1095)),
    #((KFSS_qs['QSDY'] > 1095) & (KFSS_qs['QSDY'] <= 1460)) 
]
# Define corresponding values for each condition
values = ['before', '2y', 'after_2y'] # , '4y' - if i use this i have 93% missing in the time

# Create the new column "FT_PERIOD"
KFSS_qs_1['QS_PERIOD'] = np.select(conditions, values, default='NaN')
KFSS_qs_1 = KFSS_qs_1.dropna(subset=['QSDY']) #Drop observations for which we don't have time of test

categories = ['KFSS1-Sensory Functions',
              'KFSS1-Brain Stem Functions',
              'KFSS1-Bowel and Bladder Functions',
              'KFSS1-Pyramidal Functions',
              'KFSS1-Cerebral or Mental Functions',
              'KFSS1-Visual or Optic Functions',
              'KFSS1-Cerebellar Functions']

results = {}

# Loop through each category
for category in categories:
    # Filtering based on the category
    category_df = KFSS_qs_1[KFSS_qs_1['QSTEST'] == category]
    
    # Pivot table for the category
    grouped_category_df = category_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
    category_name = category.split(' ')[0]
    grouped_category_df.columns = ['USUBJID'] + [f"{category_name.replace(' ', '_')}-{period}" for period in grouped_category_df.columns[1:]]
    
    # Store the result in the dictionary
    results[category] = grouped_category_df

# Merge all results on 'USUBJID'
result_KFSS_1 = results[categories[0]]  # Start with the first category
for category in categories[1:]:
    result_KFSS_1 = result_KFSS_1.merge(results[category], on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient and period
result_KFSS_1 = result_KFSS_1.drop_duplicates(subset=['USUBJID'])

def assign_value(row):
    if row['QSTEST'] in ['KFSS1-Bowel and Bladder Functions', 'KFSS1-Visual or Optic Functions']: #'KFSS1-Other Functions', 
        return 'PHYSICAL'
    else:
        return 'MENTAL'

# Apply the function row-wise to assign values to column D
KFSS_qs_2['QSSCAT'] = KFSS_qs_2.apply(assign_value, axis=1)

def set_scoremax(row):
    if row['QSTEST'] in ['KFSS1-Cerebellar Functions', 'KFSS1-Brain Stem Functions', 'KFSS1-Cerebral or Mental Functions']:
        return 5
    #elif row['QSTEST'] in ['KFSS1-Other Functions']:
        #return 1
    else:
        return 6

# Apply the function row-wise to set the values in column B
KFSS_qs_2['SCOREMAX'] = KFSS_qs_2.apply(set_scoremax, axis=1)
grouped_sum = KFSS_qs_2.groupby(['USUBJID', 'QSDY', 'QSSCAT']).agg({'QSSTRESN': 'sum', 'SCOREMAX': 'sum'}).reset_index()
grouped_sum.rename(columns={'QSSTRESN': 'TOTALSCORE'}, inplace=True)
grouped_sum.rename(columns={'SCOREMAX': 'TOTALMAXSCORE'}, inplace=True)
grouped_sum['QSPERC'] = grouped_sum['TOTALSCORE'] / grouped_sum['TOTALMAXSCORE'] 
KFSS_qs_2 = grouped_sum 
KFSS_qs_2 = KFSS_qs_2.drop(columns=['TOTALSCORE','TOTALMAXSCORE'])

KFSS_qs_2 = KFSS_qs_2.copy()  # Create a copy to avoid the warning
conditions = [
    (KFSS_qs_2['QSDY'] <= 1),
    ((KFSS_qs_2['QSDY'] > 1) & (KFSS_qs_2['QSDY'] <= 730)),
    #((KFSS_qs['QSDY'] > 365) & (KFSS_qs['QSDY'] <= 730)),
    ((KFSS_qs_2['QSDY'] > 730)) #& (KFSS_qs['QSDY'] <= 1095)),
    #((KFSS_qs['QSDY'] > 1095) & (KFSS_qs['QSDY'] <= 1460)) 
]
# Define corresponding values for each condition
values = ['before', '2y', 'after_2y'] # , '4y' - if i use this i have 93% missing in the time

# Create the new column "FT_PERIOD"
KFSS_qs_2['QS_PERIOD'] = np.select(conditions, values, default='NaN')
KFSS_qs_2 = KFSS_qs_2.dropna(subset=['QSDY']) #Drop observations for which we don't have time of test

# Filtering based on the condition 'QSSCAT' == 'MENTAL'
brain_df = KFSS_qs_2[KFSS_qs_2['QSSCAT'] == 'MENTAL']

# Pivot table for 'BRAIN' category
grouped_brain_df = brain_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
grouped_brain_df.columns = ['USUBJID'] + [f"KFSS_M-{period}" for period in grouped_brain_df.columns[1:]]

# Filtering based on the condition 'QSSCAT' != 'BRAIN' (no need to check both cases)
non_brain_df = KFSS_qs_2[KFSS_qs_2['QSSCAT'] == 'PHYSICAL']

# Pivot table for non-'BRAIN' category
grouped_non_brain_df = non_brain_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
grouped_non_brain_df.columns = ['USUBJID'] + [f"KFSS_P-{period}" for period in grouped_non_brain_df.columns[1:]]

# Merge the new DataFrames with the original DataFrame on 'USUBJID'
result_KFSS_2 = pd.merge(KFSS_qs_2[['USUBJID']], grouped_brain_df, on='USUBJID', how='left')
result_KFSS_2 = pd.merge(result_KFSS_2, grouped_non_brain_df, on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient and period
result_KFSS_2 = result_KFSS_2.drop_duplicates(subset=['USUBJID'])

# Reorganize columns
#desired_order = ['USUBJID', 'KFSS_M-before', 'KFSS_M-2y', 'KFSS_M-after_2y', 'KFSS_P-before', 'KFSS_P-2y', 'KFSS_P-after_2y']
#result_KFSS = pivot_df[desired_order]

# Merge both
result_KFSS = pd.merge(result_KFSS_1, result_KFSS_2, on='USUBJID', how='inner')

# RAND36 #
RAND36_qs = qs[qs['QSCAT'] == 'RAND-36 V1.0']
RAND36_qs = RAND36_qs.drop(columns=['QSSEQ','QSSTRESC','VISITNUM','VISIT'])

def replace_last_two_chars(string):
    return string[-2:]

# Apply the function to the column
RAND36_qs['QSTESTCD'] = RAND36_qs['QSTESTCD'].apply(replace_last_two_chars)

def assign_value(row):
    if row['QSSCAT'] in ['PHYSICAL FUNCTIONING', 'GENERAL HEALTH', 'ROLE LIMITATIONS DUE TO PHYSICAL HEALTH', 'PAIN', 'HEALTH CHANGE']:
        return 'PHYSICAL'
    else:
        return 'MENTAL'
# Apply the function row-wise to assign values to desired column 
RAND36_qs['QSNEWCAT'] = RAND36_qs.apply(assign_value, axis=1)

def set_scoremax(row):
    if row['QSSCAT'] in ['PHYSICAL FUNCTIONING']:
        return 3
    elif row['QSSCAT'] in ['ROLE LIMITATIONS DUE TO PHYSICAL HEALTH','ROLE LIMITATIONS DUE TO EMOTIONAL PROBLEMS']:
        return 2
    elif (row['QSSCAT'] in ['EMOTIONAL WELL-BEING','ENERGY/FATIGUE']) or (row['QSTEST'] in ['R3601-How Much Bodily Pain Have You Had']):
        return 6
    else:
        return 5
# Apply the function row-wise to set the values in column B
RAND36_qs['SCOREMAX'] = RAND36_qs.apply(set_scoremax, axis=1)

# Function to reverse code the value
def reverse_code(value, max_scale):
    return max_scale - value + 1

# List of question numbers to reverse code
questions_to_reverse = ['01', '02', '20', '22', '34', '36', '21', '23', '26', '27', '30']

# Iterate through the rows and reverse code if the question number is in the list
for index, row in RAND36_qs.iterrows():
    if row['QSTESTCD'] in questions_to_reverse:
        max_scale = row['SCOREMAX']
        RAND36_qs.at[index, 'QSSTRESN'] = reverse_code(row['QSSTRESN'], max_scale)

grouped_sum = RAND36_qs.groupby(['USUBJID', 'QSDY', 'QSNEWCAT']).agg({'QSSTRESN': 'sum', 'SCOREMAX': 'sum'}).reset_index()
grouped_sum['QSPERC'] = grouped_sum['QSSTRESN'] / grouped_sum['SCOREMAX']
RAND36_qs = grouped_sum.copy()
RAND36_qs = RAND36_qs.drop(columns=['QSSTRESN','SCOREMAX'])
RAND36_qs = RAND36_qs.copy()  # Create a copy to avoid the warning

conditions = [
    (RAND36_qs['QSDY'] <= 1),
    (RAND36_qs['QSDY'] > 1) 
]
# Define corresponding values for each condition
values = ['before', 'after'] 

# Create the new column "FT_PERIOD"
RAND36_qs['QS_PERIOD'] = np.select(conditions, values, default='NaN')
RAND36_qs = RAND36_qs.dropna(subset=['QSDY']) #Drop observations for which we don't have time of test

# Filtering based on the condition 'QSNEWCAT' == 'MENTAL'
mental_df = RAND36_qs[RAND36_qs['QSNEWCAT'] == 'MENTAL']

# Pivot table for 'MENTAL' category
grouped_mental_df = mental_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
grouped_mental_df.columns = ['USUBJID'] + [f"RAND36_M-{period}" for period in grouped_mental_df.columns[1:]]

# Filtering based on the condition 'QSNEWCAT' == 'PHYSICAL'
physical_df = RAND36_qs[RAND36_qs['QSNEWCAT'] == 'PHYSICAL']

# Pivot table for 'PHYSICAL' category
grouped_physical_df = physical_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
grouped_physical_df.columns = ['USUBJID'] + [f"RAND36_P-{period}" for period in grouped_physical_df.columns[1:]]

# Merge the new DataFrames with the original DataFrame on 'USUBJID'
result_RAND36 = pd.merge(RAND36_qs[['USUBJID']], grouped_mental_df, on='USUBJID', how='left')
result_RAND36 = pd.merge(result_RAND36, grouped_physical_df, on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient and period
result_RAND36 = result_RAND36.drop_duplicates(subset=['USUBJID'])

# Reorganize the columns
desired_order = ['USUBJID', 'RAND36_M-before', 'RAND36_M-after', 'RAND36_P-before', 'RAND36_P-after']
result_RAND36 = result_RAND36[desired_order]

# SF12 #
SF_rows = qs[qs['QSCAT'] == 'SF-12 V2']

# Apply the function to the column
SF_rows['QSTESTCD'] = SF_rows['QSTESTCD'].apply(replace_last_two_chars)

columns_to_drop = ['QSCAT', 'QSORRES', 'VISITNUM', 'QSEVLINT']
SF_rows.drop(columns=columns_to_drop, inplace=True)

max_qsdy_baseline = SF_rows.loc[SF_rows['VISIT'] == 'DAY 1', 'QSDY'].max()

def assign_value(row):
    if row['QSSCAT'] in ['GENERAL HEALTH', 'PHYSICAL FUNCTIONING', 'ROLE PHYSICAL', 'BODILY PAIN']:
        return 'PHYSICAL'
    else:
        return 'MENTAL'

# Apply the function row-wise to assign values to desired column 
SF_rows['QSTEST'] = SF_rows.apply(assign_value, axis=1)

def set_scoremax(row):
    return 3 if row['QSSCAT'] == 'PHYSICAL FUNCTIONING' else 5

# Apply the function row-wise to set the values in desired column 
SF_rows['SCOREMAX'] = SF_rows.apply(set_scoremax, axis=1)

# List of question numbers to reverse code
questions_to_reverse = ['01', '05', '6A', '6B']

# Iterate through the rows and reverse code if the question number is in the list
for index, row in SF_rows.iterrows():
    if row['QSTESTCD'] in questions_to_reverse:
        max_scale = row['SCOREMAX']
        SF_rows.at[index, 'QSSTRESN'] = reverse_code(row['QSSTRESN'], max_scale)

grouped_sum = SF_rows.groupby(['USUBJID', 'QSDY', 'QSTEST']).agg({'QSSTRESN': 'sum', 'SCOREMAX': 'sum'}).reset_index()
grouped_sum['QSPERC'] = grouped_sum['QSSTRESN'] / grouped_sum['SCOREMAX']
SF_rows = grouped_sum
SF_rows = SF_rows.drop(columns=['QSSTRESN','SCOREMAX'])

SF_rows = SF_rows.copy()  # Create a copy to avoid the warning
conditions = [
    (SF_rows['QSDY'] <= max_qsdy_baseline),
    (SF_rows['QSDY'] > max_qsdy_baseline)
]

# Define corresponding values for each condition
values = ['before', 'after'] 

# Create the new column "FT_PERIOD"
SF_rows['QS_PERIOD'] = np.select(conditions, values, default='NaN')
SF_rows = SF_rows.dropna(subset=['QSDY']) #Drop observations for which we don't have time of test

# Filtering based on the condition 'QSTEST' == 'MENTAL'
mental_df = SF_rows[SF_rows['QSTEST'] == 'MENTAL']

# Pivot table for 'MENTAL' category
grouped_mental_df = mental_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
grouped_mental_df.columns = ['USUBJID'] + [f"SF12_M-{period}" for period in grouped_mental_df.columns[1:]]

# Filtering based on the condition 'QSTEST' == 'PHYSICAL'
physical_df = SF_rows[SF_rows['QSTEST'] == 'PHYSICAL']

# Pivot table for 'PHYSICAL' category
grouped_physical_df = physical_df.pivot_table(values='QSPERC', index='USUBJID', columns='QS_PERIOD', aggfunc='median', fill_value=None).reset_index()
grouped_physical_df.columns = ['USUBJID'] + [f"SF12_P-{period}" for period in grouped_physical_df.columns[1:]]

# Merge the new DataFrames with the original DataFrame on 'USUBJID'
result_SF = pd.merge(SF_rows[['USUBJID']], grouped_mental_df, on='USUBJID', how='left')
result_SF = pd.merge(result_SF, grouped_physical_df, on='USUBJID', how='left')

# Drop duplicate rows to keep only unique rows per patient and period
result_SF = result_SF.drop_duplicates(subset=['USUBJID'])

# Reorganize the columns
desired_order = ['USUBJID', 'SF12_M-before', 'SF12_M-after', 'SF12_P-before', 'SF12_P-after']
result_SF = result_SF[desired_order]

# Merge all questionnaires
questionnaires_aggregated = pd.merge(result_BDI, result_EDSS, on='USUBJID', how='outer')
questionnaires_aggregated = pd.merge(questionnaires_aggregated, result_KFSS, on='USUBJID', how='outer')
questionnaires_aggregated = pd.merge(questionnaires_aggregated, result_RAND36, on='USUBJID', how='outer')
questionnaires_aggregated = pd.merge(questionnaires_aggregated, result_SF, on='USUBJID', how='outer')

##################################################################################################################
### DELETE THIS PART IN THE CODE IF YOU WANT THE RAND36 AND SF12 SEPARATELY
    ### BECAUSE THEY ARE DISJOINT, WE MERGE THEM INTO ONE COLUMN (since both are percentages)
    ### NOTE: for the binary indicator we can only keep one (before and after - because this will be the same)
# Create a new column 'RAND36_M-before' and fill it with values from 'SF12_M-before'
questionnaires_aggregated['M_R36-SF12-before'] = questionnaires_aggregated['RAND36_M-before'].fillna(questionnaires_aggregated['SF12_M-before'])
# Create a new column 'RAND36_P-before' and fill it with values from 'SF12_P-before'
questionnaires_aggregated['P_R36-SF12-before'] = questionnaires_aggregated['RAND36_P-before'].fillna(questionnaires_aggregated['SF12_P-before'])
# Create a new column 'R36-SF12-before' with a binary indicator
questionnaires_aggregated['R36-SF12-before_Ind'] = questionnaires_aggregated.apply(lambda row: 1 if pd.notna(row['RAND36_M-before']) else (0 if pd.notna(row['SF12_M-before']) else np.nan), axis=1)
# Drop the original columns if needed
questionnaires_aggregated = questionnaires_aggregated.drop(['SF12_P-before','SF12_M-before','RAND36_P-before','RAND36_M-before'], axis=1)

# Create a new column 'RAND36_M-after' and fill it with values from 'SF12_M-after'
questionnaires_aggregated['M_R36-SF12-after'] = questionnaires_aggregated['RAND36_M-after'].fillna(questionnaires_aggregated['SF12_M-after'])
# Create a new column 'RAND36_P-after' and fill it with values from 'SF12_P-after'
questionnaires_aggregated['P_R36-SF12-after'] = questionnaires_aggregated['RAND36_P-after'].fillna(questionnaires_aggregated['SF12_P-after'])
# Create a new column 'R36-SF12' with a binary indicator
questionnaires_aggregated['R36-SF12-after_Ind'] = questionnaires_aggregated.apply(lambda row: 1 if pd.notna(row['RAND36_M-after']) else (0 if pd.notna(row['SF12_M-after']) else np.nan), axis=1)
# Drop the original columns if needed
questionnaires_aggregated = questionnaires_aggregated.drop(['SF12_P-after','SF12_M-after','RAND36_P-after','RAND36_M-after'], axis=1)
##################################################################################################################

folder_name = 'new_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Specify the path for the CSV file
csv_file_path = os.path.join(folder_name, 'QS_agg.csv')

# Save the DataFrame to CSV
questionnaires_aggregated.to_csv(csv_file_path, index=False)

# Print message after CSV file creation
print("QS_agg.csv has been created in the folder new_data")



###############################################
################# Merge Data ##################
###############################################

# Define possible paths where CSV files might be located
possible_paths = [
    'C:/Users/lenne/OneDrive/Documenten/Master of Statistics and Data Science/2023-2024/Master thesis/Thesis_Sofia_Lennert/new_data',
    'C:/Users/anaso/Desktop/SOFIA MENDES/KU Leuven/Master Thesis/Thesis_Sofia_Lennert/new_data'
]

# Define file names
file1 = 'DM_agg.csv'
file2 = 'CE_agg.csv'
file3 = 'MH_agg.csv'
file4 = 'SM_agg.csv'
file5 = 'FT_agg.csv'
file6 = 'OE_agg.csv'
file7 = 'QS_agg.csv'

# Find full paths to the CSV files
path1 = next((f'{path}/{file1}' for path in possible_paths if os.path.exists(f'{path}/{file1}')), None)
path2 = next((f'{path}/{file2}' for path in possible_paths if os.path.exists(f'{path}/{file2}')), None)
path3 = next((f'{path}/{file3}' for path in possible_paths if os.path.exists(f'{path}/{file3}')), None)
path4 = next((f'{path}/{file4}' for path in possible_paths if os.path.exists(f'{path}/{file4}')), None)
path5 = next((f'{path}/{file5}' for path in possible_paths if os.path.exists(f'{path}/{file5}')), None)
path6 = next((f'{path}/{file6}' for path in possible_paths if os.path.exists(f'{path}/{file6}')), None)
path7 = next((f'{path}/{file7}' for path in possible_paths if os.path.exists(f'{path}/{file7}')), None)

# Check if paths are found
if None in [path1, path2, path3, path4, path5, path6, path7]: 
    print("Some files were not found. Please check the paths and filenames.")
else:
    # Read CSV files into DataFrames
    dataset1 = pd.read_csv(path1)
    dataset2 = pd.read_csv(path2)
    dataset3 = pd.read_csv(path3)
    dataset4 = pd.read_csv(path4)
    dataset5 = pd.read_csv(path5)
    dataset6 = pd.read_csv(path6)
    dataset7 = pd.read_csv(path7)

    # Merge the datasets
    merged_df = pd.merge(dataset1, dataset2, on='USUBJID', how='outer')
    merged_df = pd.merge(merged_df, dataset3, on='USUBJID', how='outer')
    merged_df = pd.merge(merged_df, dataset4, on='USUBJID', how='outer')
    merged_df = pd.merge(merged_df, dataset5, on='USUBJID', how='outer')
    merged_df = pd.merge(merged_df, dataset6, on='USUBJID', how='outer')
    merged_df = pd.merge(merged_df, dataset7, on='USUBJID', how='outer')


# If someone has NA in the number of confirmed relapses, then set to 0
merged_df['NRELAP'] = merged_df['NRELAP'].fillna(0)

# Categorize the relapses into bins
# We comment this code because we need the original variable to calculate correlations
'''
def bin_column(value):
    if value in [0, 1, 2, 3]:
        return str(value)
    else:
        return '4+'
merged_df['NRELAP'] = merged_df['NRELAP'].apply(bin_column)
'''

# Sort by USUBJID 
merged_df = merged_df.sort_values(by='USUBJID')

### EXPORT 
folder_name = 'new_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Specify the path for the CSV file
csv_file_path = os.path.join(folder_name, 'merged_data.csv')

# Save the DataFrame to CSV
merged_df.to_csv(csv_file_path, index=False)

print("Complete: Unified static dataframe has been obtained")