"""
====================
 imports and read data
====================
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sb
import math

# read csv file
df = pd.read_csv("data.csv", header=0, decimal=",")

"""
==============================================
 Transform Likert Scale into numerical values
==============================================
"""


def rename_likert_cols(data):
    # Find columns that contain Likert scale values in the first row
    likert_cols = []
    for col in data.columns:
        first_val = data.loc[0, col]
        if isinstance(first_val, str) and first_val.lower() in ['strongly disagree', 'disagree', "don't know", 'agree',
                                                                'strongly agree', "strong agree"]:
            likert_cols.append(col)

    # Rename columns by adding "L_" prefix
    new_cols = {col: 'L_' + col for col in likert_cols}
    data.rename(columns=new_cols, inplace=True)

    return data


def map_likert_scale(val):
    if val == 'Strongly agree':
        return 1
    elif val == 'Strong agree':
        return 1
    elif val == 'Agree':
        return 2
    elif val == "Don't know":
        return 3
    elif val == 'Disagree':
        return 4
    elif val == 'Strongly disagree':
        return 5
    else:
        return "wrong mapping"


# Rename and convert Likert scale columns
df_renamed = rename_likert_cols(df)

# Map the Likert scale values to numerical values for the columns with prefix "L_"
for col in df_renamed.columns:
    if col.startswith('L_'):
        df_renamed[col] = df_renamed[col].apply(map_likert_scale)

"""
====================
 Train-chief-influence
====================
"""

df_TCI = df_renamed
# impute data if NaN
df_TCI['PART 2: STORAGE AND POST-HARVEST/Have you received training on hermetic bag?'].fillna('No', inplace=True)
df_TCI["PART 7: SOCIAL CAPITAL AND NETWORKING/What is your/partner's role in the group/institution?"].fillna(
    'undefined', inplace=True)

# Transform them into numeric variables
# Convert binary columns
df_TCI["PART 2: STORAGE AND POST-HARVEST/Have you received training on hermetic bag?"] = df_TCI[
    "PART 2: STORAGE AND POST-HARVEST/Have you received training on hermetic bag?"].replace({'Yes': 1, 'No': 0})
df_TCI['Enumerator: Select Farmer Group type'] = df_TCI['Enumerator: Select Farmer Group type'].map(
    {'Treatment group': 1, 'Control group': 0})

# To check
np.asarray(df_TCI["PART 2: STORAGE AND POST-HARVEST/Have you received training on hermetic bag?"])
np.asarray(df_TCI["PART 7: SOCIAL CAPITAL AND NETWORKING/What is your/partner's role in the group/institution?"])
np.asarray(df_TCI['Enumerator: Select Farmer Group type'])


# Turn categorical data into dummy variables and merge
def create_dummy_variables(df, column):
    dummy_df = pd.get_dummies(df[column], prefix='role', drop_first=True)
    return df.join(dummy_df)


df_TCI = create_dummy_variables(df_TCI,
                                "PART 7: SOCIAL CAPITAL AND NETWORKING/What is your/partner's role in the group/institution?")

# Logistic Regression
x = df_TCI[['Enumerator: Select Farmer Group type',
            "PART 2: STORAGE AND POST-HARVEST/Have you received training on hermetic bag?", "role_Member", "role_other",
            "role_undefined"]]

# solve shape issues
y = df_TCI['PART 2: STORAGE AND POST-HARVEST/What kind of storage methods do you use?/Hermetic bag'].values

x = sm.add_constant(x)
model = sm.Logit(y, x).fit()
predictions = model.predict(x)
coefficients = model.params

# Calculate the linear combination of coefficients and input features:
linear_combination = (coefficients[2]) + coefficients[0]

# Calculate the sigmoid value:
sigmoid_value_TCI = 1 / (1 + np.exp(-linear_combination))

# Create a list with the single value
sigmoid_value_TCI = [sigmoid_value_TCI]


"""
=============================
 AVG-mention-percentage
=============================
"""
df_mentionPerc = df_renamed
df_mentionPerc = df_mentionPerc[df_mentionPerc['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the last 10 discussions you had with contacts in your village. In how many discussions were hermetic storage bags mentioned?'] <= 10] #intra
df_mentionPerc = df_mentionPerc[df_mentionPerc['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the last 10 discussions you had with contacts outside of your village. In how many discussions were hermetic storage bags mentioned?'] <= 10] #inter

mean_intra_mention_percentage = df_mentionPerc['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the last 10 discussions you had with contacts in your village. In how many discussions were hermetic storage bags mentioned?'].mean()
mean_inter_mention_percentage = df_mentionPerc['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the last 10 discussions you had with contacts outside of your village. In how many discussions were hermetic storage bags mentioned?'].mean()
# TODO: skewness


"""
=============================
 AVG_village_interaction_frequency
=============================
"""
df_interactionFreq = df_renamed
df_interactionFreq = df.dropna(subset=['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the people you know in your village. In a typical week, how many times do you communicate with each person you know? The way of communication (in-person or phone) does not matter.', 'PART 7: SOCIAL CAPITAL AND NETWORKING/Now think about people you know that don’t live in your village. In a typical week, how many people outside your village do you communicate with? Include all relatives, friends, traders, extension officers and other people. The way of communication (in-person or phone) does not matter.'])
mean_intra_interaction_frequency = df_interactionFreq['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the people you know in your village. In a typical week, how many times do you communicate with each person you know? The way of communication (in-person or phone) does not matter.'].mean()
mean_inter_interaction_frequency = df_interactionFreq['PART 7: SOCIAL CAPITAL AND NETWORKING/Now think about people you know that don’t live in your village. In a typical week, how many people outside your village do you communicate with? Include all relatives, friends, traders, extension officers and other people. The way of communication (in-person or phone) does not matter.'].mean()
# TODO: skewness



"""
=================================
 nr_default_friends_inter_village
=================================
"""
df_friends = df_renamed
nr_default_friends_inter_village = df_friends['PART 7: SOCIAL CAPITAL AND NETWORKING/Think about the people you know in your village that do not live in your household. In a typical week, how many different people from your village do you communicate with? Include all relatives, friends, neighbours and other people who live in your village. The way of communication (in-person or phone) does not matter.'].mean()
# TODO: skewness



"""
=============================
 create new dataframe to save
=============================
"""

# add derived data into this dataframe
# this dataframe will be exported as csv file to be passed to the automator
# columns are named after parameter names in netlogo model
df_results = pd.DataFrame(columns=['train_chief_influence', 'nr_default_friends_inter_village', 'avg_intra_village_interaction_frequency',
                                   'avg_inter_village_interaction_frequency', 'avg_chief_farmer_meeting_frequency',
                                   'farmgroup_meeting_attendance_percentage', 'avg_intra_mention_percentage',
                                   'avg_inter_mention_percentage', 'avg_meeting_mention_percentage',
                                   'percentage_negative_WoM', 'base_adoption_probability'])

# Assign the values
df_results['train_chief_influence'] = sigmoid_value_TCI

# nr_default_friends_inter_village
df_results['nr_default_friends_inter_village'] = [nr_default_friends_inter_village]

# avg_intra_mention_percentage
df_results['avg_intra_mention_percentage'] = [mean_intra_mention_percentage]

# avg_inter_mention_percentage
df_results['avg_inter_mention_percentage'] = [mean_inter_mention_percentage]

# avg_intra_village_interaction_frequency
df_results['avg_intra_village_interaction_frequency'] = [mean_intra_interaction_frequency]

# avg_inter_village_interaction_frequency
df_results['avg_inter_village_interaction_frequency'] = [mean_inter_interaction_frequency]

# avg_chief_farmer_meeting_frequency
df_results['avg_chief_farmer_meeting_frequency'] = [30]

# farmgroup-meeting-attendance-percentage
# eigentlich Slider rein?

# avg_meeting_mention_percentage

# percentage-negative-wom
df_results['percentage_negative_WoM'] = [0.19]

# base-adoption-probability
df_results['base_adoption_probability'] = [0.05]



"""
=========
 export
=========
"""

# export data as csv file
df_results.to_csv("data-processed.csv")
