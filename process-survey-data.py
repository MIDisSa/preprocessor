# imports
import pandas as pd

# read csv file
df = pd.read_csv("data.csv", header=0, decimal=",")

# create new dataframe 
## add derived data into this dataframe
## this dataframe will be exported as csv file to be passed to the automator
## columns are named after parameter names in netlogo model
df_results = pd.DataFrame(columns=['nr_default_friends_inter_village', 'avg_intra_village_interaction_frequency', 'avg_inter_village_interaction_frequency', 'avg_chief_farmer_meeting_frequency', 'farmgroup_meeting_attendance_percentage', 'avg_intra_mention_percentage', 'avg_inter_mention_percentage', 'avg_meeting_mention_percentage', 'percentage_negative_WoM', 'base_adoption_probability'])

# process data
## avg-nr-inter-village-friends

## avg-intra-village-interaction-frequency

## avg-inter-village-interaction-frequency

## avg-farmgroup-meeting-frequency

## farmgroup-meeting-attendance-percentage

## percentage-negative-wom

## base-adoption-probability

# export data as csv file
df_results.to_csv("data-processed.csv")