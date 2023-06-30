# imports
import pandas as pd

# read csv file
df = pd.read_csv("data.csv", header=0, decimal=",")

# process data
## avg-nr-inter-village-friends

## avg-intra-village-interaction-frequency

## avg-inter-village-interaction-frequency

## avg-farmgroup-meeting-frequency

## farmgroup-meeting-attendance-percentage

## percentage-negative-wom

## base-adoption-probability

# export data as csv file
df.to_csv("data-processed.csv")