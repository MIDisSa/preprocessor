import numpy as np
import matplotlib.pyplot as plt
import re


# idea: Netlogo -> create population of turtles, assign them size, generate skewed random number
# Then we write data into txt file --> plot it here

def plot_histogram(data_source, skewness_type):
    data = []
    with open(data_source, 'r') as file:
        for line in file:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)  # regex matches floating numbers and int
            for match in matches:
                data.append(float(match))

    # Plot
    plt.hist(data, bins=20)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {skewness_type} Output")
    plt.show()


data_source_right_skew = "/Users/anki/Documents/Uni_Zurich/Master_project/data_right_skew.txt"
data_source_left_skew = "/Users/anki/Documents/Uni_Zurich/Master_project/data_left_skew.txt"
data_source_left_skew_steeper = "/Users/anki/Documents/Uni_Zurich/Master_project/data_right_skew_steeper.txt"
data_source_slider_skew = "/Users/anki/Documents/Uni_Zurich/Master_project/data_slider_skew.txt"

plot_histogram(data_source_right_skew, "Right Skew")
plot_histogram(data_source_left_skew, "Left Skew")
plot_histogram(data_source_left_skew_steeper, "Left Skew Steeper")
plot_histogram(data_source_slider_skew, "With Slider")