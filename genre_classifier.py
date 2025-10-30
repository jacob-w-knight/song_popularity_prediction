import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Download data from HuggingFace
def download_data():
    df = pd.read_csv(
        "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
    ).iloc[:, 1:]
    df.to_csv("spotify_data.csv", index=False)
    return


# Call function to download data
# download_data()

# Read file
df = pd.read_csv("spotify_data.csv")

# Extract numeric data
numeric_columns = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Build the numeric dataframe
numeric_df = df[numeric_columns]

# Create a larger figure
plt.figure(figsize=(12, 10))

# Plot title
plt.title("Correlation Matrix")

# Compute the correlation matrix
corr = numeric_df.corr()

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap=cmap, square=True)

# Display the plot
plt.show()
