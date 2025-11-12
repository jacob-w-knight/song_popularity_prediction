"""
Visualization utilities for genre analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from adjustText import adjust_text


def plot_dendrogram(linkage_matrix, labels, figsize=(18, 6)):
    """
    Plot hierarchical clustering dendrogram

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from scipy
    labels : list
        Genre labels
    figsize : tuple, optional
        Figure size
    """
    plt.figure(figsize=figsize)
    dendrogram(linkage_matrix, labels=labels, orientation="top")
    plt.title("Dendrogram")
    plt.xlabel("Track Genres")
    plt.ylabel("Distances")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_mds_clusters(
    embedding, cluster_labels, genre_names, cluster_name_map=None, figsize=(14, 10)
):
    """
    Plot MDS visualization with cluster coloring

    Parameters
    ----------
    embedding : np.ndarray
        2D MDS coordinates
    cluster_labels : array-like
        Cluster assignments
    genre_names : list
        Genre names
    cluster_name_map : dict, optional
        Mapping of cluster IDs to names
    figsize : tuple, optional
        Figure size
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="Set1")

    # Create DataFrame
    import pandas as pd

    plot_df = pd.DataFrame(
        {
            "MDS_1": embedding[:, 0],
            "MDS_2": embedding[:, 1],
            "Genre": [g.title() for g in genre_names],
            "Cluster_Name": (
                [
                    cluster_name_map.get(label, f"Cluster {label}")
                    for label in cluster_labels
                ]
                if cluster_name_map
                else [f"Cluster {label}" for label in cluster_labels]
            ),
        }
    )

    # Create plot
    plt.figure(figsize=figsize)

    ax = sns.scatterplot(
        data=plot_df,
        x="MDS_1",
        y="MDS_2",
        hue="Cluster_Name",
        s=200,
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
    )

    # Add non-overlapping labels using adjustText
    texts = []
    for i, row in plot_df.iterrows():
        text = ax.text(
            row["MDS_1"],
            row["MDS_2"],
            row["Genre"],
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="gray",
                linewidth=0.5,
            ),
        )
        texts.append(text)

    # Adjust text positions to avoid overlap
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6, lw=1),
        expand_points=(1.2, 1.2),
        expand_text=(1.1, 1.1),
        force_points=0.3,
        force_text=0.8,
        ax=ax,
    )

    plt.title("Music Genre Similarity Landscape", fontsize=16, fontweight="bold")
    plt.xlabel("MDS Component 1", fontsize=12)
    plt.ylabel("MDS Component 2", fontsize=12)

    legend = ax.legend(
        title="Genre Clusters",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.show()


def create_radar_subplots(genre_data, figsize=(20, 14)):
    """
    Create radar chart subplots for each genre

    Parameters
    ----------
    genre_profiles : pd.DataFrame
        Average features per genre (scaled 0-100)
    figsize : tuple, optional
        Figure size
    """
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    genres = list(genre_data.index)
    features = list(genre_data.columns)

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    # Create figure with subplots
    fig, axes = plt.subplots(
        2, 3, figsize=(20, 14), subplot_kw=dict(projection="polar")
    )
    axes = axes.flatten()

    # Specific color palette
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    for i, genre in enumerate(genres):
        ax = axes[i]

        # Get values and close the shape
        values = list(genre_data.loc[genre])
        values += values[:1]

        # Create the radar plot
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=3,
            color=colors[i % len(colors)],
            markersize=8,
        )
        ax.fill(angles, values, alpha=0.3, color=colors[i % len(colors)])

        # Customize appearance
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=10, fontweight="semibold")
        ax.set_ylim(0, 1)

        # Enhanced title with emoji
        ax.set_title(f"{genre.title()}", fontsize=16, fontweight="bold", pad=30)

        # Custom grid
        ax.set_rticks([0.2, 0.4, 0.6, 0.8])
        ax.set_rgrids(
            [0.2, 0.4, 0.6, 0.8],
            labels=["0.2", "0.4", "0.6", "0.8"],
            fontsize=8,
            alpha=0.7,
        )
        ax.grid(True, alpha=1)

    # Hide unused subplots
    for j in range(len(genres), 6):
        axes[j].set_visible(False)

    # Main title
    fig.suptitle(
        "Music Genre Audio Feature Profiles", fontsize=20, fontweight="bold", y=0.95
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
