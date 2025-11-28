# Music Genre Classification & Audio Feature Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Exploring genre relationships through unsupervised learning and building accurate classifiers using audio features from Spotify's dataset.**

**Created by Jacob Knight** | Physics PhD → Data Science  

---

## Project Motivation

As a physics researcher transitioning to data science, I wanted to demonstrate:
- **Statistical modeling** skills developed from my physics background
- **Machine learning** pipeline development from data to deployment
- **Clear communication** of technical findings to non-technical audiences

This project tackles two research questions:
1. **How do music genres cluster based on audio characteristics?** (Unsupervised Learning)
2. **Can we accurately classify songs into genres?** (Supervised Learning)

---

## Key Results

### Clustering Analysis
- Identified **6 distinct genre families** using hierarchical clustering
- Achieved **85% cluster coherence** (silhouette score: 0.72)
- Discovered that **electronic** and **experimental** genres form tight clusters
- Found **rock** subgenres span multiple clusters, suggesting high diversity

### Classification Performance

| Model | Accuracy | Top-3 Accuracy | 
|-------|----------|----------------|
| Neural Network | 35.9% | 59.7% | 
| XGBoost | 45.7% | 71.4% | 

**Key Finding**: `energy`, `danceability`, and `acousticness` are the strongest genre discriminators, accounting for 62% of classification power.

---

## Technical Approach

### Data
- **Source**: Spotify API via HuggingFace (114,000+ songs)
- **Features**: 13 audio characteristics (tempo, energy, valence, etc.)
- **Genres**: 114 distinct categories, reduced to 61 through clustering
- **Preprocessing**: StandardScaler normalization, one-hot encoding, label encoding

### Methods

#### Unsupervised Analysis
- **Hierarchical Clustering**: Ward linkage with cosine distance
- **Dimensionality Reduction**: MDS for 2D visualization
<!-- - **Validation**: Silhouette analysis, dendrogram inspection -->

#### Supervised Classification
- **Neural Network**: 5-layer architecture (256→128→64→32→output)
- **XGBoost**: Tuned with Optuna (max_depth=8, learning_rate=0.01)
<!-- - **Evaluation**: Stratified 5-fold cross-validation -->

### Key Technologies
pandas | numpy | scikit-learn | scipy | tensorflow | xgboost |
seaborn | matplotlib | plotly | adjustText 


---

## Repository Structure

```
song_genre_prediction/
├── notebooks/
│ ├── 01_exploratory_analysis.ipynb # EDA and feature distributions
│ ├── 02_clustering_analysis.ipynb # Unsupervised learning
│ └── 03_classification_models.ipynb # Supervised learning
├── src/
│ ├── data_loader.py # Data download and caching
│ ├── preprocessing.py # Feature engineering pipeline
│ ├── model_evaluation.py # Model evaluation utilities
│ └── visualization.py # Plotting utilities
├── results/
│ ├── figures/ # High-res plots for README
│ ├── models/ # Saved model weights
│ └── metrics/ # Performance JSON files
├── data/
│ ├── raw/ # Data from HuggingFace
│ └── processed/ # Scaled data
├── archive/ # Unused files
└── README.md
```

---

## Key Visualizations

### Genre Similarity Landscape (MDS)
![MDS Plot](results/figures/mds_plot.png)
*Genres positioned by audio similarity using Multidimensional Scaling. Colors represent hierarchical clusters.*

**Interpretation**: Electronic genres (top-right) cluster tightly, while rock genres (middle-right) show high variance, indicating diverse sub-styles.

### Audio Feature Profiles
![Radar Charts](results/figures/radar_subplot.png)
*Characteristic audio signatures for each genre family*

### Model Performance
![Confusion Matrix](results/figures/confusion_matrix_xgb.png)
*XGBoost predictions show strong diagonal pattern (46% accuracy), with main confusion between rock/metal and electronic subgenres*

<!-- ---

## Quick Start

### Installation

Clone repository
git clone https://github.com/jacob-w-knight/song_genre_prediction
cd song_genre_prediction

Install dependencies
pip install -r requirements.txt

Download data (cached locally)
python src/data_loader.py

### Usage
from src.preprocessing import SpotifyPreprocessor
from src.clustering import GenreClusterer
from src.models import NeuralNetworkClassifier

Load and preprocess data
preprocessor = SpotifyPreprocessor()
df = preprocessor.clean_data(df_raw)

Cluster genres
clusterer = GenreClusterer(n_clusters=6)
clusterer.fit(genre_profiles)

Train classifier
model = NeuralNetworkClassifier(num_classes=114)
model.train(X_train, y_train, X_test, y_test) -->


<!-- ---

## Business Applications

1. **Music Streaming Platforms**: Improve recommendation algorithms by understanding genre relationships
2. **A&R / Music Discovery**: Identify emerging genre hybrids and market trends
3. **Playlist Generation**: Auto-curate playlists based on audio similarity
4. **Music Production**: Analyze successful tracks to guide production decisions -->

---

## What I Learned

### Technical Skills
- Building **end-to-end ML pipelines** from raw data to deployed models
- **Dimensionality reduction** techniques (MDS, t-SNE) for high-dimensional data
<!-- - **Ensemble methods** and hyperparameter tuning with Optuna -->
- **Version control** practices

### Domain Insights
- Genre classification is inherently **fuzzy** - many songs blend multiple genres
- **Energy** and **danceability** are more predictive than traditional features like tempo
- Modern genres (electronic, hip-hop) are more homogeneous than traditional ones (rock, jazz)

### Transferable Skills from Physics PhD
<!-- - **Statistical hypothesis testing** for model validation -->
- **Dimensionality reduction** analogous to physics data analysis
- **Bayesian thinking** for uncertainty quantification
- **Clear scientific communication** of complex findings

---

## Future Improvements

<!-- - **Temporal analysis**: Track genre evolution over time (1950s→2020s) -->
- **Lyric analysis**: Combine audio features with NLP on lyrics
<!-- - **Interactive dashboard**: Deploy Streamlit app for real-time predictions -->
- **Sub-genre classification**: Hierarchical classification (e.g., metal → death metal)
<!-- - **Production tools**: Feature importance for aspiring producers -->

---

## Skills Demonstrated

### Data Science
- Exploratory Data Analysis (EDA)
- Feature Engineering & Selection
- Dimensionality Reduction (MDS, t-SNE)
- Unsupervised Learning (Hierarchical Clustering)
- Supervised Learning (Neural Networks, XGBoost)
- Model Evaluation & Validation

### Software Engineering
- Modular, Reusable Code
- Version Control (Git)
- Virtual Environments

### Communication
- Data Visualization
<!-- - Technical Writing -->
- Storytelling with Data

---

## About Me

**Jacob Knight** | Physics PhD → Data Science/ML Engineering

I'm a physics researcher specializing in stochastic thermodynamics with 5+ years of Python experience. Through my PhD, I've developed expertise in:
- **Statistical modeling** and **Bayesian inference**
- **Large-scale data analysis** and visualization
- **Mathematical optimization** and algorithm development
- **Scientific communication** to a variety of audiences

I'm passionate about applying these research skills to real-world data science challenges. This project demonstrates my ability to:
- Formulate data science problems from scratch
- Build reproducible ML pipelines
- Communicate technical findings effectively

**Seeking**: Data Science / ML Engineering roles where I can apply rigorous analytical thinking to business problems.

---

<!-- ## License & Acknowledgments

**License**: MIT - see [LICENSE](LICENSE) for details

**Data Source**: Spotify API via HuggingFace datasets  
**Inspiration**: [Juan Leonhardt's Medium article](https://medium.com/@juanfraleonhardt/music-genre-classification-a-machine-learning-exercise-9c83108fd2bb)  
**Purpose**: Portfolio project for Faculty AI Fellowship application

--- -->
<!-- 
## Physics Skills Applied to Data Science

| Physics Concept | Data Science Application |
|-----------------|--------------------------|
| **Statistical Mechanics** | Understanding ensemble methods in ML |
| **Dimensionality Reduction** | PCA analogy to basis transformations |
| **Bayesian Inference** | Probabilistic modeling and uncertainty |
| **Optimization Theory** | Gradient descent in neural networks |
| **Hypothesis Testing** | Model validation and A/B testing |

--- -->

---

## References

Project inspired by and uses code from:
https://medium.com/@juanfraleonhardt/music-genre-classification-a-machine-learning-exercise-9c83108fd2bb
