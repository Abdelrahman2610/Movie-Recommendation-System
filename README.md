# Movie Recommendation System

## Overview

This project implements a comprehensive movie recommendation system using collaborative filtering techniques on the MovieLens 100K dataset. The system leverages user ratings to generate personalized movie suggestions through various methods, including user-based and item-based collaborative filtering, matrix factorization via Singular Value Decomposition (SVD), and a hybrid approach.

Key objectives:
- Build and evaluate recommendation models for accuracy and relevance.
- Visualize data distributions, similarities, and performance metrics.
- Provide a scalable foundation for recommendation systems in applications like streaming services.

The project is implemented in Python using a Jupyter notebook, with models saved as pickled files and visualizations exported as PNG images.

## Features

- **Collaborative Filtering**:
  - User-based: Recommends movies based on similar users' preferences.
  - Item-based: Recommends movies similar to those the user has liked.
- **Matrix Factorization**: Uses SVD (via Scikit-learn and Surprise library) for latent factor modeling.
- **Hybrid Model**: Combines multiple techniques for improved recommendations.
- **Evaluation Metrics**: Precision@5 and Recall@5 to assess recommendation quality.
- **Data Exploration**: Includes histograms for rating distributions and heatmaps for similarity matrices.
- **Model Persistence**: Saved models for reuse (e.g., similarity matrices and SVD models).
- **Visualizations**: Exported plots for insights into data and model performance.

## Dataset

The system uses the [MovieLens 100K dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset) from Kaggle, which includes:
- 100,000 ratings from 943 users on 1,682 movies.
- User demographics, movie details (titles, genres, release dates), and pre-split training/test sets.

Data files are not included in the repository due to size constraints. Download them from Kaggle.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/Abdelrahman2610/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages using:
   ```
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - pandas==2.2.2
   - numpy==1.26.4
   - scikit-learn==1.5.1
   - surprise==1.1.4
   - matplotlib==3.9.2
   - seaborn==0.13.2
   - joblib==1.4.2

4. **Download the Dataset**:
   - Manually: Download from [Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset).

     
## Usage

1. **Run the Jupyter Notebook**:
   ```
   jupyter notebook movie-recommendation-system.ipynb
   ```
   - Execute cells sequentially to load data, train models, generate recommendations, and export visualizations/models.
   - Outputs are saved to `/kaggle/working/` (adapt paths for local runs).

2. **Generate Recommendations**:
   - In the notebook, use functions like `recommend_movies_user_based(user_id, n=5)` to get top-N recommendations for a user.
   - Example: For user ID 1, retrieve unseen movies predicted to have high ratings.

3. **Evaluate Models**:
   - The notebook computes Precision@5 and Recall@5 on the test set.
   - Visualize results using the generated plots.

4. **Load Saved Models**:
   - Use `joblib.load('models/user_similarity.pkl')` to reload similarity matrices or SVD models for inference.

## Results

The models were evaluated using Precision@5 (the proportion of recommended items that are relevant) and Recall@5 (the proportion of relevant items that are recommended).

Results are summarized below:

| Method              | Precision@5 | Recall@5 |
|---------------------|-------------|----------|
| User-Based CF       | 0.1096     | 0.1036   |
| Item-Based CF       | 0.0796     | 0.0694   |
| Surprise SVD        | 0.0408     | 0.0322   |
| Hybrid              | 0.0208     | 0.0122   |

User-based collaborative filtering outperformed others in this setup. The hybrid model shows potential but may require further tuning for better integration.

## Visualizations

The notebook generates and saves the following visualizations in the `plots/` directory for data exploration and model analysis:

- **metrics_comparison.png**: Bar charts comparing Precision@5 and Recall@5 across models.
- **item_similarity_heatmap.png**: Heatmap showing cosine similarity between the first 50 movies (item-based).
- **user_similarity_heatmap.png**: Heatmap showing cosine similarity between the first 50 users (user-based).
- **ratings_per_user.png**: Histogram of the distribution of ratings per user.
- **ratings_per_movie.png**: Histogram of the distribution of ratings per movie.
- **ratings_distribution.png**: Histogram of overall rating values (1-5 scale).

These plots highlight data sparsity, similarity patterns, and performance insights.

## Saved Models

Trained models are saved in the root directory (or `/kaggle/working/` in the notebook) for persistence and reuse:

- **item_similarity.pkl**: Pickled item-item similarity matrix (cosine similarity).
- **user_similarity.pkl**: Pickled user-user similarity matrix (cosine similarity).
- **surprise_svd_model.pkl**: Pickled SVD model from the Surprise library.

Load these with `joblib` for quick predictions without retraining.

## Project Structure

```
movie-recommendation-system/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files/folders to ignore in Git
├── notebooks/
    └── movie-recommendation-system.ipynb  # Main Jupyter notebook
├── plots/                      # Generated visualizations
│   ├── item_similarity_heatmap.png
│   ├── metrics_comparison.png
│   ├── ratings_distribution.png
│   ├── ratings_per_movie.png
│   ├── ratings_per_user.png
│   └── user_similarity_heatmap.png
└── models/ 
     ├── item_similarity.pkl         # Item similarity matrix
     ├── user_similarity.pkl         # User similarity matrix
     └── surprise_svd_model.pkl      # Surprise SVD model

```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## Acknowledgments

- **Dataset**: Provided by GroupLens Research via MovieLens.
- **Libraries**: Thanks to the maintainers of Pandas, Scikit-learn, Surprise, Matplotlib, and Seaborn.
- **Inspiration**: Based on standard collaborative filtering techniques in recommendation systems literature.

