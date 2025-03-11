# MLB Game Outcome Prediction using Elo Ratings and Feature Engineering
This project aims to predict the outcome of Major League Baseball (MLB) games using a combination of Elo ratings,  feature engineering, and machine learning.  It leverages historical game data to create a predictive model that incorporates various factors influencing game results, such as team Elo ratings, travel distance, time zone differences, and team performance metrics.

## Features
* **Elo Rating System:** Implements an Elo rating system to quantify team strength and predict game outcomes based on relative team abilities.
* **Feature Engineering:** Extracts and combines various features from raw game data, such as travel distance, time zone differences, and game statistics, to enhance the model's predictive power.
* **Machine Learning Model:** Trains a Gradient Boosting Classifier to predict the probability of a home team victory.
* **Statistical Analysis:** Performs various statistical tests, including Granger causality tests, to analyze the relationships between different features and game outcomes.
* **Data Visualization:** Generates visualizations to illustrate feature importances, time series analysis, and geographical distribution of games.

## Usage
The project consists of several Jupyter Notebooks that perform different tasks:

1. **`Bradley-TerryModel.ipynb`**: This notebook implements a Bradley-Terry model to estimate team strengths based on game results.
2. **`Feature_Engineering.ipynb`**: This notebook performs feature engineering, including calculating travel distances and time zone differences.
3. **`FinalModelRiceDatathon.ipynb`**: This notebook trains and evaluates a machine learning model to predict game outcomes.
4. **`Maps.ipynb`**: This notebook creates interactive maps visualizing team travel patterns.
5. **`getelo (1).ipynb`**: This notebook implements an Elo rating system incorporating team pitching performance.

## Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **Create a conda environment (recommended):**
    ```bash
    conda create -n mlb_prediction python=3.9
    conda activate mlb_prediction
    ```
3.  **Install dependencies:**  (See Dependencies section below)

## Technologies Used
* **Python:** The primary programming language for the project.
* **Pandas:** Used for data manipulation and analysis.
* **NumPy:** Used for numerical computation.
* **Scikit-learn:** Used for machine learning model training and evaluation.
* **Statsmodels:** Used for statistical modeling and analysis.
* **Choix:** Used for Bradley-Terry Model
* **Keras:** Used for Deep Learning Model (in FinalModelRiceDatathon.ipynb)
* **SHAP:** Used for Model Explainability (in FinalModelRiceDatathon.ipynb)
* **Matplotlib & Seaborn:** Used for data visualization.
* **Folium:** Used for creating interactive maps.
* **Requests:** Used for fetching data from external APIs.

## Statistical Analysis
The project employs several statistical methods:

* **Elo Rating System:**  A method for calculating the relative skill levels of players or teams.
* **Bradley-Terry Model:** A statistical model used to estimate the relative strengths of teams based on pairwise comparisons.
* **Granger Causality Test:** A statistical test used to determine whether one time series is useful in forecasting another.
* **Regression Analysis:** Used to model the relationship between features and game outcomes.
* **AUC-ROC Score, Accuracy Score, Log Loss:**  Used for model evaluation.

## Dependencies
The project's dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

*README.md was made with [Etchr](https://etchr.dev)*