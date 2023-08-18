# Analysis Report: Yelp Restaurants Data Preprocessing and Model Evaluation

**Authors:**
**Date:** August 16, 2023

## Abstract
This report details the methods, data preprocessing techniques, challenges, and results associated with analyzing the Yelp restaurants dataset. The ultimate goal is to prepare the data for model building and predict if a restaurant is open or not, based on its features.

## Introduction
Yelp datasets provide a vast amount of information about businesses, including their attributes and reviews. Analyzing this data can yield insights into the factors that affect a restaurant's operational status.

## Methods

### Data Used
1. `yelp-restaurants.csv`: Contains attributes of Yelp restaurants.
2. `yelp-reviews.csv`: Contains reviews related to the Yelp restaurants.

### Preprocessing Steps
1. **Loading necessary libraries:** `dask`, `distributed`, `AST`, and `tqdm`.
2. **Loading and basic cleaning:** Remove columns with identifiers, filter for non-restaurants, and deal with missing values.
3. **Attributes mapping:** Extract and convert attribute columns like 'BusinessParking', 'GoodForMeal', 'Ambience' from nested structures to individual columns.
4. **Feature reduction:** Columns with more than 20% missing values are dropped.
5. **Type conversion:** String values are converted to integer types where applicable.
6. **Feature encoding:** Some features are numerically encoded, like the 'Alcohol' column being mapped to ordinal numbers.
7. **Merging dataframes:** The processed restaurant attributes are then merged with their corresponding reviews to produce a comprehensive dataset.

### Feature Selection
1. Correlation heatmaps were used to identify relevant features.
2. Over and under sampling techniques were applied to address class imbalances.
3. `SelectKBest` was used to identify the top 5 relevant features. Avoid overfitting by using too many irrelevant features that might not have strong predictive values.
4. Principal Component Analysis (PCA) was employed to further reduce dimensionality.

### Model Building and Evaluation
1. **Baseline Model:** A RandomForest classifier was trained on the data.
2. **Ensemble Strategy:** Stacking Classifier was used with estimators like DecisionTree, RandomForest, GradientBoosting, and LogisticRegression. Aim to even out errors and eliminate model variance.
3. **Hyperparameter tuning:** GridSearchCV was employed on the Stacking Classifier to find optimal hyperparameters.
4. Models were evaluated using metrics such as accuracy, ROC-AUC, and precision-recall curves.

## Challenges & Improvements

1. **Nested Attributes:** Many attributes were nested, necessitating complex transformations to be usable.
2. **Missing Data:** A large chunk of data had missing values. An approach of dropping columns with more than 20% missing values might be too aggressive.
3. **Class Imbalance:** The target variable, 'is_open', may be imbalanced. Random over-sampling and under-sampling techniques were employed to address this, but more sophisticated methods like SMOTE could also be considered.
4. **Feature Selection:** While `SelectKBest` was used, other techniques or domain expertise could further refine feature selection.
5. **Small Yelp dataset:** Yelp's research dataset does not contain big cities like New York or LA. Yelp's API is only for developers, not researchers. Hard to combine with other alternative data sources that are only publicly available for big cities.
6. **Aspect sentiment analysis of reviews:** Using GPT to get sentiment proved impractical (much better and cheaper tools exist, slow api requests, service breaks often, inconsistent responses)
7. **Google data:**
8. **Time series:**

## Discussion of Results

1. **Feature Importance:** Correlation heatmaps and `SelectKBest` suggest that certain features like 'stars', 'review_count', and 'RestaurantsPriceRange2' play pivotal roles.
2. **PCA:** The scree plot indicated the importance of each principal component. It's valuable to decide how many components to retain.
3. **Model Performance:** The baseline RandomForest model showed decent accuracy and ROC-AUC values. However, the Stacking Classifier, after hyperparameter tuning, showed promise in improving the predictive power.
4. **Precision-Recall Curves:** Both models (RandomForest and Stacking Classifier) were compared using precision-recall curves, which is crucial in imbalanced datasets.
5. **Confusion Matrices:** These matrices provided insights into the true positive, false positive, true negative, and false negative values for the models, allowing for a deeper understanding of model performance.

## Conclusion
Through rigorous preprocessing and modeling techniques, we've managed to build predictive models on the Yelp dataset to determine if a restaurant remains open. The Stacking Classifier seems to be the more promising approach, outperforming the baseline RandomForest model. This study can be beneficial for stakeholders interested in understanding the dynamics of restaurant operations using Yelp data.

Future research can explore more sophisticated preprocessing methods, employ different feature selection techniques, and utilize more advanced modeling strategies for better results.
