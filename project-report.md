# Analysis Report: Yelp Restaurants Data Preprocessing and Model Evaluation

**Authors:**
**Date:** August 16, 2023

## Abstract
This report details the methods, data preprocessing techniques, challenges, and results associated with analyzing the Yelp restaurants dataset. The ultimate goal is to prepare the data for model building and predict if a restaurant is open or not, based on its features.

Machine learning models, including but not limited to DecisionTree, RandomForest, GradientBoosting, and LogisticRegression, are trained on the extracted features to predict the survival outcome of restaurants over a specific time period. The survival outcome is defined as the ability of a restaurant to remain operational or cease operations during the observed time frame. The predictive models are rigorously evaluated and fine-tuned using various performance metrics such as accuracy, precision, recall, and F1-score. Cross-validation techniques are employed to ensure the robustness and generalizability of the models. Additionally, feature importance analysis is conducted to identify the most influential factors contributing to a restaurant's survival rate.

## Introduction
In today's dynamic culinary landscape, the restaurant industry is characterized by a high level of competitiveness and volatility. Understanding the factors that contribute to the success or failure of restaurants is crucial for both entrepreneurs and investors. This project aims to explore the potential of utilizing restaurant reviews from Yelp, a popular user-generated review platform, to predict the survival rate of restaurants. 

Besides reviews, Yelp datasets also provide a vast amount of information about businesses, including their attributes such as opening hours and price range. Analyzing this data can yield insights into the factors that affect a restaurant's operational status.

This paper delves into the intriguing realm of restaurant survival prediction by harnessing the combined power of restaurant reviews and attributes sourced from Yelp. We embark on a journey to uncover the latent insights concealed within the vast expanse of online reviews and intrinsic characteristics of restaurants. By leveraging advanced data analysis techniques, including Natural Language Processing (NLP) and machine learning, we aim to construct predictive models that shed light on the factors influencing the longevity of restaurants in this highly competitive industry.

(Add something about NLP with OpenAI API here)

## Literature review
### “How was your meal?” Examining customer experience using Google maps reviews
Study Objective: Predicting restaurant performance using customer reviews in the UK.
Methods:
Data collection from Google Maps.
Sentiment analysis using VADER tool.
Dataset: 5,010 restaurants / 935,386 reviews.
Results:
Food: Most influential attribute for 5-star ratings.
Service: Key in minimizing 1-star reviews.
Atmosphere: Significant in elevating 2-star to 3-star ratings.
Value: Negative impact on 5-star ratings, positive impact on lower ratings.
Valence Analysis:
Alcoholic beverages: Positive impact on customer experiences.
Dietary options: High valence, satisfying customers' requirements.
Basic food items: Relatively low valence.
Implications:
Improve food, service, atmosphere, and value for enhanced performance.
Prioritize improvements based on specific rating levels.
Consider the impact of alcoholic beverages and dietary options.

### Restaurant survival prediction using customer-generated content
Objective: Predict restaurant survival using customer-generated content
Methods: Aspect-Based Sentiment Analysis (ABSA) combined with Conditional Survival Forest (CSF) algorithm
Dataset: Large-scale Yelp dataset w/ restaurant reviews, ratings, and relevant information
ABSA-CSF Model:
Extract sentiment from reviews based on important aspects
Predict survival w/ extracted features
Key Findings:
ABSA-CSF model outperforms other models
Location and ‘tastiness’ sentiment crucial for survival prediction
Factors of importance vary by restaurant type
Implications:
Importance of online reviews in business survival
Allocate resources based on essential factors
Conclusion: ABSA-CSF model effectively predicts restaurant survival using online reviews

### Predicting International Restaurant Success with Yelp
Objective: Identify key features for restaurant success across different countries using the Yelp Dataset.
Methods:
Feature selection: Univariate feature selection 
Classification models: Naive Bayes, logistic regression, SVM, decision trees, random forest, GDA.
Textual analysis: Naive Bayes classifier for review analysis.
Dataset:
25,071 restaurants from the US, UK, Canada, and Germany.
Yelp Dataset Challenge data.
Results:
Commonly important features: Street parking, reservations, review count, ambience, noise level, attire.
Country-specific features: Divey ambience (US), parking (North America), alcohol availability (Europe).
Best-performing model: GDA with test accuracies ranging from 55% to 60%.

## Methods

### Data Used
1. `yelp-restaurants.csv`: Contains attributes of Yelp restaurants.
2. `yelp-reviews.csv`: Contains reviews related to the Yelp restaurants.

### Preprocessing Steps
1. **Loading necessary libraries:** `dask`, `distributed`, `AST`, and `tqdm`.
   - The Dask library in Python provides a powerful and flexible framework for parallel and distributed computing. It is particularly beneficial when working with large datasets that cannot fit into memory or when dealing with computationally intensive tasks. Dask enables parallel execution of tasks, which can significantly speed up computations. It automatically divides the data and tasks into smaller chunks and processes them concurrently, utilizing all available CPU core
  - Tqdm library provides progress bars to loops and iterators, allowing the tracking of progress, especially for dealing with large datasets and lengthy computations

3. **Loading and basic cleaning:** Remove columns with identifiers, filter for non-restaurants, and deal with missing values.
4. **Filter popular restaurants:** Only take restaurants that have more than 50 reviews. Restaurants with too few reviews might have risk of bias opinions
5. **Attributes mapping:** Extract and convert attribute columns like 'BusinessParking', 'GoodForMeal', 'Ambience' from nested structures to individual columns.
6. **Feature reduction:** Columns with more than 20% missing values are dropped.
7. **Type conversion:**  String values are converted to integer types where applicable using one hot encoding or manual conversion. Some categories are self-translated into numerical based on their values instead of one hot encoding to avoid having too many columns. For example, noise level of quiet, average, loud and very loud would have the value of 0, 1, 2, 3.
8. **Feature encoding:** Some features are numerically encoded, like the 'Alcohol' column being mapped to ordinal numbers.
9. **Merging dataframes:** The processed restaurant attributes are then merged with their corresponding reviews to produce a comprehensive dataset.

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
