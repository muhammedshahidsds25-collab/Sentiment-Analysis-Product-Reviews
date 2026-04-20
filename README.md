# Sentiment Analysis of Product Reviews

## Team Members
- ANU GOPAL V
- RASHA K K  
- MUHAMMED SHAHID S  

## Project Topic
Sentiment Analysis of Product Reviews using TF-IDF and Machine Learning Classifiers


---

## 📊 Exploratory Data Analysis (EDA)

- Analyzed sentiment distribution across reviews
- Identified strong class imbalance (Negative dominant)
- Examined review length patterns
- Observed variation in text characteristics across sentiments

### Key Findings
- Negative reviews dominate the dataset
- Neutral class is very small
- Data imbalance may affect model performance

 ## ⚠️ Data Challenges

- Class imbalance between sentiment classes
- Presence of noise (URLs, special characters)
- Short and inconsistent review texts

## Modeling Approach
- Applied TF-IDF Vectorization with unigrams and bigrams
- Limited features to reduce noise and improve performance
- Added linguistic features:
  - Sentence length
  - Punctuation density
  - Average word length
  - Presence of negation words
- Combined sparse (TF-IDF) and dense features

## Feature Selection
- Used Chi-Square (χ²) test to select top features
- Reduced dimensionality to improve model efficiency
- Selected top 500 most relevant features
  
## Machine Learning Models

The following classifiers were trained and evaluated:
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Gradient Boosting Classifier

## Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- Macro F1 Score (important due to class imbalance)

## Best Model
Logistic Regression achieved the best performance:
- Accuracy: ~0.79
- Macro F1 Score: ~0.63
It provided the best balance across all sentiment classes.



  Results and Evaluation Analysis

The performance of the machine learning models was evaluated using multiple classification metrics and visual analysis techniques. The evaluation process ensured a comprehensive understanding of model behavior across different sentiment classes.

🔹 Evaluation Metrics

- Accuracy – Measures overall correctness  
- Precision – Measures correctness of positive predictions  
- Recall – Measures ability to find all relevant instances  
- Macro F1 Score – Balanced evaluation across all classes  

🔹 Model Comparison

Models used:
- Logistic Regression  
- Multinomial Naive Bayes  
- Linear SVM  
- Gradient Boosting  

Best model selected based on Macro F1 Score.

🔹 Confusion Matrix Analysis

Confusion matrix was used to understand:
- Correct predictions  
- Misclassifications  
- Class-wise performance  

🔹 Key Observations

- Accuracy alone is not enough  
- Macro F1 gives better insight  
- Model performance varies across sentiment classes  







