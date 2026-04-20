# Sentiment Analysis of Product Reviews

## Team Members
- ANU GOPAL V
- RASHA K K  
- MUHAMMED SHAHID S  

## Project Topic
Sentiment Analysis of Product Reviews using TF-IDF and Machine Learning Classifiers

 📁 Project Structure

- 01_data_eda.ipynb – Data analysis
- 02_modeling.ipynb – Model building
- 03_results.ipynb – Results and evaluation
- sentiment_analysis.py – Main pipeline
- streamlit_app.py – Web application
- plots/ – Visualizations
- saved_models/ – Trained models


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


📊 Results and Evaluation Analysis

The performance of all machine learning models was evaluated using multiple metrics to ensure a fair comparison, especially due to the **imbalanced dataset**.

---

 🔹 Evaluation Metrics

- **Accuracy** – Overall correctness of predictions  
- **Precision** – Correctness of predicted labels  
- **Recall** – Ability to identify all relevant instances  
- **⭐ Macro F1 Score** – Balanced metric across all classes (used as primary metric)

---

 🔹 Model Comparison

| Model                  | Accuracy | Macro F1 |
|-----------------------|----------|----------|
| Logistic Regression ⭐ | ~0.79    | ~0.63    |
| Linear SVM            | ~0.88    | ~0.59    |
| Gradient Boosting     | ~0.87    | ~0.60    |
| Multinomial NB        | ~0.85    | ~0.56    |

> 👉 **Logistic Regression selected as best model based on Macro F1 Score**

---

🔹 Confusion Matrix Analysis

- Strong performance on **Negative class**  
- Good balance on **Positive class**  
- Poor detection of **Neutral class** due to low representation  

---

 🔹 Key Observations

- Accuracy alone is misleading due to class imbalance  
- Macro F1 provides a more reliable evaluation  
- Logistic Regression offers the best balance across classes  
- Neutral class remains the most challenging to predict
  
  

 🌐 Deployment

The trained model is deployed using a Streamlit web application.

Features of the deployed system:
- Classifies product reviews into Positive, Neutral, and Negative
- Supports single review prediction
- Allows batch prediction using CSV upload
- Displays prediction confidence scores

This makes the model accessible as an interactive and user-friendly application.  




▶️ How to Run the Project

1. Install dependencies:
   pip install -r requirements.txt

2. Run the main pipeline:
   python sentiment_analysis.py <dataset.csv>

3. Run the Streamlit app:
   streamlit run streamlit_app.py







