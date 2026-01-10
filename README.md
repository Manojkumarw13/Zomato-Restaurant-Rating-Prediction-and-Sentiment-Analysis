# Zomato Restaurant Rating Prediction & Sentiment Analysis

## 📌 Project Overview
The restaurant industry in India, particularly in bustling metros like Hyderabad, is characterized by intense competition and evolving customer preferences. In this data-driven era, understanding the factors that drive high ratings is crucial for business survival. 

This project, **"Zomato Restaurant Rating Prediction & Sentiment Analysis,"** leverages Machine Learning and Natural Language Processing (NLP) to decode the "Voice of the Customer." By bridging the gap between unstructured text reviews and structured ratings, we provide a predictive model that allows restaurant owners to proactively manage their reputation.

## 📂 Project Structure
- **`Sample_ML_Submission_Template.ipynb`**: The main Jupyter Notebook containing the end-to-end analysis, from data wrangling to model deployment.
- **`Zomato Restaurant names and Metadata.csv`**: Metadata for 105 restaurants (Cost, Links, Cuisines, Timings).
- **`Zomato Restaurant reviews.csv`**: A collection of **10,000+ customer reviews** linked to the restaurants.
- **`Zomato project.pptx`**: A presentation summarizing the project findings.
- **`zomato_rating_model.pkl`**: The trained Machine Learning model saved for future use.

## 💡 Key Insights & Business Impact
Our extensive Exploratory Data Analysis (EDA) and Hypothesis Testing revealed critical insights. Here is a breakdown of what we found and **where** it came from:

### 1. Customer Preferences & Product Quality
*   **Insight**: "North Indian" and "Chinese" are overwhelmingly the most popular cuisines served and reviewed.
    *   *Source*: **Chart - 6 (Top 10 Cuisine Types)**
    *   *Business Impact*: New entrants should include these "safe bet" cuisines in their menu to capture the mass market.
*   **Insight**: Core product terms like "food", "tasty", "chicken", and "good" dominate the reviews, appearing far more frequently than "service" or "ambiance".
    *   *Source*: **Chart - 9 (Word Cloud)**
    *   *Business Impact*: Food quality is the primary driver of satisfaction. Marketing should focus on taste and quality rather than just decor.

### 2. User Engagement & Behavior
*   **Insight**: A small group of "Super Foodies" contributes a disproportionate number of reviews.
    *   *Source*: **Chart - 11 (Top 10 Most Active Reviewers)**
    *   *Business Impact*: Engaging these key influencers with exclusive tasting events can generate significant organic reach and credibility.
*   **Insight**: Reviews with pictures tend to have higher ratings. Happy customers like to show off their food.
    *   *Source*: **Chart - 10 (Pictures vs Rating)** & **Hypothesis Test 3**
    *   *Business Impact*: Restaurants should improve the visual appeal ("Instagrammability") of their plating and incentivize users to upload photos.
*   **Insight**: Extreme ratings (1.0 and 5.0) are associated with longer review lengths. Customers write detailed essays when they feel strongly (either delight or rage).
    *   *Source*: **Chart - 2 (Review Length vs Rating)** & **Hypothesis Test 2**
    *   *Business Impact*: Long reviews are goldmines for feedback. Automated sentiment analysis on these can yield specific, actionable improvements.

### 3. Operational Intelligence
*   **Insight**: Peak review posting times occur post-lunch (2-3 PM) and post-dinner (9-11 PM).
    *   *Source*: **Chart - 13 (Hour of Review)**
    *   *Business Impact*: Social media support teams should be most active during these windows to respond instantly to feedback.
*   **Insight**: Review volume has grown exponentially in recent years, signaling massive digital adoption.
    *   *Source*: **Chart - 7 (Trend of Reviews over Years)**
    *   *Business Impact*: Online Reputation Management (ORM) is no longer optional; it is a critical business function.

### 4. Financial Dynamics
*   **Insight**: There is a positive correlation between Cost and Rating. Premium restaurants generally enjoy slightly higher ratings.
    *   *Source*: **Chart - 4 (Cost vs Rating)** & **Hypothesis Test 1**
    *   *Business Impact*: Higher prices create an expectation of quality, but if met (ambiance + service), they lead to better ratings. Budget restaurants must work harder to "wow" customers to achieve similar scores.

## 🛠️ Tech Stack & Methodology
- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `xgboost`, `nltk`
- **Techniques**: 
    - **EDA**: Univariate/Bivariate analysis, Hypothesis Testing (T-test).
    - **NLP**: Text Cleaning, Stopword Removal, Lemmatization, TF-IDF Vectorization.
    - **Machine Learning**: 
        - **Data Splitting**: Strict train-test separation to separate proper feature engineering.
        - **Models**: Linear Regression, **XGBoost Regressor** (Tuned), **Random Forest Regressor** (Tuned).

## 🏆 Model Performance
We addressed the complex problem of predicting ratings from text and metadata.
- **Champion Model**: **XGBoost / Random Forest Regressor**
- **Why**: Ensemble methods successfully captured the non-linear relationship between sentiment-heavy text vectors (TF-IDF) and the numerical rating, outperforming the linear baseline.
- **Features Used**: `Votes`, `Cost`, `Engagement_Score`, `Review_Text_Vectors`.

## 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn nltk xgboost`.
3.  Open `Sample_ML_Submission_Template.ipynb` in Jupyter/Colab.
4.  Run all cells. The notebook is configured with `random_state=42` for reproducible results.

## 📜 License
This project is for educational/portfolio purposes.
