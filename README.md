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

## 📊 Comprehensive Business Analysis

### 📌 1. City & Location Analysis
- **Metro cities** (Bangalore, Mumbai, Delhi NCR, Hyderabad) have the highest number of restaurants and transactions.
- These cities also show **higher average ratings** and **higher spending per order**.
- Certain localities inside these cities act as **restaurant density hubs**, making them ideal for advertising and promotions.

**Business Meaning**: Zomato's revenue and growth are heavily concentrated in metro cities and premium localities.

### 📌 2. Restaurant Type Analysis
- **Quick Bites** and **Casual Dining** dominate the platform in volume.
- **Fine Dining** has fewer restaurants but **much higher spending** per customer.
- **Cafes & Dessert Parlors** have strong engagement but lower ticket size.

**Business Meaning**: Zomato earns volume from Quick Bites and profits from Fine Dining.

### 📌 3. Cost for Two (Price Analysis)
- Most customers spend between **₹300–₹700**.
- Restaurants priced above **₹1500** form a high-value niche segment.
- Very cheap restaurants generate traffic but **low profit per order**.

**Business Meaning**: ₹300–₹700 is Zomato's sweet spot for offers, ads, and restaurant onboarding.

### 📌 4. Ratings & Reviews Analysis
- Restaurants rated **4.0+** receive:
  - ✅ More orders
  - ✅ More reviews
  - ✅ Better customer loyalty
- Restaurants below **3.5** struggle to attract customers.

**Business Meaning**: Customer trust and revenue are driven primarily by ratings.

### 📌 5. Online Order Analysis
- Restaurants with **online ordering enabled**:
  - ✅ Have higher ratings
  - ✅ Have more reviews
  - ✅ Receive more orders
- Offline-only restaurants are falling behind.

**Business Meaning**: Online ordering is a critical growth driver for restaurants and Zomato.

### 📌 6. Table Booking Analysis
- Restaurants with table booking are usually:
  - 💎 Higher priced
  - ⭐ Better rated
  - 🏆 More premium

**Business Meaning**: Table booking indicates high-end dining behavior and premium customers.

### 📌 7. Delivery Analysis
- Restaurants offering **delivery** get:
  - 📈 More transactions
  - 💬 More customer engagement
- Dine-in only restaurants grow slower.

**Business Meaning**: Delivery availability directly increases revenue.

### 📌 8. Cuisine Analysis
**Top demand cuisines:**
- 🥘 North Indian
- 🍜 Chinese
- 🍔 Fast Food
- 🍛 South Indian
- 🍗 Biryani

**Premium cuisines** (Italian, Continental, Japanese) attract high-spending users.

**Business Meaning**: Indian + Chinese = volume, International cuisines = premium revenue.

### 📌 9. Correlation Analysis
- Higher price → Higher ratings
- Higher ratings → More reviews
- More reviews → More orders

**Business Meaning**: Quality drives trust, and trust drives sales.

### 📌 10. Customer Behavior Patterns
**Customers prefer:**
- ✅ High-rated restaurants
- ✅ Online ordering
- ✅ Restaurants with many reviews

**Customers avoid:**
- ❌ Low-rated places
- ❌ Restaurants with no reviews

**Business Meaning**: Customers behave in a risk-avoiding, trust-based way.

### 📌 11. Strategic Business Insights

| Data Finding | Zomato Strategy |
|-------------|-----------------|
| Metros dominate | Focus ads and partnerships there |
| ₹300–₹700 most popular | Optimize offers in this price range |
| 4+ ratings matter | Promote high-rated restaurants |
| Delivery boosts sales | Push delivery onboarding |
| Reviews increase trust | Incentivize reviews |
| Premium restaurants earn more | Offer premium memberships |

---

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
