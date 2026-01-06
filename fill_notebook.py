
import json
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

def fill_notebook():
    notebook_path = "Sample_ML_Submission_Template.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # --- HELPER TO FIND CELL BY CONTENT OR ID ---
    def find_cell_by_content(content_snippet):
        for i, cell in enumerate(nb.cells):
            if content_snippet.lower() in cell.source.lower():
                return i
        return -1
    
    def set_code(index, code):
        if index != -1 and index + 1 < len(nb.cells):
            # Check if next cell is code, if not, insert one or replace valid one
            if nb.cells[index+1].cell_type == 'code':
                nb.cells[index+1].source = code
            else:
                # If next is markdown (e.g. next section header), insert code before it? 
                # The template usually has code cells after markdown headers.
                # Let's check strict "code" cell type
                nb.cells.insert(index+1, new_code_cell(code))

    # --- 1. Project Summary & Guidelines ---
    # (Leaving mostly as is, maybe filling Project Name)
    idx = find_cell_by_content("# **Project Name**")
    if idx != -1:
        nb.cells[idx].source = "# **Project Name**    - Zomato Restaurant Rating Prediction & Sentiment Analysis\n"

    # --- 2. Import Libraries ---
    idx = find_cell_by_content("### Import Libraries")
    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
"""
    set_code(idx, code_imports)

    # --- 3. Load Dataset ---
    idx = find_cell_by_content("### Dataset Loading")
    code_load = """# Load the datasets
meta_df = pd.read_csv("Zomato Restaurant names and Metadata.csv")
reviews_df = pd.read_csv("Zomato Restaurant reviews.csv")
"""
    set_code(idx, code_load)

    # --- 4. Dataset First View ---
    idx = find_cell_by_content("### Dataset First View")
    code_view = """print("Metadata Head:")
display(meta_df.head())
print("\\nReviews Head:")
display(reviews_df.head())
"""
    set_code(idx, code_view)

    # --- 5. Dataset Rows & Columns ---
    idx = find_cell_by_content("### Dataset Rows & Columns count")
    code_shape = """print("Metadata Shape:", meta_df.shape)
print("Reviews Shape:", reviews_df.shape)
"""
    set_code(idx, code_shape)

    # --- 6. Dataset Info ---
    idx = find_cell_by_content("### Dataset Information")
    code_info = """print("--- Metadata Info ---")
meta_df.info()
print("\\n--- Reviews Info ---")
reviews_df.info()
"""
    set_code(idx, code_info)

    # --- 7. Duplicate Values ---
    idx = find_cell_by_content("#### Duplicate Values")
    code_dup = """print("Metadata Duplicates:", meta_df.duplicated().sum())
print("Reviews Duplicates:", reviews_df.duplicated().sum())
"""
    set_code(idx, code_dup)

    # --- 8. Missing Values ---
    idx = find_cell_by_content("#### Missing Values/Null Values")
    code_miss = """print("--- Metadata Missing ---")
print(meta_df.isnull().sum())
print("\\n--- Reviews Missing ---")
print(reviews_df.isnull().sum())
"""
    set_code(idx, code_miss)

    idx_viz_miss = find_cell_by_content("# Visualizing the missing values")
    code_viz_miss = """plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.heatmap(meta_df.isnull(), cbar=False, cmap='viridis')
plt.title('Metadata Missing Values')

plt.subplot(1, 2, 2)
sns.heatmap(reviews_df.isnull(), cbar=False, cmap='viridis')
plt.title('Reviews Missing Values')
plt.show()
"""
    if idx_viz_miss != -1:
         nb.cells[idx_viz_miss].source = code_viz_miss

    # --- 9. Data Wrangling ---
    idx = find_cell_by_content("### Data Wrangling Code")
    code_wrangle = """# 1. Cleaning Cost in Metadata (remove commas and convert to float)
meta_df['Cost'] = meta_df['Cost'].astype(str).str.replace(',', '').astype(float)

# 2. Merging Datasets
# Metadata has 'Name', Reviews has 'Restaurant'
# We merge reviews with metadata to get cost/cuisine info for each review
df = pd.merge(reviews_df, meta_df, left_on='Restaurant', right_on='Name', how='inner')

# 3. Handle Missing Values in Merged DF if any
df.dropna(subset=['Review', 'Rating', 'Cost'], inplace=True)

# 4. Converting Time to datetime
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# 5. Rating is sometimes object if it has text, let's force numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df.dropna(subset=['Rating'], inplace=True)

print("Merged Dataset Shape:", df.shape)
display(df.head())
"""
    set_code(idx, code_wrangle)

    # --- 10. Visualization Charts ---
    
    # Chart 1: Rating Distribution
    idx = find_cell_by_content("# Chart - 1 visualization code")
    code_c1 = """# Distribution of Ratings
plt.figure(figsize=(8, 5))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
"""
    if idx != -1: nb.cells[idx].source = code_c1
    
    # Chart 2: Cost vs Rating
    idx = find_cell_by_content("# Chart - 2 visualization code")
    code_c2 = """# Cost vs Rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='Rating', y='Cost', data=df, palette='magma')
plt.title('Cost Distribution by Rating')
plt.show()
"""
    if idx != -1: nb.cells[idx].source = code_c2

    # Chart 3: Top 10 Restaurants by Review Count
    idx = find_cell_by_content("# Chart - 3 visualization code")
    code_c3 = """# Top 10 Most Reviewed Restaurants
top_restaurants = df['Restaurant'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_restaurants.values, y=top_restaurants.index, palette='coolwarm')
plt.title('Top 10 Most Reviewed Restaurants')
plt.xlabel('Number of Reviews')
plt.show()
"""
    if idx != -1: nb.cells[idx].source = code_c3

    # Chart 4: Word Cloud
    idx = find_cell_by_content("# Chart - 4 visualization code")
    code_c4 = """# Word Cloud of Reviews
text = " ".join(review for review in df.Review.astype(str))
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most Common Words in Reviews')
plt.show()
"""
    if idx != -1: nb.cells[idx].source = code_c4
    
    # Chart 5: Review Length vs Rating
    idx = find_cell_by_content("# Chart - 5 visualization code")
    code_c5 = """# Review Length vs Rating
df['Review_Length'] = df['Review'].astype(str).apply(len)
plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='Review_Length', data=df, palette='Oranges')
plt.title('Average Review Length by Rating')
plt.show()
"""
    if idx != -1: nb.cells[idx].source = code_c5

    # --- 11. Hypothesis Testing ---
    idx_h1 = find_cell_by_content("### Hypothetical Statement - 1")
    # Usually code follows. Let's find "Perform Statistical Test"
    
    # H1: Expensive restaurants (>800) have higher ratings than cheaper ones (<800)
    idx_stat1 = find_cell_by_content("# Perform Statistical Test to obtain P-Value")
    code_stat1 = """from scipy.stats import ttest_ind

expensive = df[df['Cost'] > 800]['Rating']
cheap = df[df['Cost'] <= 800]['Rating']

t_stat, p_val = ttest_ind(expensive, cheap)
print(f"T-statistic: {t_stat}, P-value: {p_val}")

if p_val < 0.05:
    print("Reject Null Hypothesis: There is a significant difference in ratings.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference.")
"""
    if idx_stat1 != -1: nb.cells[idx_stat1].source = code_stat1

    # H2: Ratings differ significantly across different top cuisines
    # For simplicity, let's test "North Indian" vs "Chinese" mentions if possible, or just Cost ranges again?
    # Let's do H2: Rated 5 vs Rated 1 Review Length
    idx_stat2 = find_cell_by_content("### Hypothetical Statement - 2")
    # Finding the next stat test block manually or via search
    # We need to search from previous index forward to avoid overwriting the same block
    # Simple workaround: find all blocks and assign by index
    stat_indices = []
    for i, cell in enumerate(nb.cells):
        if "# Perform Statistical Test to obtain P-Value" in cell.source:
            stat_indices.append(i)
    
    if len(stat_indices) > 1:
        code_stat2 = """# H2: Is there a significant difference in Review Length between 5-star and 1-star ratings?
stat_5 = df[df['Rating'] == 5]['Review_Length']
stat_1 = df[df['Rating'] == 1]['Review_Length']

t_stat, p_val = ttest_ind(stat_5, stat_1)
print(f"T-statistic: {t_stat}, P-value: {p_val}")
"""
        nb.cells[stat_indices[1]].source = code_stat2

    if len(stat_indices) > 2:
        code_stat3 = """# H3: Correlation betwen Cost and Rating is non-zero?
from scipy.stats import pearsonr
corr, p_val = pearsonr(df['Cost'], df['Rating'])
print(f"Pearson Correlation: {corr}, P-value: {p_val}")
"""
        nb.cells[stat_indices[2]].source = code_stat3


    # --- 12. Handling Missing Values ---
    idx = find_cell_by_content("# Handling Missing Values & Missing Value Imputation")
    code_impute = """# Already dropped critical missing rows in Wrangle step.
# If any remaining numericals have NaNs, fill with median.
df['Cost'].fillna(df['Cost'].median(), inplace=True)
print("Missing values after handling:")
print(df.isnull().sum())
"""
    if idx != -1: nb.cells[idx].source = code_impute

    # --- 13. Categorical Encoding ---
    idx = find_cell_by_content("# Encode your categorical columns")
    code_encode = """# We will use One-Hot Encoding for 'Collections' or potentially 'Cuisines' if we split them.
# For simplicity in this template, let's encode the 'Name' or just rely on text/cost for now.
# But 'Cuisines' is important. It's a list. We can create dummy variables for top 5 cuisines.

top_cuisines = ['North Indian', 'Chinese', 'Continental', 'Italian', 'Biryani']
for cuisine in top_cuisines:
    df[cuisine] = df['Cuisines'].astype(str).apply(lambda x: 1 if cuisine in x else 0)

print("Added Dummy Variables for Top Cuisines")
display(df.head())
"""
    if idx != -1: nb.cells[idx].source = code_encode

    # --- 14. Text Processing ---
    idx = find_cell_by_content("# Remove Punctuations")
    code_clean = """import string
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df['Clean_Review'] = df['Review'].astype(str).apply(clean_text)
print("Text Cleaned.")
"""
    if idx != -1: nb.cells[idx].source = code_clean
    
    idx = find_cell_by_content("# Vectorizing Text")
    code_tfidf = """tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = tfidf.fit_transform(df['Clean_Review']).toarray()
print("TF-IDF Matrix Shape:", X_text.shape)
"""
    if idx != -1: nb.cells[idx].source = code_tfidf

    # --- 15. Feature Selection/Split ---
    idx = find_cell_by_content("# Select your features wisely")
    # Actually, let's skip to Data Splitting
    idx_split = find_cell_by_content("# Split your data to train and test")
    code_split = """# Concatenate TF-IDF features with Numerical features (Cost, Top Cuisines)
X_num = df[['Cost'] + top_cuisines].values
X = np.hstack((X_num, X_text))
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
"""
    if idx_split != -1: nb.cells[idx_split].source = code_split

    # --- 16. Models ---
    
    # Model 1
    idx_m1 = find_cell_by_content("# ML Model - 1 Implementation")
    code_m1 = """# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
"""
    if idx_m1 != -1: nb.cells[idx_m1].source = code_m1
    
    idx_eval1 = find_cell_by_content("# Visualizing evaluation Metric Score chart")
    # This might find multiple. We need the first one after Model 1.
    # Simplified approach: Use unique comments if possible, but they are generic.
    # I'll rely on the order in `idx_m1` context or just set generic output
    # Let's find all indices of "# Visualizing evaluation Metric Score chart"
    eval_indices = []
    for i, cell in enumerate(nb.cells):
        if "# Visualizing evaluation Metric Score chart" in cell.source:
            eval_indices.append(i)
    
    if len(eval_indices) > 0:
        code_eval1 = """print('Linear Regression MSE:', mean_squared_error(y_test, y_pred_lr))
print('Linear Regression R2:', r2_score(y_test, y_pred_lr))

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()
"""
        nb.cells[eval_indices[0]].source = code_eval1

    # Model 2
    idx_m2 = find_cell_by_content("# ML Model - 2") # Searching markdown header or nearby code?
    # The code cell usually says "# ML Model - 1 Implementation" again if copy pasted? 
    # No, the template has distinct sections I should find.
    # Let's find the second occurence of specific strings or assume later in file.
    
    # Actually, simpler to just append code if finding is hard, but I want to fill the template.
    # The template has "ML Model - 2" in markdown. The code cell follows.
    
    # ...Skipping granular "fit" logic for every model and just doing a robust fill.
    
    # Random Forest
    rf_code = """# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print('Random Forest MSE:', mean_squared_error(y_test, y_pred_rf))
print('Random Forest R2:', r2_score(y_test, y_pred_rf))
"""
    # Find cell after "ML Model - 2" markdown
    idx_head_m2 = find_cell_by_content("### ML Model - 2")
    if idx_head_m2 != -1:
        set_code(idx_head_m2 + 2, rf_code) # Skip description headers

    # Model 3
    xgb_code = """# XGBoost Regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
xg_reg.fit(X_train, y_train)
y_pred_xgb = xg_reg.predict(X_test)

print('XGBoost MSE:', mean_squared_error(y_test, y_pred_xgb))
print('XGBoost R2:', r2_score(y_test, y_pred_xgb))
"""
    idx_head_m3 = find_cell_by_content("### ML Model - 3")
    if idx_head_m3 != -1:
        set_code(idx_head_m3 + 2, xgb_code)

    # --- Save ---
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook filled successfully.")

if __name__ == "__main__":
    fill_notebook()
