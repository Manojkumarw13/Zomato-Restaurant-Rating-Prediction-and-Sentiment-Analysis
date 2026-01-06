
import pandas as pd

def check_merge_quality():
    try:
        # Load datasets
        meta_df = pd.read_csv("Zomato Restaurant names and Metadata.csv")
        reviews_df = pd.read_csv("Zomato Restaurant reviews.csv")
        
        # Normalize names for better matching (lowercase, strip)
        meta_names = set(meta_df['Name'].str.lower().str.strip())
        review_names = set(reviews_df['Restaurant'].str.lower().str.strip())
        
        # Check overlap
        common = meta_names.intersection(review_names)
        missing_in_meta = review_names - meta_names
        missing_in_reviews = meta_names - review_names
        
        print(f"Total Unique in Metadata: {len(meta_names)}")
        print(f"Total Unique in Reviews: {len(review_names)}")
        print(f"Common Restaurants: {len(common)}")
        print(f"Missing in Metadata: {len(missing_in_meta)}")
        print(f"Missing in Reviews: {len(missing_in_reviews)}")
        
        if len(missing_in_meta) > 0:
            print("\nExample Missing in Metadata:", list(missing_in_meta)[:5])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_merge_quality()


Except for the GitHub link cell fill all the cells in the template I should have to finish it.
Understand the problem statements and the business problems and

Fill the project summary, problem statement, and understand your variables, check unique values for each variable, then fill all the 15 charts cells and also the 3 questions mentioned below every chart and then do all three hypothesis testing and answer the Questions mentioned below that, in feature Engineering