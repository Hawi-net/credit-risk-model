import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================================
# 1. CUSTOM TRANSFORMER FOR RATIO ENGINEERING
# =====================================================================
class BreastCancerFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates interaction features without dropping data prematurely."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Create an aggregate interaction feature: Area-to-Radius Ratio
        if 'mean_area' in X_copy.columns and 'mean_radius' in X_copy.columns:
            X_copy['area_to_radius_ratio'] = X_copy['mean_area'] / (X_copy['mean_radius'] + 1e-5)
        return X_copy

# =====================================================================
# 2. PIPELINE GENERATION FUNCTION
# =====================================================================
def build_preprocessing_pipeline(final_numerical_features):
    """
    Creates a single fitted sklearn Pipeline object.
    Only processes and keeps the specified features, effectively dropping 
    the multi-collinear ones identified in EDA.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Package into a ColumnTransformer. 
    # Remainder='drop' ensures anything not in final_numerical_features is dropped!
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, final_numerical_features)
        ],
        remainder='drop' 
    )

    # Master pipeline chain
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', BreastCancerFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

# =====================================================================
# 3. RUNTIME EXECUTION
# =====================================================================
if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Breast_Cancer_data.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Target data file not found at {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded Raw Dataset: {df.shape}")

        X = df.drop(columns=['diagnosis'])
        y = df['diagnosis']

        # Define columns we want to KEEP and transform based on EDA findings
        # We explicitly omit 'mean_perimeter' and 'mean_area' here to eliminate collinearity
        keep_features = ['mean_radius', 'mean_texture', 'mean_smoothness', 'area_to_radius_ratio']

        # Build pipeline using our explicitly curated features list
        pipeline = build_preprocessing_pipeline(final_numerical_features=keep_features)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Fit and transform
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        X_test_processed = pipeline.transform(X_test)

        # Save arrays
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_processed)
        np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_processed)
        y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

        print(f"🎉 Task 3 Complete! Processed arrays exported to {OUTPUT_DIR}")
        print(f"Processed Train Shape: {X_train_processed.shape}")