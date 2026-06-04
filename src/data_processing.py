import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

# =====================================================================
# 1. CUSTOM TRANSFORMER FOR TIME COMPONENTS
# =====================================================================
class XenteDateTimeExtractor(BaseEstimator, TransformerMixin):
    """Pipeline step to extract time structural components from timestamps."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'transactionstarttime' in X_copy.columns:
            dates = pd.to_datetime(X_copy['transactionstarttime'])
            X_copy['TransactionHour'] = dates.dt.hour
            X_copy['TransactionDay'] = dates.dt.day
            X_copy['TransactionMonth'] = dates.dt.month
            X_copy['TransactionYear'] = dates.dt.year
            X_copy = X_copy.drop(columns=['transactionstarttime'])
        return X_copy

# =====================================================================
# 2. PIPELINE BUILDER ENGINE
# =====================================================================
def build_xente_processing_pipeline(numerical_cols, categorical_cols):
    """Chains together imputation, date handling, scaling, and categorical encoders."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline(steps=[
        ('time_extractor', XenteDateTimeExtractor()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

# =====================================================================
# 3. TASK 4: PROXY TARGET VARIABLE ENGINEERING FUNCTION
# =====================================================================
def engineer_proxy_target(df):
    """
    Calculates RFM metrics per customer, segments them using K-Means,
    and isolates the least engaged segment as the high-risk target proxy.
    """
    print("🎯 Engineering synthetic proxy target labels via RFM K-Means...")
    
    # Copy to avoid side-effects
    df_clean = df.copy()
    df_clean['transactionstarttime'] = pd.to_datetime(df_clean['transactionstarttime'])
    
    # Establish snapshot date consistently
    snapshot_date = df_clean['transactionstarttime'].max()
    
    # Calculate RFM values per CustomerId
    rfm = df_clean.groupby('customerid').agg({
        'transactionstarttime': lambda x: (snapshot_date - x.max()).days, # Recency
        'transactionid': 'count',                                         # Frequency
        'amount': 'sum'                                                   # Monetary
    }).rename(columns={
        'transactionstarttime': 'recency',
        'transactionid': 'frequency',
        'amount': 'monetary'
    })
    
    # Scale log-transformed metrics for stable K-Means calculations
    rfm_log = np.log1p(rfm.clip(lower=0))
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # Segment into 3 distinct behavioral groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify the highest risk cluster programmatically: 
    # High Risk = high recency (inactive), low frequency, low monetary value
    cluster_profiles = rfm.groupby('cluster').mean()
    
    # Risk score metric: higher recency rank, lower frequency/monetary rank
    # Alternatively, find the cluster with the maximum mean recency days
    high_risk_cluster = cluster_profiles['recency'].idxmax()
    
    # Generate binary flag
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    
    # Map back to original dataset records using customer identifiers
    target_map = rfm['is_high_risk'].to_dict()
    df_clean['is_high_risk'] = df_clean['customerid'].map(target_map)
    
    print("Target variable mapping complete! High-risk class breakdown:")
    print(df_clean['is_high_risk'].value_counts(normalize=True))
    
    return df_clean

# =====================================================================
# 4. RUNTIME EXECUTION
# =====================================================================
if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.csv") 
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    if not os.path.exists(DATA_PATH):
        print("Error: Could not locate your transaction file at: " + str(DATA_PATH))
    else:
        raw_df = pd.read_csv(DATA_PATH)
        raw_df.columns = raw_df.columns.str.strip().str.lower()
        
        # Step 1: Run Task 4 Proxy Target Generation
        df_with_target = engineer_proxy_target(raw_df)
        
        # Isolate baseline training features and new label column
        # Remove unique identifiers and drop 'fraudresult'/'pricingstrategy' to preserve validation integrity
        X = df_with_target.drop(columns=['is_high_risk', 'fraudresult', 'pricingstrategy', 'transactionid', 'batchid', 'accountid', 'subscriptionid', 'customerid'], errors='ignore')
        y = df_with_target['is_high_risk']

        # Setup feature mapping layout
        base_numerical = ['amount', 'value']
        extended_time_numerical = base_numerical + ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
        categorical_features = ['currencycode', 'countrycode', 'providerid', 'productcategory', 'productid', 'channelid']

        # Build feature preprocessing pipeline
        pipeline = build_xente_processing_pipeline(extended_time_numerical, categorical_features)
        
        # Train-test splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Fit transformations
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)

        # Save model-ready array dumps out to processed folder
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_processed)
        np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_processed)
        y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

        print("\n🎉 Task 4 Integrated Production Pipeline Executed Successfully!")
        print("Processed Train Dimensions: " + str(X_train_processed.shape))
        print("Processed Target Dimensions: " + str(y_train.shape))