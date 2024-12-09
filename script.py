import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
#from preprocess import DataPreprocessor

drop_cols = ['customerID']
internet_service_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
phone_service_cols = ['MultipleLines']
# After col formatting variables
target = 'churn'
num_col = ['tenure', 'totalcharges', 'monthlycharges']

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop cols
df = df.drop(drop_cols, axis = 1)

# Set to null wrong values from TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan).astype('float')

# Set SeniorCitizen to Yes or No like other binary columns
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'}).astype('object')

# Check for nulls
null_check = df.isnull().sum()
print("Null values in columns:")
print(null_check[null_check > 0])

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\nTotal duplicate rows: {duplicate_count}")
if duplicate_count:
    df = df.drop_duplicates()

# Split by service
df_internet = df[df['InternetService'] != 'No'].copy()
df_internet = df_internet.drop(columns=phone_service_cols)
df_phone = df[df['PhoneService'] != 'No'].copy()
df_phone = df_phone.drop(columns=internet_service_cols + ['PhoneService'])

# Turn cat to booleans
def preprocess_cat(df: pd.DataFrame) -> pd.DataFrame:

    processed_df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:

        unique_values = processed_df[col].unique()
    
        # If binary column
        if len(unique_values) == 2:
            if set(unique_values) == {'Yes', 'No'}:
                processed_df[col] = processed_df[col].map({'Yes': True, 'No': False}).astype('boolean')
            elif col == 'gender':
                processed_df['is_male'] = processed_df[col].map({'Male': True, 'Female': False}).astype('boolean')
                processed_df = processed_df.drop('gender', axis = 1)
            else:
                target_value = unique_values[1]
                processed_df[f'{col}_is_{target_value.lower()}'] = (processed_df[col] == target_value)
                processed_df.drop(columns=[col], inplace=True)
        
        # If multiclass column
        elif len(unique_values) > 2:
            # One-hot encode
            one_hot = pd.get_dummies(processed_df[col], prefix=col)
            processed_df = pd.concat([processed_df.drop(columns=[col]), one_hot], axis=1)
    
    return processed_df


df_internet = preprocess_cat(df_internet)
df_phone = preprocess_cat(df_phone)

# Format feature names
df_internet.columns = df_internet.columns.str.lower().str.replace(' ', '_')
df_phone.columns = df_phone.columns.str.lower().str.replace(' ', '_')

# Train test split
df_train_internet, df_test_internet = train_test_split(df_internet, test_size=0.2, stratify=df_internet[target])
df_train_phone, df_test_phone = train_test_split(df_phone, test_size=0.2, stratify=df_phone[target])

# Imputing missing values
imputer = KNNImputer(n_neighbors=10)
df_train_internet[:] = imputer.fit_transform(df_train_internet)
df_test_internet[:] = imputer.transform(df_test_internet)
df_train_phone[:] = imputer.fit_transform(df_train_phone)
df_test_phone[:] = imputer.transform(df_test_phone)

def revert_back_to_booleans(df):
    bool_features = [col for col in df.columns if col not in num_col]
    for col in bool_features:
        if col in df.columns:
            df[col] = df[col].astype('boolean')
    return df

df_train_internet = revert_back_to_booleans(df_train_internet)
df_test_internet = revert_back_to_booleans(df_test_internet)
df_train_phone = revert_back_to_booleans(df_train_phone)
df_test_phone = revert_back_to_booleans(df_test_phone)

print(df_train_internet.head())