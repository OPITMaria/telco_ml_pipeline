import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
    
    def load_data(self):
        """Load data and perform initial cleaning"""
        self.df = pd.read_csv(self.filepath)
        
        # Drop customer ID column
        self.df = self.df.drop('customerID', axis=1)
        
        # Clean TotalCharges column
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(" ", np.nan).astype('float')
        
        # Convert SeniorCitizen to Yes/No
        self.df['SeniorCitizen'] = self.df['SeniorCitizen'].map({1: 'Yes', 0: 'No'}).astype('object')

        # tenure as float
        self.df['tenure'] = self.df['tenure'].astype('float')
        
        return self.df
    
    def check_data_quality(self):
        """Check for null values and duplicates"""
        null_check = self.df.isnull().sum()
        print("Null values in columns:")
        print(null_check[null_check > 0])
        
        duplicate_count = self.df.duplicated().sum()
        print(f"\nTotal duplicate rows: {duplicate_count}")
        
        if duplicate_count:
            self.df = self.df.drop_duplicates()
        
        return self.df

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.internet_service_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        self.phone_service_cols = ['MultipleLines']
    
    def split_by_service(self):
        """Split dataframe by internet and phone services"""
        df_internet = self.df[self.df['InternetService'] != 'No'].copy()
        df_internet = df_internet.drop(columns=self.phone_service_cols)
        
        df_phone = self.df[self.df['PhoneService'] != 'No'].copy()
        df_phone = df_phone.drop(columns=self.internet_service_cols + ['PhoneService'])

        # Internet-only customers (InternetService != 'No' AND PhoneService == 'No')
        df_internet_only = self.df[(self.df['InternetService'] != 'No') & (self.df['PhoneService'] == 'No')].copy()
        
        return df_internet, df_phone, df_internet_only

    
    @staticmethod
    def preprocess_categorical(df):
        """Convert categorical columns to boolean or one-hot encoded"""
        processed_df = df.copy()
        cat_cols = df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            unique_values = processed_df[col].unique()
            
            # Binary columns
            if len(unique_values) == 2:
                if set(unique_values) == {'Yes', 'No'}:
                    processed_df[col] = processed_df[col].map({'Yes': True, 'No': False}).astype('boolean')
                elif col == 'gender':
                    processed_df['is_male'] = processed_df[col].map({'Male': True, 'Female': False}).astype('boolean')
                    processed_df = processed_df.drop('gender', axis=1)
                else:
                    target_value = unique_values[1]
                    processed_df[f'{col}_is_{target_value.lower()}'] = (processed_df[col] == target_value)
                    processed_df = processed_df.drop(columns=[col])
            
            # Multiclass columns
            elif len(unique_values) > 2:
                one_hot = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df.drop(columns=[col]), one_hot], axis=1)
        
        return processed_df

# class DataSplitter:
#     def __init__(self, df_internet, df_phone, test_size: float):
#         self.df_internet = df_internet
#         self.df_phone = df_phone
#         self.num_col = ['tenure', 'totalcharges', 'monthlycharges']
#         self.target = 'churn'
#         self.test_size = test_size
    
#     def train_test_split(self):
#         common_index = self.df_phone.index.intersection(self.df_internet.index)
#         df_phone_internet_common_train = self.df_phone.loc[common_index]
#         y = df_phone_internet_common_train[self.target]
#         train_index, test_index = train_test_split(common_index, test_size=0.2, stratify=y, random_state=42)

#         df_phone_common_train = self.df_phone.loc[train_index]
#         df_internet_common_train = self.df_internet.loc[train_index]
#         df_phone_common_test = self.df_phone.loc[test_index]
#         df_internet_common_test = self.df_internet.loc[test_index]

#         df_phone_only = self.df_phone.drop(index = common_index)
#         df_internet_only = self.df_internet.drop(index = common_index)

#         df_phone_only_train, df_phone_only_test = train_test_split(
#             df_phone_only, 
#             test_size=0.2, 
#             stratify=df_phone_only[self.target],
#             random_state=42
#         )

#         df_internet_only_train, df_internet_only_test = train_test_split(
#             df_internet_only, 
#             test_size=0.2, 
#             stratify=df_internet_only[self.target],
#             random_state=42
#         )

#         df_train_internet = pd.concat([df_internet_common_train, df_internet_only_train])
#         df_test_internet = pd.concat([df_internet_common_test, df_internet_only_test])
#         df_train_phone = pd.concat([df_phone_common_train, df_phone_only_train])
#         df_test_phone = pd.concat([df_phone_common_test, df_phone_only_test])

              
#         return df_train_internet, df_test_internet, df_train_phone, df_test_phone

# DATASPLITTER, MARIA'S EDITS
class DataSplitter:
    def __init__(self, df_internet=None, df_phone=None, test_size: float = 0.2):
        self.df_internet = df_internet
        self.df_phone = df_phone
        self.num_col = ['tenure', 'totalcharges', 'monthlycharges']
        self.target = 'churn'
        self.test_size = test_size

    def split_dataset(self, df):
        """Splits a given dataset into training and testing sets."""
        return train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[self.target] if self.target in df else None,
            random_state=42
        )

    def train_test_split(self):
        """Splits internet and phone datasets into train/test sets."""
        df_train_internet, df_test_internet = None, None
        df_train_phone, df_test_phone = None, None

        # Process internet dataset if provided
        if self.df_internet is not None:
            if self.df_phone is not None:
                # Handle common indices
                common_index = self.df_phone.index.intersection(self.df_internet.index)
                df_internet_common = self.df_internet.loc[common_index]
                df_internet_only = self.df_internet.drop(index=common_index)
                train_index, test_index = train_test_split(
                    common_index,
                    test_size=self.test_size,
                    stratify=self.df_phone.loc[common_index, self.target] if self.target in self.df_phone else None,
                    random_state=42
                )
                # Split the common subset
                df_train_internet_common = df_internet_common.loc[train_index]
                df_test_internet_common = df_internet_common.loc[test_index]
                # Split internet-only subset
                df_train_internet_only, df_test_internet_only = self.split_dataset(df_internet_only)
                # Combine results
                df_train_internet = pd.concat([df_train_internet_common, df_train_internet_only])
                df_test_internet = pd.concat([df_test_internet_common, df_test_internet_only])
            else:
                # Split the entire internet dataset
                df_train_internet, df_test_internet = self.split_dataset(self.df_internet)

        # Process phone dataset if provided
        if self.df_phone is not None:
            if self.df_internet is not None:
                # Handle common indices
                df_phone_common = self.df_phone.loc[common_index]
                df_phone_only = self.df_phone.drop(index=common_index)
                # Split the common subset
                df_train_phone_common = df_phone_common.loc[train_index]
                df_test_phone_common = df_phone_common.loc[test_index]
                # Split phone-only subset
                df_train_phone_only, df_test_phone_only = self.split_dataset(df_phone_only)
                # Combine results
                df_train_phone = pd.concat([df_train_phone_common, df_train_phone_only])
                df_test_phone = pd.concat([df_test_phone_common, df_test_phone_only])
            else:
                # Split the entire phone dataset
                df_train_phone, df_test_phone = self.split_dataset(self.df_phone)

        return df_train_internet, df_test_internet, df_train_phone, df_test_phone
    
    def impute_missing_values(self, df_train, df_test):
        """Impute missing values using KNN for numeric columns only."""
        # Select only numeric columns for imputation
        numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns

        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=10)

        # Impute numeric columns in train and test datasets
        df_train_imputed = pd.DataFrame(
            imputer.fit_transform(df_train[numeric_cols]),
            columns=numeric_cols,
            index=df_train.index
        )
        df_test_imputed = pd.DataFrame(
            imputer.transform(df_test[numeric_cols]),
            columns=numeric_cols,
            index=df_test.index
        )

        # Update original DataFrames with imputed values
        df_train.update(df_train_imputed)
        df_test.update(df_test_imputed)

        return df_train, df_test
    
    def revert_to_booleans(self, df):
        """Converts specific numeric columns back to boolean."""
        bool_features = [col for col in df.columns if col not in self.num_col]
        for col in bool_features:
            if col in df.columns:
                # Convert non-zero and non-null values to True, others to False
                df[col] = df[col].notnull() & (df[col] != 0)
        return df
 

 # Maria's edits END


    # def impute_missing_values(self, df_train, df_test):
    #     """Impute missing values using KNN and cast to appropriate types."""
    #     imputer = KNNImputer(n_neighbors=10)
        
    #     # Impute missing values
    #     df_train_imputed = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns)
    #     df_test_imputed = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns)
        
    #     # Restore the original dtypes
    #     for col in df_train.columns:
    #         original_dtype = df_train[col].dtype
    #         if original_dtype == 'boolean':  # Explicitly cast boolean columns
    #             df_train_imputed[col] = df_train_imputed[col].astype(bool)
    #             df_test_imputed[col] = df_test_imputed[col].astype(bool)
    #         else:  # Restore other types as necessary
    #             df_train_imputed[col] = df_train_imputed[col].astype(original_dtype)
    #             df_test_imputed[col] = df_test_imputed[col].astype(original_dtype)

    #     return df_train_imputed, df_test_imputed

# Maria's Edits
def impute_missing_values(self, df_train, df_test):
    """Impute missing values using KNN for numeric columns only."""
    numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns

    # Apply imputer only to numeric columns
    imputer = KNNImputer(n_neighbors=10)
    df_train_numeric = pd.DataFrame(
        imputer.fit_transform(df_train[numeric_cols]),
        columns=numeric_cols,
        index=df_train.index
    )
    df_test_numeric = pd.DataFrame(
        imputer.transform(df_test[numeric_cols]),
        columns=numeric_cols,
        index=df_test.index
    )

    # Replace numeric columns in original DataFrames
    df_train.update(df_train_numeric)
    df_test.update(df_test_numeric)

    return df_train, df_test
# Maria's edits END

    
    
    def revert_to_booleans(self, df):
        """Convert numerical columns back to boolean"""
        bool_features = [col for col in df.columns if col not in self.num_col]
        for col in bool_features:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df




