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
        
        return df_internet, df_phone
    
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

class DataSplitter:
    def __init__(self, df_internet, df_phone):
        self.df_internet = df_internet
        self.df_phone = df_phone
        self.num_col = ['tenure', 'totalcharges', 'monthlycharges']
        self.target = 'churn'
    
    def train_test_split(self):
        """Split data into train and test sets"""
        df_train_internet, df_test_internet = train_test_split(
            self.df_internet, 
            test_size=0.2, 
            stratify=self.df_internet[self.target]
        )
        
        df_train_phone, df_test_phone = train_test_split(
            self.df_phone, 
            test_size=0.2, 
            stratify=self.df_phone[self.target]
        )
        
        return df_train_internet, df_test_internet, df_train_phone, df_test_phone
    
    def impute_missing_values(self, df_train, df_test):
        """Impute missing values using KNN"""
        imputer = KNNImputer(n_neighbors=10)
        df_train[:] = imputer.fit_transform(df_train)
        df_test[:] = imputer.transform(df_test)
        
        return df_train, df_test
    
    def revert_to_booleans(self, df):
        """Convert numerical columns back to boolean"""
        bool_features = [col for col in df.columns if col not in self.num_col]
        for col in bool_features:
            if col in df.columns:
                df[col] = df[col].astype('boolean')
        return df


