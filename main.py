from preprocess import DataLoader, DataPreprocessor, DataSplitter

# File path
filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Load data
data_loader = DataLoader(filepath)
df = data_loader.load_data()
df = data_loader.check_data_quality()

# Preprocess data
preprocessor = DataPreprocessor(df)
df_internet, df_phone, df_internet_only = preprocessor.split_by_service()

# Preprocess categorical columns
df_internet = preprocessor.preprocess_categorical(df_internet)
df_phone = preprocessor.preprocess_categorical(df_phone)
df_internet_only = preprocessor.preprocess_categorical(df_internet_only)


# Format column names
df_internet.columns = df_internet.columns.str.lower().str.replace(' ', '_')
df_phone.columns = df_phone.columns.str.lower().str.replace(' ', '_')
df_internet_only.columns = df_internet_only.columns.str.lower().str.replace(' ', '_')


# Split and process data
splitter = DataSplitter(df_internet, df_phone, 0.2)
df_train_internet, df_test_internet, df_train_phone, df_test_phone = splitter.train_test_split()

# Impute and revert to booleans
df_train_internet, df_test_internet = splitter.impute_missing_values(df_train_internet, df_test_internet)
df_train_phone, df_test_phone = splitter.impute_missing_values(df_train_phone, df_test_phone)

df_train_internet = splitter.revert_to_booleans(df_train_internet)
df_test_internet = splitter.revert_to_booleans(df_test_internet)
df_train_phone = splitter.revert_to_booleans(df_train_phone)
df_test_phone = splitter.revert_to_booleans(df_test_phone)


df_train_internet.to_csv('preprocessed_data/df_train_internet.csv', index = False)
df_test_internet.to_csv('preprocessed_data/df_test_internet.csv', index = False)
df_train_phone.to_csv('preprocessed_data/df_train_phone.csv', index = False)
df_test_phone.to_csv('preprocessed_data/df_test_phone.csv', index = False)

# Train-test split for internet-only customers
splitter = DataSplitter(df_internet_only, None, test_size=0.2)
df_train_internet_only, df_test_internet_only, _, _ = splitter.train_test_split()

# Impute and revert to booleans for internet-only dataset
df_train_internet_only, df_test_internet_only = splitter.impute_missing_values(df_train_internet_only, df_test_internet_only)

df_train_internet_only = splitter.revert_to_booleans(df_train_internet_only)
df_test_internet_only = splitter.revert_to_booleans(df_test_internet_only)

# Save internet-only datasets
df_train_internet_only.to_csv('preprocessed_data/df_train_internet_only.csv', index=False)
df_test_internet_only.to_csv('preprocessed_data/df_test_internet_only.csv', index=False)


print("Internet-only training dataset saved as 'preprocessed_data/df_train_internet_only.csv'")