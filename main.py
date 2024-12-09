from preprocess import DataLoader, DataPreprocessor, DataSplitter

# File path
filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Load data
data_loader = DataLoader(filepath)
df = data_loader.load_data()
df = data_loader.check_data_quality()

# Preprocess data
preprocessor = DataPreprocessor(df)
df_internet, df_phone = preprocessor.split_by_service()

# Preprocess categorical columns
df_internet = preprocessor.preprocess_categorical(df_internet)
df_phone = preprocessor.preprocess_categorical(df_phone)

# Format column names
df_internet.columns = df_internet.columns.str.lower().str.replace(' ', '_')
df_phone.columns = df_phone.columns.str.lower().str.replace(' ', '_')

# Split and process data
splitter = DataSplitter(df_internet, df_phone)
df_train_internet, df_test_internet, df_train_phone, df_test_phone = splitter.train_test_split()

# Impute and revert to booleans
df_train_internet, df_test_internet = splitter.impute_missing_values(df_train_internet, df_test_internet)
df_train_phone, df_test_phone = splitter.impute_missing_values(df_train_phone, df_test_phone)

df_train_internet = splitter.revert_to_booleans(df_train_internet)
df_test_internet = splitter.revert_to_booleans(df_test_internet)
df_train_phone = splitter.revert_to_booleans(df_train_phone)
df_test_phone = splitter.revert_to_booleans(df_test_phone)

# Print results
print(df_train_internet.head())