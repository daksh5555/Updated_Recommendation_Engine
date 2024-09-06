import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the extracted CSV file
input_file = 'extracted_popularity_data_original.csv'  # Replace with the path to your extracted CSV file
data = pd.read_csv(input_file)

# Define the fields to process
fields_to_process = ['view_count', 'rating_count', 'upvote_count', 'share_count', 'comment_count']

# Replace zeros with np.nan for imputation
data.replace(0, np.nan, inplace=True)

# Define the imputer (using mean imputation)
imputer = SimpleImputer(strategy='mean')

# Apply imputation to the relevant fields
data[fields_to_process] = imputer.fit_transform(data[fields_to_process])

# Round up any float values to the next whole number
data[fields_to_process] = np.ceil(data[fields_to_process])

# Convert the values to integers
data[fields_to_process] = data[fields_to_process].astype(int)

# Preprocess the 'created_at' column (Unix timestamps) using Min-Max scaling
scaler = MinMaxScaler()
data['created_at_scaled'] = scaler.fit_transform(data[['created_at']])

# Add 'created_at_scaled' to the matrix
id_matrix = data.set_index('id')

# Save the preprocessed matrix to a new CSV file
output_file = 'preprocessed_id_matrix_imputed_rounded_scaled.csv'  # Replace with the desired output file name
id_matrix.to_csv(output_file)

print(f"Preprocessing complete. Matrix saved to {output_file}")
