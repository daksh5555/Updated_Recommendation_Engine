import pandas as pd
from datetime import datetime

# Load the original CSV file
input_file = 'socialverse_data.csv'  # Replace with the path to your input CSV file
data = pd.read_csv(input_file)

# Convert millisecond timestamp to human-readable format
def ms_to_readable(timestamp):
    try:
        # Convert milliseconds to seconds by dividing by 1000
        return datetime.fromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OSError, TypeError):
        # If conversion fails, return the original value
        return str(timestamp)

# Extract the relevant columns
extracted_data = data[['id', 'view_count', 'rating_count', 'upvote_count', 'share_count', 'comment_count', 'created_at']]

# Create a new column with human-readable dates
extracted_data['readable_created_at'] = extracted_data['created_at'].apply(ms_to_readable)

# Save the extracted data with human-readable dates
output_file_readable = 'extracted_popularity_data_readable.csv'
extracted_data[['id', 'view_count', 'rating_count', 'upvote_count', 'share_count', 'comment_count', 'readable_created_at']].to_csv(output_file_readable, index=False)

# Save the extracted data with original timestamps
output_file_original = 'extracted_popularity_data_original.csv'
extracted_data[['id', 'view_count', 'rating_count', 'upvote_count', 'share_count', 'comment_count', 'created_at']].to_csv(output_file_original, index=False)

# Save the timestamps separately with both original and readable formats
output_file_timestamps = 'extracted_timestamps.csv'
extracted_data[['created_at', 'readable_created_at']].to_csv(output_file_timestamps, index=False)

print(f"Data successfully extracted and saved to:")
print(f"1. {output_file_readable} (with human-readable dates)")
print(f"2. {output_file_original} (with original timestamps)")
print(f"3. {output_file_timestamps} (timestamps only, both formats)")