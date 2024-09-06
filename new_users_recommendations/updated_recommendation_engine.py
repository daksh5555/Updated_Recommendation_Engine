import pandas as pd
import time
import logging
import configparser
import os

# Setup logging
logging.basicConfig(
    filename='recommendation_engine.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load configuration from a file
config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters
INPUT_FILE = config['FILES']['InputFile']
USERNAMES_FILE = config['FILES']['UsernamesFile']
OUTPUT_FILE = config['FILES']['OutputFile']
RECOMMENDATIONS_OUTPUT_FILE = config['FILES']['RecommendationsOutputFile']
TOP_N = int(config['RECOMMENDATIONS']['TopN'])
SLEEP_TIME = int(config['SETTINGS']['SleepTime'])

# Weights for each field (can be configured in config.ini)
weights = {
    'view_count': float(config['WEIGHTS']['ViewCount']),
    'rating_count': float(config['WEIGHTS']['RatingCount']),
    'upvote_count': float(config['WEIGHTS']['UpvoteCount']),
    'share_count': float(config['WEIGHTS']['ShareCount']),
    'comment_count': float(config['WEIGHTS']['CommentCount'])
}

def load_data():
    try:
        data = pd.read_csv(INPUT_FILE, index_col='id')
        return data
    except Exception as e:
        logging.error(f"Error loading data from {INPUT_FILE}: {e}")
        raise

def load_usernames():
    try:
        usernames_data = pd.read_csv(USERNAMES_FILE)
        return usernames_data
    except Exception as e:
        logging.error(f"Error loading usernames from {USERNAMES_FILE}: {e}")
        raise

def generate_recommendations(new_users=None):
    try:
        data = load_data()
        usernames_data = load_usernames()

        logging.info(f"Data columns: {data.columns}")
        logging.info(f"Usernames columns: {usernames_data.columns}")

        # Normalize weights if they don't sum to 1.0
        total_weight = sum(weights.values())
        logging.info(f"Total weight: {total_weight}")
        if total_weight != 1.0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            logging.info(f"Normalized weights: {normalized_weights}")
        else:
            normalized_weights = weights

        # Calculate the popularity score
        data['popularity_score'] = (
            normalized_weights['view_count'] * data['view_count'] +
            normalized_weights['rating_count'] * data['rating_count'] +
            normalized_weights['upvote_count'] * data['upvote_count'] +
            normalized_weights['share_count'] * data['share_count'] +
            normalized_weights['comment_count'] * data['comment_count']
        )

        # Debug information
        logging.info(f"Data head:\n{data.head()}")

        # Sort IDs based on popularity score
        top_popular = data[['popularity_score']].sort_values(by='popularity_score', ascending=False).reset_index()

        # Check for required columns
        if 'created_at_scaled' not in data.columns:
            logging.error("'created_at_scaled' column is missing from data")
            raise KeyError("'created_at_scaled' column is missing from data")

        # Sort IDs based on 'created_at_scaled'
        top_recent = data[['created_at_scaled']].sort_values(by='created_at_scaled', ascending=False).reset_index()

        # Debug information
        logging.info(f"Top popular items:\n{top_popular.head()}")
        logging.info(f"Top recent items:\n{top_recent.head()}")

        # Ensure at least 10 popular items and 1 recent item are available
        if len(top_popular) < 10 or len(top_recent) == 0:
            logging.error("Insufficient data to generate recommendations.")
            raise ValueError("Insufficient data to generate recommendations.")

        # List to store final recommendations
        recommendations_list = []

        # Generate recommendations in batches of 4 popular items and 1 new post
        num_recommendations = 50
        while num_recommendations > 0:
            # Take up to 4 popular items
            popular_items = top_popular.head(4)
            top_popular = top_popular.iloc[4:]  # Use iloc for slicing

            # Take 1 new post
            recent_item = top_recent.head(1)
            top_recent = top_recent.iloc[1:]  # Use iloc for slicing

            # Add popular items and new post to recommendations
            for _, video_row in popular_items.iterrows():
                if num_recommendations > 0:
                    recommendations_list.append({'recommended_id': video_row['id']})
                    num_recommendations -= 1

            if num_recommendations > 0 and not recent_item.empty:
                recommendations_list.append({'recommended_id': recent_item.iloc[0]['id']})  # Use iloc for consistency
                num_recommendations -= 1

            # Stop if no more items are available
            if top_popular.empty and top_recent.empty:
                break

        # Create a DataFrame and save the recommendations
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Recommendations complete. Recommendations saved to {OUTPUT_FILE}")

        if new_users:
            # Identify users with no ID history and generate recommendations
            users_with_no_history = usernames_data[usernames_data['username'].isin(new_users)]
            recommendations_for_users = []
            for _, user_row in users_with_no_history.iterrows():
                user_id = user_row['username']
                for _, rec_row in recommendations_df.iterrows():
                    recommendations_for_users.append({'username': user_id, 'recommended_id': rec_row['recommended_id']})

            # Save the user recommendations
            recommendations_for_users_df = pd.DataFrame(recommendations_for_users)
            recommendations_for_users_df.to_csv(RECOMMENDATIONS_OUTPUT_FILE, index=False)
            logging.info(f"User recommendations complete. Recommendations saved to {RECOMMENDATIONS_OUTPUT_FILE}")

            # Print recommendations for each new user
            for user_id in new_users:
                user_recs = recommendations_for_users_df[recommendations_for_users_df['username'] == user_id]
                print(f"Recommendations for {user_id}:")
                print(user_recs[['recommended_id']])
                print()

    except KeyError as e:
        logging.error(f"KeyError in generate_recommendations: {e}", exc_info=True)
        raise
    except ValueError as e:
        logging.error(f"ValueError in generate_recommendations: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}", exc_info=True)
        raise

def run_recommendation_engine():
    print("Starting the recommendation engine...")  # Indicate that the engine has started
    processed_users = set()  # Set to keep track of processed usernames

    while True:
        try:
            print("Loading usernames data...")
            usernames_data = load_usernames()
            current_users = set(usernames_data['username'])

            # Check for new users
            new_users = current_users - processed_users

            if new_users:
                print(f"New users detected: {new_users}. Restarting the recommendation engine.")
                logging.info(f"New users detected: {new_users}. Restarting the recommendation engine.")
                generate_recommendations(new_users=new_users)  # Pass new users to the function
                processed_users = current_users  # Update processed users after generating recommendations
                print("Recommendation engine completed processing new users.")
            else:
                print("No new users detected. No need to restart the engine.")
                logging.info("No new users detected. No need to restart the engine.")

        except Exception as e:
            print(f"Error in recommendation engine loop: {e}")
            logging.error(f"Error in recommendation engine loop: {e}", exc_info=True)
        finally:
            print(f"Sleeping for {SLEEP_TIME} seconds before next check...")
            time.sleep(SLEEP_TIME)  # Wait before checking again

# Start the scheduler
if __name__ == "__main__":
    run_recommendation_engine()
