# Updated_Recommendation_Engine

For Personalized And New Users.

# ncf_and_content_based
1.) Personalized For Pre-Existing Users With Post Behaviour And History.
2.) PPT- "NCF Model Explanation" which tells the flow of working of the  engine using neural networks.
3.) Run- model_test.py.
4.) Note: Before preprocessing, I extracted titles from the Title field and descriptions from the Category from the original dataset Field because I only need this. for eg id, category, slug, title, identifier, comment_count, upvote_count, view_count, exit_count, rating_count, average_rating, share_count, video_link, contract_address, chain_id, chart_url, base_token, is_locked, created_at, first_name, last_name, username, upvoted, bookmarked,thumbnail_url, following,picture_url these all are header fields I only need title field values and description which is under the sub-category of the category header field.
For this, I did feature engineering.


# new_users_recommendation
1.) For New Users WitHout Any History.
2.) PPT- "Recommendation Engine Overview" which tells the flow of working on the engine.
3.) Run- updated_recommendation_engine.py.
4.) Note: Same for this also extracting only useful fields, doing feature engineering.
5.) New user recommendations are based on popular and the newest items are also added to avoid neglecting their importance.


