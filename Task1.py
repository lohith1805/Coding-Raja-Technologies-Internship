import pandas as pd
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from flask import Flask, request, jsonify

# Step 1: Data Collection
# Define the file paths
ratings_file_path = "C:\\Users\\lohit\\OneDrive\\Desktop\\ml-latest-small\\ratings.csv"
movies_file_path = "C:\\Users\\lohit\\OneDrive\\Desktop\\ml-latest-small\\movies.csv"

# Column names for the ratings file
rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(ratings_file_path, names=rating_columns, skiprows=1)

# Ensure movieId is of type int by first checking for NaN values and handling them
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
ratings.dropna(subset=['movieId'], inplace=True)
ratings['movieId'] = ratings['movieId'].astype(int)

# Column names for the movies file
movie_columns = ['movieId', 'title', 'genres']
movies = pd.read_csv(movies_file_path, names=movie_columns, skiprows=1)

# Ensure movieId is of type int by first checking for NaN values and handling them
movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
movies.dropna(subset=['movieId'], inplace=True)
movies['movieId'] = movies['movieId'].astype(int)

# Merge the ratings and movies dataframes
data = pd.merge(ratings, movies, on='movieId')

# Step 2: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values if necessary
# (No missing values in MovieLens dataset after above preprocessing)

# Step 3: User-Item Matrix
# Create a pivot table
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Step 4: Collaborative Filtering
# Load the dataset into surprise
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split into training and test sets
trainset, testset = train_test_split(data_surprise, test_size=0.25)

# Use User-based Collaborative Filtering
algo = KNNBasic(sim_options={'user_based': True})
algo.fit(trainset)

# Step 5: Model Evaluation
# Predict ratings for the testset
predictions = algo.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Step 6: Top-N Recommendations
# Function to get top-N recommendations
def get_top_n_recommendations(user_id, n=10):
    # Get a list of all movie IDs
    all_movie_ids = ratings['movieId'].unique()
    
    # Predict ratings for all movies the user hasn't seen
    unseen_movies = [movie for movie in all_movie_ids if movie not in ratings[ratings['userId'] == user_id]['movieId'].values]
    predictions = [algo.predict(user_id, movie_id) for movie_id in unseen_movies]
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top-N recommendations
    top_n = predictions[:n]
    top_n_movies = [pred.iid for pred in top_n]
    return movies[movies['movieId'].isin(top_n_movies)][['movieId', 'title']]

# Example usage
user_id_example = 1
top_n_recommendations = get_top_n_recommendations(user_id=user_id_example, n=10)
print(top_n_recommendations)

# Step 7: Interactive Interface
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    top_n = get_top_n_recommendations(user_id)
    return jsonify(top_n.to_dict(orient='records'))

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0')
    except Exception as e:
        print(f"Error starting the server: {e}")
