from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Book1_converted.csv")
df['imdbID'] = df['imdbID'].astype(str).str.strip().str.lower()

# Clean NaNs from important features
df = df.dropna(subset=['imdbRating', 'imdbVotes'])

# Create target column
df['user_like'] = np.where(df['imdbRating'] >= 7, 1, 0)

# Model features
features = ['imdbRating', 'imdbVotes']
X = df[features]
y = df['user_like']

# Train-test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'movie_recommendation_model.pkl')

# Load trained model
model = joblib.load('movie_recommendation_model.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        title = request.form.get("title")
        genre = request.form.get("genre")
        director = request.form.get("director")
        actor = request.form.get("actor")

        filtered = df.copy()

        if title:
            filtered = filtered[filtered['title'].str.contains(title, case=False, na=False)]
        if genre:
            filtered = filtered[filtered['genre'].str.contains(genre, case=False, na=False)]
        if director:
            filtered = filtered[filtered['director'].str.contains(director, case=False, na=False)]
        if actor:
            filtered = filtered[filtered['cast'].str.contains(actor, case=False, na=False)]

        recommendations = filtered.to_dict(orient='records')
    else:
        recommendations = df.head(20).to_dict(orient='records')

    return render_template("index.html", recommendations=recommendations)

@app.route("/movie/<imdb_id>")
def movie_detail(imdb_id):
    imdb_id = str(imdb_id).strip().lower()
    movie = df[df['imdbID'] == imdb_id]
    if movie.empty:
        return "Movie not found."
    return render_template("movie_detail.html", movie=movie.iloc[0])

@app.route("/predict/<imdb_id>")
def predict_movie(imdb_id):
    imdb_id = str(imdb_id).strip().lower()
    movie = df[df['imdbID'] == imdb_id]
    if movie.empty:
        return "Movie not found."

    features = movie[['imdbRating', 'imdbVotes']]
    prediction = model.predict(features)[0]
    result = "You might like this!" if prediction == 1 else "Probably not your favorite."

    return render_template("movie_detail.html", movie=movie.iloc[0], result=result)

if __name__ == "__main__":
    app.run(debug=True)
