from flask import Flask, render_template, request
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
df = pd.read_csv("Book1_converted.csv")
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        title = request.form.get("title", "").strip().lower()
        genre = request.form.get("genre", "").strip().lower()
        director = request.form.get("director", "").strip().lower()
        actor = request.form.get("actor", "").strip().lower()

        filtered_df = df.copy()

        if title:
            filtered_df = filtered_df[filtered_df["title"].fillna("").str.lower().str.contains(title)]
        if genre:
            filtered_df = filtered_df[filtered_df["genre"].fillna("").str.lower().str.contains(genre)]
        if director:
            filtered_df = filtered_df[filtered_df["director"].fillna("").str.lower().str.contains(director)]
        if actor:
            filtered_df = filtered_df[filtered_df["cast"].fillna("").str.lower().str.contains(actor)]

        recommendations = filtered_df.to_dict(orient="records")

    return render_template("index.html", recommendations=recommendations)

@app.route("/movie/<imdb_id>")
def movie_detail(imdb_id):
    imdb_id = str(imdb_id).strip()
    logging.info(f"Received IMDB ID: {imdb_id}")
    
    movie_row = df[df["imdbID"].astype(str).str.strip() == imdb_id]

    if not movie_row.empty:
        movie = movie_row.iloc[0].to_dict()
        generate_movie_eda(df, movie)
        return render_template("movie_detail.html", movie=movie)
    else:
        return render_template("movie_detail.html", movie=None, error="Movie not found.")

def generate_movie_eda(df, movie):
    os.makedirs("static/plots", exist_ok=True)

    genre = movie.get("genre", "").split(",")[0].strip()
    director = movie.get("director", "").strip()
    
    try:
        rating = float(movie.get("rating", 0))
    except:
        rating = 0
        
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    genre = movie.get("genre", "").split(",")[0].strip()
    genre_movies = df[df["genre"].str.contains(genre, case=False, na=False)]
    
    # Handle edge cases
    avg_genre_rating = genre_movies["rating"].mean()
    if pd.isna(avg_genre_rating):
        avg_genre_rating = 0
    
    try:
        rating = float(movie.get("rating", 0))
    except:
        rating = 0
    
    print(f"[DEBUG] Genre: {genre}")
    print(f"[DEBUG] Selected Movie Rating: {rating}")
    print(f"[DEBUG] Avg Genre Rating: {avg_genre_rating}")
    print(f"[DEBUG] Number of genre-matched movies: {len(genre_movies)}")


    plt.figure(figsize=(5, 4))
    sns.barplot(x=["Selected Movie", "Genre Avg"], y=[rating, avg_genre_rating], palette=["gold", "blue"])
    plt.title("Rating vs Genre Average")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.savefig("static/plots/rating_vs_genre.png")
    plt.close()

    # 2. Genre Popularity
    genre_counts = df["genre"].str.split(",").explode().str.strip().value_counts()
    top_genres = genre_counts.head(10)

    plt.figure(figsize=(7, 4))
    bars = sns.barplot(x=top_genres.index, y=top_genres.values, palette="Blues")
    for bar, label in zip(bars.patches, top_genres.index):
        if genre.lower() in label.lower():
            bar.set_color("gold")
    plt.xticks(rotation=45)
    plt.title("Genre Popularity (Highlighting Selected Movie's Genre)")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.savefig("static/plots/genre_popularity_highlight.png")
    plt.close()

    # 3. Director Movie Ratings
    if director:
        director_movies = df[df["director"].str.contains(director, case=False, na=False)]
        director_movies["rating"] = pd.to_numeric(director_movies["rating"], errors="coerce")

        plt.figure(figsize=(6, 4))
        sns.histplot(director_movies["rating"].dropna(), bins=8, kde=True, color="skyblue", label=director)
        plt.axvline(rating, color='gold', linestyle='--', label="Selected Movie")
        plt.title(f"{director}'s Movies Rating Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/plots/director_rating_dist.png")
        plt.close()


if __name__ == "__main__":
    app.run(debug=True)
