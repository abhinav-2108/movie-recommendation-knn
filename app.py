import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.recommendation {
    padding: 15px;
    border-radius: 10px;
    background-color: #f1f3f5;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Movie recommendations using K-Nearest Neighbors (KNN)'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("movie_ratings_merged.csv")

    # Remove missing values
    df = df.dropna()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Convert data types
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df


try:
    df = load_data()

except FileNotFoundError:

    st.error(
        "movie_ratings_merged.csv was not found. "
        "Make sure the CSV file is in the same folder as app.py."
    )

    st.stop()


# =========================================================
# DATASET INFORMATION
# =========================================================

st.sidebar.header("📊 Dataset Information")

st.sidebar.metric(
    "Total Ratings",
    len(df)
)

st.sidebar.metric(
    "Total Users",
    df["userId"].nunique()
)

st.sidebar.metric(
    "Total Movies",
    df["movieId"].nunique()
)

st.sidebar.metric(
    "Average Rating",
    round(df["rating"].mean(), 2)
)


# =========================================================
# CREATE USER-MOVIE RATING MATRIX
# =========================================================

movie_matrix = df.pivot_table(
    index="userId",
    columns="title",
    values="rating"
).fillna(0)


# =========================================================
# MOVIE FEATURE MATRIX
# =========================================================

# Each movie becomes one data point.
# Each user's rating becomes a feature.

movie_features = movie_matrix.T


# =========================================================
# TRAIN KNN MODEL
# =========================================================

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

knn.fit(movie_features)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(movie_name, number_of_recommendations):

    movie_index = movie_features.index.get_loc(movie_name)

    distances, indices = knn.kneighbors(
        movie_features.iloc[movie_index]
        .values
        .reshape(1, -1),
        n_neighbors=number_of_recommendations + 1
    )

    recommendations = []

    for i in range(1, len(indices[0])):

        recommended_movie = movie_features.index[
            indices[0][i]
        ]

        distance = distances[0][i]

        similarity = (1 - distance) * 100

        recommendations.append({
            "Movie": recommended_movie,
            "Similarity": round(similarity, 2)
        })

    return pd.DataFrame(recommendations)


# =========================================================
# MOVIE RECOMMENDATION SECTION
# =========================================================

st.header("🎯 Find Similar Movies")

st.write(
    "Select a movie and the KNN algorithm will find movies "
    "with similar user-rating patterns."
)


# Movie selection

movie_list = sorted(movie_features.index.tolist())

selected_movie = st.selectbox(
    "🎬 Select a Movie",
    movie_list
)


# Number of recommendations

max_recommendations = min(10, len(movie_list) - 1)

number_of_recommendations = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=max_recommendations,
    value=min(5, max_recommendations)
)


# Recommendation button

if st.button(
    "🔍 Recommend Movies",
    type="primary"
):

    recommendations = recommend_movies(
        selected_movie,
        number_of_recommendations
    )

    st.subheader(
        f"Movies Similar to: {selected_movie}"
    )

    # Selected movie information

    selected_movie_data = df[
        df["title"] == selected_movie
    ]

    avg_rating = selected_movie_data["rating"].mean()

    total_ratings = len(selected_movie_data)

    genres = selected_movie_data["genres"].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "⭐ Average Rating",
            round(avg_rating, 2)
        )

    with col2:

        st.metric(
            "👥 Number of Ratings",
            total_ratings
        )

    with col3:

        st.metric(
            "🎭 Genre",
            genres
        )


    # Recommendation results

    st.write("### Recommended Movies")

    for index, row in recommendations.iterrows():

        movie = row["Movie"]

        similarity = row["Similarity"]

        st.markdown(
            f"""
            <div class="recommendation">

            <b>🎬 {movie}</b>

            <br>

            Similarity: <b>{similarity}%</b>

            </div>
            """,
            unsafe_allow_html=True
        )


    # Table

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# EDA SECTION
# =========================================================

st.divider()

st.header("📊 Exploratory Data Analysis")

st.write(
    "The following visualizations help understand the "
    "movie rating dataset before applying KNN."
)


# =========================================================
# ROW 1
# =========================================================

col1, col2 = st.columns(2)


# -------------------------
# Rating Distribution
# -------------------------

with col1:

    st.subheader("⭐ Rating Distribution")

    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        rating_counts.index.astype(str),
        rating_counts.values
    )

    ax.set_xlabel("Rating")

    ax.set_ylabel("Number of Ratings")

    ax.set_title("Distribution of Movie Ratings")

    st.pyplot(fig)

    plt.close(fig)


# -------------------------
# Top Rated Movies
# -------------------------

with col2:

    st.subheader("🏆 Top Rated Movies")

    top_rated = (
        df.groupby("title")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    top_rated.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Average Rating")

    ax.set_ylabel("Movie")

    ax.set_title("Top 10 Movies by Average Rating")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# ROW 2
# =========================================================

col1, col2 = st.columns(2)


# -------------------------
# Most Rated Movies
# -------------------------

with col1:

    st.subheader("🔥 Most Rated Movies")

    most_rated = (
        df["title"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    most_rated.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Number of Ratings")

    ax.set_ylabel("Movie")

    ax.set_title("Top 10 Most Rated Movies")

    st.pyplot(fig)

    plt.close(fig)


# -------------------------
# Rating Box Plot
# -------------------------

with col2:

    st.subheader("📦 Rating Box Plot")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        df["rating"],
        vert=True
    )

    ax.set_ylabel("Rating")

    ax.set_title("Distribution of Ratings")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# GENRE DISTRIBUTION
# =========================================================

st.subheader("🎭 Genre Distribution")

genre_data = (
    df["genres"]
    .str.split("|")
    .explode()
)

genre_counts = (
    genre_data
    .value_counts()
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

genre_counts.sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_xlabel("Number of Ratings")

ax.set_ylabel("Genre")

ax.set_title("Top Genres")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# USER-MOVIE RATING HEATMAP
# =========================================================

st.subheader("🔥 User-Movie Rating Heatmap")

fig, ax = plt.subplots(figsize=(14, 7))

sns.heatmap(
    movie_matrix,
    cmap="viridis",
    linewidths=0.2,
    ax=ax
)

ax.set_xlabel("Movies")

ax.set_ylabel("Users")

ax.set_title("User-Movie Rating Matrix")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# HOW KNN WORKS
# =========================================================

st.divider()

st.header("🤖 How KNN Works in This Recommendation System")

st.markdown("""
### Step 1 — User-Movie Matrix

The rating data is converted into a matrix where:

- Rows = Users
- Columns = Movies
- Values = Ratings

### Step 2 — Movie Features

The matrix is transposed so that every movie becomes a data point.

The ratings given by users act as features.

### Step 3 — KNN Algorithm

The K-Nearest Neighbors algorithm compares movies based on their
rating patterns.

### Step 4 — Cosine Distance

Cosine distance is used to measure how similar two movies are.

Smaller distance means greater similarity.

### Step 5 — Recommendation

For the selected movie, KNN finds the nearest movies and displays
them as recommendations.

Therefore, the recommendations are generated using the actual
KNN algorithm rather than a simple genre-based filter.
""")


# =========================================================
# TECHNICAL INFORMATION
# =========================================================

st.divider()

st.header("⚙️ Technical Details")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
        **Algorithm**

        K-Nearest Neighbors (KNN)
        """
    )

with col2:

    st.info(
        """
        **Distance Metric**

        Cosine Distance
        """
    )

with col3:

    st.info(
        """
        **Feature**

        User Rating Patterns
        """
    )


# =========================================================
# DATASET PREVIEW
# =========================================================

with st.expander("📄 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Movie Recommendation System using K-Nearest Neighbors (KNN) | "
    "Developed using Python, Pandas, Scikit-learn and Streamlit"
)