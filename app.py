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
# TITLE
# =========================================================

st.title("🎬 Movie Recommendation System")

st.markdown(
    "### Movie recommendations using K-Nearest Neighbors (KNN)"
)

st.write(
    "Select a movie and the KNN algorithm will find movies "
    "with similar user-rating patterns."
)

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("movie recomandation.csv")

    df = df.dropna()
    df = df.drop_duplicates()

    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "movie_ratings_merged.csv was not found. "
        "Make sure the CSV file is in the same GitHub folder as app.py."
    )

    st.stop()


# =========================================================
# SIDEBAR
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
# USER-MOVIE MATRIX
# =========================================================

movie_matrix = df.pivot_table(
    index="userId",
    columns="title",
    values="rating"
).fillna(0)


# =========================================================
# MOVIE FEATURE MATRIX
# =========================================================

movie_features = movie_matrix.T


# =========================================================
# KNN MODEL
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

        movie = movie_features.index[indices[0][i]]

        distance = distances[0][i]

        similarity = max(0, (1 - distance) * 100)

        recommendations.append({
            "Movie": movie,
            "Similarity": round(similarity, 2)
        })

    return pd.DataFrame(recommendations)


# =========================================================
# FIND SIMILAR MOVIES
# =========================================================

st.header("🎯 Find Similar Movies")

movie_list = sorted(movie_features.index.tolist())

selected_movie = st.selectbox(
    "🎬 Select a Movie",
    movie_list
)

max_recommendations = min(
    10,
    len(movie_list) - 1
)

number_of_recommendations = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=max_recommendations,
    value=min(5, max_recommendations)
)


# =========================================================
# RECOMMEND BUTTON
# =========================================================

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

    # -----------------------------------------------------
    # SELECTED MOVIE INFORMATION
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RECOMMENDED MOVIES
    # -----------------------------------------------------

    st.subheader("🎬 Recommended Movies")

    for _, row in recommendations.iterrows():

        movie = row["Movie"]

        similarity = row["Similarity"]

        st.info(
            f"🎬 **{movie}**\n\n"
            f"Similarity: **{similarity}%**"
        )

    # -----------------------------------------------------
    # RECOMMENDATION TABLE
    # -----------------------------------------------------

    st.subheader("📋 Recommendation Details")

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# EDA SECTION
# SAME AS COLAB
# =========================================================

st.divider()

st.header("📊 Exploratory Data Analysis")

st.write(
    "The following visualizations are the same EDA plots "
    "performed in the Google Colab notebook."
)


# =========================================================
# 1. TOP 10 MOST RATED MOVIES
# SAME AS COLAB CELL 14
# =========================================================

st.subheader("1️⃣ Top 10 Most Rated Movies")

top_movies = df["title"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))

top_movies.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Top 10 Most Rated Movies")
ax.set_xlabel("Movie")
ax.set_ylabel("Number of Ratings")
ax.tick_params(axis="x", rotation=75)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 2. MOVIE RATING DISTRIBUTION - PIE CHART
# SAME AS COLAB CELL 15
# =========================================================

st.subheader("2️⃣ Movie Rating Distribution")

rating_counts = df["rating"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 7))

ax.pie(
    rating_counts,
    labels=rating_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Movie Rating Distribution")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 3. USER ID VS MOVIE RATING
# SAME AS COLAB CELL 16
# =========================================================

st.subheader("3️⃣ User ID vs Movie Rating")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["userId"],
    df["rating"],
    alpha=0.5
)

ax.set_title("User ID vs Movie Rating")
ax.set_xlabel("User ID")
ax.set_ylabel("Rating")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 4. BOX PLOT
# SAME AS COLAB CELL 17
# =========================================================

st.subheader("4️⃣ Box Plot of Movie Ratings")

fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    x=df["rating"],
    ax=ax
)

ax.set_title("Box Plot of Movie Ratings")
ax.set_xlabel("Rating")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 5. HISTOGRAM
# SAME AS COLAB CELL 18
# =========================================================

st.subheader("5️⃣ Distribution of Movie Ratings")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    df["rating"],
    bins=9
)

ax.set_title("Distribution of Movie Ratings")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 6. CORRELATION HEATMAP
# SAME AS COLAB CELL 19
# =========================================================

st.subheader("6️⃣ Correlation Heatmap")

corr = df[
    ["userId", "movieId", "rating", "timestamp"]
].corr()

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

ax.set_title("Correlation Heatmap")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# HOW KNN WORKS
# =========================================================

st.divider()

st.header("🤖 How KNN Works")

st.markdown("""
### 1. User-Movie Matrix

The rating data is converted into a matrix where:

- Rows represent users
- Columns represent movies
- Values represent ratings

### 2. Movie Feature Matrix

The user-movie matrix is transposed so that each movie
becomes a data point.

The ratings given by users act as features.

### 3. KNN Algorithm

K-Nearest Neighbors compares movies based on their
user-rating patterns.

### 4. Cosine Distance

Cosine distance measures the difference between movie
rating patterns.

A smaller distance means the movies are more similar.

### 5. Recommendation

KNN finds the nearest movies to the selected movie
and displays them as recommendations.
""")


# =========================================================
# TECHNICAL DETAILS
# =========================================================

st.divider()

st.header("⚙️ Technical Details")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "Algorithm\n\n"
        "K-Nearest Neighbors (KNN)"
    )

with col2:

    st.info(
        "Distance Metric\n\n"
        "Cosine Distance"
    )

with col3:

    st.info(
        "Features\n\n"
        "User Rating Patterns"
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
    "Python | Pandas | Scikit-learn | Streamlit"
)