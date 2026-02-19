from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
