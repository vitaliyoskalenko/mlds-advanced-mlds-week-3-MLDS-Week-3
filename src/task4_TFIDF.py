"""Task 4: TF-IDF Calculation."""
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(texts: List[str]) -> pd.DataFrame:
    """
    Convert a list of text strings into a TF-IDF representation.

    Args:
        texts (List[str]): A list of text strings.

    Returns:
        pd.DataFrame: A DataFrame representing the TF-IDF model,
                      with words as columns and their TF-IDF scores as values.

    Raises:
        TypeError: If texts is not a list.
        ValueError: If texts is empty or contains non-string elements.

    Example:
        >>> texts = ["I love NLP", "NLP is fun"]
        >>> calculate_tfidf(texts)
                fun         i        is      love       nlp
        0  0.000000  0.579739  0.000000  0.579739  0.447214
        1  0.652491  0.000000  0.652491  0.000000  0.385372
    """
    if not isinstance(texts, list):
        raise TypeError("Input texts must be a list")
    if not texts:
        raise ValueError("Input texts cannot be empty")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All texts must be strings")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df


if __name__ == "__main__":
    texts = [
        "Converts a list of text strings into a Bag of Words representation",
        "The words that changed the strings",
        "Your duty words of the heart",
        "The Bag full of strings"
    ]
    print(calculate_tfidf(texts))
    
