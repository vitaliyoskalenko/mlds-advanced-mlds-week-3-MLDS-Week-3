"""Task 7: Integrated NLP Pipeline."""
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.task1_text_cleaning import clean_and_tokenize
from src.task2_stemm_lemma import stem_and_lemmatize
from src.task5_word2vec import train_word2vec
from src.task6_Ngrams import generate_ngrams, count_ngrams


def analyze_documents(
    documents: List[str],
    use_lemmatization: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    vector_method: str = "tfidf"
) -> Dict[str, Any]:
    """
    Perform comprehensive NLP analysis on a collection of documents.

    This pipeline integrates text preprocessing, vectorization, word embeddings,
    and n-gram analysis to provide a complete document analysis.

    Args:
        documents (List[str]): List of text documents to analyze.
        use_lemmatization (bool): If True, use lemmatization; else use stemming.
                                  Default is True.
        ngram_range (Tuple[int, int]): Range of n-gram sizes for vectorization.
                                        Default is (1, 2).
        vector_method (str): Vectorization method - "bow" or "tfidf".
                            Default is "tfidf".

    Returns:
        Dict[str, Any]: Comprehensive analysis results including preprocessed
                       documents, vectorization, embeddings, and n-gram analysis.

    Raises:
        TypeError: If documents is not a list or contains non-strings.
        ValueError: If documents is empty, has fewer than 2 documents,
                   vector_method is invalid, or ngram_range is invalid.

    Example:
        >>> docs = ["I love NLP", "NLP is fun", "Machine learning rocks"]
        >>> results = analyze_documents(docs)
        >>> results['preprocessed']['total_tokens']
        9
    """
    # Input validation
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")
    if not documents:
        raise ValueError("Documents cannot be empty")
    if not all(isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")
    if vector_method not in ("bow", "tfidf"):
        raise ValueError('vector_method must be "bow" or "tfidf"')
    if (not isinstance(ngram_range, tuple) or len(ngram_range) != 2
            or not all(isinstance(value, int) for value in ngram_range)):
        raise ValueError("Invalid ngram_range: expected a tuple of two integers")
    if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError("Invalid ngram_range: min must be >= 1 and <= max")

    # Step 1: Preprocess documents
    preprocessed_docs = []
    all_tokens = []

    for document in documents:
        if document.strip():
            tokens = clean_and_tokenize(document)
            processed = stem_and_lemmatize(tokens)
            tokens = processed["lemmatized"] if use_lemmatization else processed["stemmed"]
        else:
            tokens = []
        preprocessed_docs.append(tokens)
        all_tokens.extend(tokens)

    # Calculate preprocessing statistics
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    avg_doc_length = total_tokens / len(documents)

    # Step 2: Vectorization
    # Reconstruct text from processed tokens for vectorization
    processed_texts = [" ".join(tokens) for tokens in preprocessed_docs]

    if vector_method == "bow":
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    doc_labels = [f"doc_{i}" for i in range(len(documents))]
    feature_df = pd.DataFrame(
        matrix.toarray(),
        columns=feature_names,
        index=doc_labels
    )

    # Step 3: Word embeddings
    # Filter out empty documents for Word2Vec
    tokenized_sentences = [tokens for tokens in preprocessed_docs if tokens]

    if tokenized_sentences:
        word2vec_model = train_word2vec(tokenized_sentences)
        vocabulary_size = len(word2vec_model.wv)
        vector_size = int(word2vec_model.wv.vector_size)
    else:
        word2vec_model = None
        vocabulary_size = 0
        vector_size = 0

    # Step 4: N-gram analysis
    # Combine all documents for n-gram analysis
    all_bigrams = []
    all_trigrams = []

    for tokens in preprocessed_docs:
        text = " ".join(tokens)
        if len(tokens) >= 2:
            all_bigrams.extend(generate_ngrams(text, 2))
        if len(tokens) >= 3:
            all_trigrams.extend(generate_ngrams(text, 3))

    bigram_counts = count_ngrams(all_bigrams) if all_bigrams else {}
    trigram_counts = count_ngrams(all_trigrams) if all_trigrams else {}

    # Sort by frequency
    top_bigrams = sorted(bigram_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    top_trigrams = sorted(trigram_counts.items(), key=lambda item: item[1], reverse=True)[:10]

    # Calculate bigram diversity
    bigram_diversity = len(bigram_counts) / len(all_bigrams) if all_bigrams else 0.0

    # Step 5: Document similarity
    similarity_matrix = cosine_similarity(feature_df.values)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=doc_labels,
        columns=doc_labels
    )

    # Compile results
    results = {
        "preprocessed": {
            "documents": preprocessed_docs,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_doc_length": round(avg_doc_length, 2)
        },
        "vectorization": {
            "method": vector_method,
            "feature_matrix": feature_df,
            "feature_names": list(feature_names),
            "shape": feature_df.shape
        },
        "word_embeddings": {
            "model": word2vec_model,
            "vocabulary_size": vocabulary_size,
            "vector_size": vector_size
        },
        "ngram_analysis": {
            "top_bigrams": top_bigrams,
            "top_trigrams": top_trigrams,
            "bigram_diversity": round(bigram_diversity, 3)
        },
        "document_similarity": similarity_df
    }
    
    return results


def find_similar_documents(
    doc_index: int,
    similarity_matrix: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[int, float]]:
    """
    Find the most similar documents to a given document.

    Args:
        doc_index (int): Index of the document to find similarities for.
        similarity_matrix (pd.DataFrame): Document similarity matrix.
        top_n (int): Number of similar documents to return. Default is 3.

    Returns:
        List[Tuple[int, float]]: List of (document_index, similarity_score) tuples,
                                 sorted by similarity in descending order.

    Raises:
        ValueError: If doc_index is out of range or top_n is invalid.
        TypeError: If similarity_matrix is not a DataFrame.

    Example:
        >>> sim_matrix = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
        >>> find_similar_documents(0, sim_matrix, top_n=1)
        [(1, 0.8)]
    """
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame")
    if not isinstance(doc_index, int) or isinstance(doc_index, bool):
        raise TypeError("doc_index must be an integer")
    if doc_index < 0 or doc_index >= len(similarity_matrix):
        raise ValueError(f"doc_index must be between 0 and {len(similarity_matrix) - 1}")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    # Get similarity scores for the document (excluding itself)
    similarities = similarity_matrix.iloc[doc_index]

    # Remove self-similarity
    similarities = similarities.drop(similarity_matrix.index[doc_index])

    # Sort by similarity score (descending)
    sorted_similarities = sorted(
        similarities.items(), key=lambda item: item[1], reverse=True
    )

    # Extract document indices and scores
    result = []
    for doc_name, score in sorted_similarities[:top_n]:
        doc_idx = int(doc_name.split('_')[1])
        result.append((doc_idx, round(score, 4)))
    
    return result


def get_document_keywords(
    doc_index: int,
    feature_matrix: pd.DataFrame,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Extract top keywords from a document based on feature scores.

    Args:
        doc_index (int): Index of the document to extract keywords from.
        feature_matrix (pd.DataFrame): Feature matrix (BoW or TF-IDF).
        top_n (int): Number of top keywords to return. Default is 5.

    Returns:
        List[Tuple[str, float]]: List of (keyword, score) tuples,
                                 sorted by score in descending order.

    Raises:
        ValueError: If doc_index is out of range or top_n is invalid.
        TypeError: If feature_matrix is not a DataFrame.

    Example:
        >>> features = pd.DataFrame({'nlp': [0.9, 0.2], 'fun': [0.5, 0.8]})
        >>> get_document_keywords(0, features, top_n=2)
        [('nlp', 0.9), ('fun', 0.5)]
    """
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix must be a pandas DataFrame")
    if not isinstance(doc_index, int) or isinstance(doc_index, bool):
        raise TypeError("doc_index must be an integer")
    if doc_index < 0 or doc_index >= len(feature_matrix):
        raise ValueError(f"doc_index must be between 0 and {len(feature_matrix) - 1}")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    # Get feature scores for the document
    scores = feature_matrix.iloc[doc_index]

    # Sort by score (descending) and get top N
    top_features = scores.sort_values(ascending=False).head(top_n)

    # Convert to list of tuples
    result = [(feature, round(score, 4)) for feature, score in top_features.items()]
    
    return result


def compare_vectorization_methods(
    documents: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Compare BoW and TF-IDF representations side-by-side.

    Args:
        documents (List[str]): List of text documents to compare.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'bow' and 'tfidf' DataFrames.

    Raises:
        TypeError: If documents is not a list or contains non-strings.
        ValueError: If documents is empty or has fewer than 2 documents.

    Example:
        >>> docs = ["I love NLP", "NLP is fun"]
        >>> comparison = compare_vectorization_methods(docs)
        >>> 'bow' in comparison and 'tfidf' in comparison
        True
    """
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")
    if not documents:
        raise ValueError("Documents cannot be empty")
    if not all(isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")

    # Run analysis with both methods
    bow_results = analyze_documents(documents, vector_method="bow")
    tfidf_results = analyze_documents(documents, vector_method="tfidf")

    return {
        "bow": bow_results['vectorization']['feature_matrix'],
        "tfidf": tfidf_results['vectorization']['feature_matrix']
    }


if __name__ == "__main__":
    # Example usage
    sample_documents = [
        "Natural language processing is amazing and fun to learn.",
        "Machine learning and NLP are related fields in AI.",
        "Deep learning models are powerful for NLP tasks.",
        "Python is great for machine learning and data science."
    ]
    
    print("=" * 80)
    print("INTEGRATED NLP PIPELINE ANALYSIS")
    print("=" * 80)
    
    # Run comprehensive analysis
    results = analyze_documents(
        sample_documents,
        use_lemmatization=True,
        ngram_range=(1, 2),
        vector_method="tfidf"
    )
    
    # Display results
    print("\n1. PREPROCESSING STATISTICS")
    print("-" * 80)
    print(f"Total tokens: {results['preprocessed']['total_tokens']}")
    print(f"Unique tokens: {results['preprocessed']['unique_tokens']}")
    print(f"Average document length: {results['preprocessed']['avg_doc_length']}")
    
    print("\n2. VECTORIZATION")
    print("-" * 80)
    print(f"Method: {results['vectorization']['method'].upper()}")
    print(f"Feature matrix shape: {results['vectorization']['shape']}")
    print(f"Number of features: {len(results['vectorization']['feature_names'])}")
    
    print("\n3. WORD EMBEDDINGS")
    print("-" * 80)
    print(f"Vocabulary size: {results['word_embeddings']['vocabulary_size']}")
    print(f"Vector dimensionality: {results['word_embeddings']['vector_size']}")
    
    print("\n4. N-GRAM ANALYSIS")
    print("-" * 80)
    print(f"Top 5 bigrams:")
    for bigram, count in results['ngram_analysis']['top_bigrams'][:5]:
        print(f"  {bigram}: {count}")
    print(f"Bigram diversity: {results['ngram_analysis']['bigram_diversity']}")
    
    print("\n5. DOCUMENT SIMILARITY")
    print("-" * 80)
    print("Similarity matrix:")
    print(results['document_similarity'])
    
    print("\n6. SIMILAR DOCUMENTS TO DOC_0")
    print("-" * 80)
    similar_docs = find_similar_documents(0, results['document_similarity'], top_n=2)
    for doc_idx, similarity in similar_docs:
        print(f"  Document {doc_idx}: {similarity:.4f}")
    
    print("\n7. TOP KEYWORDS IN DOC_0")
    print("-" * 80)
    keywords = get_document_keywords(0, results['vectorization']['feature_matrix'], top_n=5)
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
    
    print("\n8. COMPARISON: BOW vs TF-IDF")
    print("-" * 80)
    comparison = compare_vectorization_methods(sample_documents[:3])
    print("BoW shape:", comparison['bow'].shape)
    print("TF-IDF shape:", comparison['tfidf'].shape)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
