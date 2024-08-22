from sklearn.feature_extraction.text import CountVectorizer
sentence=["the dog is in the house"]
tri_vectorizer= CountVectorizer(ngram_range=(2,2))
tri_grams=tri_vectorizer.fit_transform(sentence)
print("\nTri-gram Vocabulary:",tri_vectorizer.get_feature_names_out())
print("Tri-gram Matrix:\n",tri_grams.toarray())



