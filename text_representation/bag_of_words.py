from sklearn.feature_extraction.text import CountVectorizer
sentences =["I love to eat Burgers","I love to eat momo", "I love to eat pizza"]
vectorizer= CountVectorizer()
bow = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())   

print(bow.toarray())