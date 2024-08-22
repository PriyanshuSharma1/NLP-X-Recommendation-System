#Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic
#that is intended to reflect how important a word is to a document in a collection or corpus. 
# It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. 
# The tf-idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, 
# which helps to adjust for the fact that some words appear more frequently in general.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sentences=["The cat jumped",
        "The white tiger roared",
        "Bird flying in the sky"]
vectorizer= TfidfVectorizer()
tfidf=vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()
tfidf_array=tfidf.toarray()
# Create a pandas DataFrame for better visualization
df = pd.DataFrame(data=tfidf_array, columns=feature_names)

# Add the original sentences as an index
df.index = [f"Sentence {i+1}" for i in range(len(sentences))]

# Display the DataFrame
print(df)
