import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

# Load dataset
df = pd.read_csv('./data/predicted_sentiment.csv')

# Load the models
nb_model = pickle.load(open('./model/nb.pkl', 'rb'))
nb_balanced_model = pickle.load(open('./model/nb_balanced.pkl', 'rb'))

# Read stop words from CSV without header
stopwords_df = pd.read_csv('./data/stopwords.csv', header=None, names=['stopword'])
list_stopwords = stopwords_df['stopword'].tolist()

# Sidebar menu for navigation
menu = st.sidebar.selectbox('Select an Option', ['Sentiment Analysis', 'Analysis Overview'])

# Function to predict sentiment using the chosen model
def predict_sentiment(text, model):
    if text:
        prediction = model.predict([text])  # Directly using the text as input since it's already vectorized
        return prediction[0]
    return None

if menu == 'Sentiment Analysis':
    # Title of the application
    st.title('YouTube Comment Sentiment Analysis')

    # Input text from the user (on the main page)
    st.subheader('User Input Text')
    user_input = st.text_area("Enter a comment for analysis:")

    # Choose the model for sentiment analysis
    model_choice = st.selectbox('Choose Model', ['Naive Bayes', 'Naive Bayes (Balanced)'])

    if st.button('Submit'):
        if user_input:
            selected_model = nb_model if model_choice == 'Naive Bayes (Balanced)' else nb_balanced_model
            predicted_sentiment = predict_sentiment(user_input, selected_model)

            # Display the prediction
            st.subheader('Sentiment Analysis Result')
            st.write(f"Predicted Sentiment for the entered comment: **{predicted_sentiment}**")

elif menu == 'Analysis Overview':
    # Clustering Analysis section
    st.title('Analysis Overview of YouTube Comments')

    # Word Cloud Visualization
    st.subheader('Word Cloud of Comments')

    # Combine all comments into a single string
    all_comments = ' '.join(df['cleaned_comment'])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=list_stopwords,
                          colormap='viridis',
                          random_state=42).generate(all_comments)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Top Words by Sentiment
    st.subheader('Top Words by Sentiment')

    def get_top_n_words(dataframe, n=10):
        # Combine comments into a single string
        all_comments = ' '.join(dataframe['cleaned_comment'])
        # Split into words and filter out stopwords
        words = [word for word in all_comments.split() if word not in list_stopwords]
        word_count = Counter(words)
        return word_count.most_common(n)

    # Filter comments by sentiment
    positive_comments = df[df['predicted_sentiment'] == 'positive']
    negative_comments = df[df['predicted_sentiment'] == 'negative']
    neutral_comments = df[df['predicted_sentiment'] == 'neutral']

    top_n_positive = get_top_n_words(positive_comments)
    top_n_negative = get_top_n_words(negative_comments)
    top_n_neutral = get_top_n_words(neutral_comments)

    # Create subplots for each sentiment
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot for positive
    sns.barplot(x=[x[1] for x in top_n_positive], y=[x[0] for x in top_n_positive], ax=axes[0])
    axes[0].set_title('Top Words in Positive Comments')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Words')

    # Plot for negative
    sns.barplot(x=[x[1] for x in top_n_negative], y=[x[0] for x in top_n_negative], ax=axes[1])
    axes[1].set_title('Top Words in Negative Comments')
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Words')

    # Plot for neutral
    sns.barplot(x=[x[1] for x in top_n_neutral], y=[x[0] for x in top_n_neutral], ax=axes[2])
    axes[2].set_title('Top Words in Neutral Comments')
    axes[2].set_xlabel('Count')
    axes[2].set_ylabel('Words')

    st.pyplot(fig)

    # Clustering Visualization
    st.subheader('Clustering Visualization')

    # Transform the cleaned comments using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=list_stopwords)
    X = vectorizer.fit_transform(df['cleaned_comment'])

    # KMeans clustering
    kmeans = KMeans(n_clusters=16, random_state=42)  # Adjust the number of clusters as needed
    kmeans.fit(X)

    # Reduce dimensionality for visualization using PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())
    df['cluster'] = kmeans.labels_

    # Clustering Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=df['cluster'], palette='viridis', alpha=0.7)
    plt.title('Clustering of YouTube Comments')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    st.pyplot(plt)

    # Word Frequency Table
    st.subheader('Most Common Words')
    # Filter out stopwords before counting
    filtered_words = [word for word in all_comments.split() if word not in list_stopwords]
    word_count = Counter(filtered_words)
    common_words_df = pd.DataFrame(word_count.most_common(20), columns=['Word', 'Frequency'])

    st.dataframe(common_words_df)
