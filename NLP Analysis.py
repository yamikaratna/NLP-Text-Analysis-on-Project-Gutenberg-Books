# Import necessary libraries
import nltk
import requests
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Force re-download of required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def load_text_from_gutenberg(url):
    """Fetch the text from a Project Gutenberg URL."""
    response = requests.get(url)
    text = response.text
    
    # Extract the actual book content
    start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    return text

def preprocess_text(text):
    """Tokenizes, lowercases, removes stopwords, and lemmatizes the text."""
    # Remove non-alphabetical characters
    text = re.sub(r'\W+', ' ', text)

    # Tokenize and lowercase
    tokens = text.lower().split()

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

    return processed_tokens

def get_top_n_tokens(tokens, n=10):
    """Returns a DataFrame of the top N most frequent tokens."""
    freq_dist = FreqDist(tokens)
    return pd.DataFrame(freq_dist.most_common(n), columns=['Word', 'Frequency'])

def get_top_n_bigrams(tokens, n=10):
    """Returns a DataFrame of the top N most common bigrams."""
    bigram_list = list(ngrams(tokens, 2))
    bigram_freq = Counter(bigram_list)
    return pd.DataFrame(bigram_freq.most_common(n), columns=['Bigram', 'Frequency'])

def compute_lexical_diversity(tokens):
    """Computes lexical diversity of the text."""
    return len(set(tokens)) / len(tokens) if tokens else 0

def generate_wordcloud(tokens):
    """Generates and displays a word cloud from tokens."""
    wordcloud = WordCloud(width=500, height=500, background_color="white").generate(" ".join(tokens))
    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def main():
    """Runs all NLP tasks in a single execution."""
    url = input("Enter Project Gutenberg URL: ")
    text = load_text_from_gutenberg(url)

    print("\n📖 Processing text... Please wait.")
    tokens = preprocess_text(text)

    # Display top 10 most frequent words
    print("\n🔢 Top 10 Most Frequent Words:")
    print(get_top_n_tokens(tokens))

    # Display top 10 most common bigrams
    print("\n🔠 Top 10 Most Common Bigrams:")
    print(get_top_n_bigrams(tokens))

    # Compute and display lexical diversity score
    lexical_diversity = compute_lexical_diversity(tokens)
    print(f"\n📊 Lexical Diversity Score: {lexical_diversity:.4f}")

    # Generate and display word cloud
    print("\n☁ Generating Word Cloud...")
    generate_wordcloud(tokens)

# Run the script
if __name__ == "__main__":
    main()


    
