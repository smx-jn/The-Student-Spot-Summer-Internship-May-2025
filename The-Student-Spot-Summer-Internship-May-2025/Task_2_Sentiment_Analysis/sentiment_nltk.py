import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary data
nltk.download('vader_lexicon')

# Sample sentences for testing
sentences = [
    "I love the new design of your website!",
    "This is the worst experience I've ever had.",
    "The product is okay, not too great but not bad either.",
    "Absolutely fantastic service!",
    "I'm disappointed with the support team."
]

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Analyze each sentence
for sentence in sentences:
    print(f"\nSentence: {sentence}")
    score = sia.polarity_scores(sentence)
    print("Sentiment Scores:", score)
    if score['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
    elif score['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")
