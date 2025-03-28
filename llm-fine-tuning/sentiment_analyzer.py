import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class SemanticEmotionAnalyzer:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        
        # Initialize VADER sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Load SpaCy model
        try:
            self.nlp = spacy.load('en_core_web_md')
        except OSError:
            print("SpaCy model not found. Please download using: python -m spacy download en_core_web_md")
        
        # Emotion lexicon
        self.emotion_lexicon = {
            'anxiety': ['worry', 'nervous', 'stress', 'panic', 'afraid'],
            'depression': ['sad', 'hopeless', 'tired', 'worthless', 'lonely'],
            'anger': ['frustrated', 'irritated', 'furious', 'mad', 'resentful'],
            'joy': ['happy', 'excited', 'grateful', 'content', 'optimistic'],
            'fear': ['scared', 'terrified', 'anxious', 'dread']
        }
    
    def detect_emotion(self, text):
        """
        Perform comprehensive emotion detection
        """
        # Sentiment scores
        sentiment_scores = self.sia.polarity_scores(text)
        
        # Emotion lexicon matching
        emotion_scores = {}
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        for emotion, keywords in self.emotion_lexicon.items():
            emotion_scores[emotion] = sum(1 for token in tokens if token in keywords)
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
        
        return {
            'sentiment': sentiment_scores,
            'emotion_intensity': emotion_scores,
            'dominant_emotion': dominant_emotion
        }
    
    def generate_emotion_summary(self, emotion_analysis):
        """
        Generate a human-readable emotion summary
        """
        compound_score = emotion_analysis['sentiment']['compound']
        dominant_emotion = emotion_analysis['dominant_emotion']
        
        if compound_score > 0.05:
            base_summary = "You seem to be feeling positive "
        elif compound_score < -0.05:
            base_summary = "You appear to be experiencing some negative emotions "
        else:
            base_summary = "You seem to be in a neutral state "
        
        return f"{base_summary}with a sense of {dominant_emotion}."
