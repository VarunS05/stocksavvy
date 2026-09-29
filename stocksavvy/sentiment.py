"""News sentiment scoring using VADER (tuned for short, informal text like
headlines and social posts -- a better fit here than a general-purpose
polarity classifier).
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text):
    """Returns {neg, neu, pos, compound} for a piece of news text.

    compound is VADER's normalized aggregate score in [-1, 1]; neg/neu/pos
    are proportions of the text attributed to each sentiment and sum to ~1.
    """
    if not text or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _analyzer.polarity_scores(text)
