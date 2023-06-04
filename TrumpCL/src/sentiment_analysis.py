import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from process_labels import TweetProcessing


def sentiment_analysis(text):
    """
    this function uses the nltk SentimentIntensityAnalyzer as a pretrained sentiment analysis model. 
    it returns the sentiment of any text. options are positive (pos), negative (neg), and neutral (neu)
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    result = ""
    if sentiment_scores['compound']>= 0.4:
        result = "pos"
    elif sentiment_scores['compound']<= -0.6:
        result =  "neg"
    else: result = "neu"
    return result

"""
Here I actually run the sentiment analysis of the tweets and store their results in my sentiments dict
"""
tweet_labeler = TweetProcessing()
tweet_labeler.preprocessing()
num_increased, num_decreased, num_nochange = [tweet_labeler.count_ratings[key] for key in ["increase", "decrease", "nochange"]]
sentiments = {
    "pos": {"increase": 0, "decrease": 0, "nochange": 0},
    "neg": {"increase": 0, "decrease": 0, "nochange": 0},
    "neu": {"increase": 0, "decrease": 0, "nochange": 0}
}

for tweet,change in tweet_labeler.all_data:
    sentiment = sentiment_analysis(tweet)
    sentiments[sentiment][change] += 1

def calculate_pct(sentiment):
    """
    this function calculates the percentages of how much a given sentiment corresponds with a change in approval rating.
    for example, calculate_pct("pos") returns 3 values, where each valye is the percentage of positive/negative/neutral
    sentiment tweets that correspond with an increase in approval rating.  
    """
    percentage_inc = sentiments[sentiment]["increase"]/num_increased
    percentage_dec = sentiments[sentiment]["decrease"]/num_decreased
    percentage_nch = sentiments[sentiment]["nochange"]/num_nochange
    return percentage_inc, percentage_dec, percentage_nch

print("Rating Counts:", tweet_labeler.count_ratings)
print("Sentiment Distribution:", sentiments)
print("Positive Sentiment Percentages (increase, decrease, nochange):", calculate_pct("pos"))
print("Negative Sentiment Percentages (increase, decrease, nochange):", calculate_pct("neg"))
print("Neutral Sentiment Percentages (increase, decrease, nochange):", calculate_pct("neu"))
