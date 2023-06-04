import csv
import re
from datetime import datetime, timedelta, date
import spacy
from sklearn.model_selection import train_test_split


class TweetProcessing:
    def __init__(self):
        self.tweets = {}
        self.ratings = {}
        self.combined_data = {}
        self.all_data = []
        self.count_ratings = {"increase": 0, "decrease": 0, "nochange": 0}
        self.train_data = []
        self.dev_data = []
        self.test_data = []

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_mentions(self, text):
        mention_pattern = re.compile(r'@\w+')
        return mention_pattern.sub('', text)
    
    def remove_special_characters(self, text):
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)
    
    def clean_tweet(self, text):
        """this function cleans tweets of urls, @ mentions, and any special characters"""
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_special_characters(text)
        return text

    def read_tweets(self, infile='./raw_data/tweets.csv'):
        """this function reads in a csv file of tweets from Donals Trump and their dates, storing the results in self.tweets"""
        start_date = datetime.strptime("2017-01-23", "%Y-%m-%d").date()
        end_date = datetime.strptime("2021-01-20", "%Y-%m-%d").date()
        with open(infile, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) < 9: continue #make sure all data is there
                text = self.clean_tweet(row[1]) #clean the tweet
                date = row[7]
                datetime_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                date_only = datetime_object.date()
                if start_date < date_only < end_date:
                    self.tweets[datetime_object] = text

    def read_ratings(self, infile='./raw_data/approval_ratings.csv'):
        """this function reads in a csv file of Donals Trump's approval rating by date, storing the results in self.ratings"""
        with open(infile, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) < 10:
                    continue
                if row[1] == "All polls":
                    date = row[2]
                    datetime_object = datetime.strptime(date, "%m/%d/%Y")
                    date_only = datetime_object.date()
                    rating = float(row[3])
                    self.ratings[date_only] = rating

    def combine_tweets_ratings(self):
        """
        this function associates each tweet with a percent change in approval rating. 
        if the rating increased by at least 0.5% from the day before to the day after the tweet, 
        then i give a label "increase". if less than  0.5%, then i say "decrease", and in between 
        is "no change".
        """
        num_increase = 0
        num_decrease = 0
        num_nochange = 0
        for tweet in self.tweets:
            text = self.tweets[tweet]
            date_only = tweet.date()
            previous_day = date_only - timedelta(days=1)
            next_day = date_only + timedelta(days=1)
            try:
                pct_change = (self.ratings[next_day] - self.ratings[previous_day])/self.ratings[previous_day]
                if text not in self.combined_data:
                    if pct_change > 0.005:
                        self.combined_data[text] = "increase"
                        self.count_ratings["increase"]+=1
                    elif pct_change < -0.005:
                        self.combined_data[text] = "decrease"
                        self.count_ratings["decrease"]+=1
                    else:
                        num_nochange +=1
                        self.combined_data[text] = "nochange"
                        self.count_ratings["nochange"]+=1
            except:
                print("somethings up...debug time!")

    def preprocessing(self):
        """
        this function sets up training and test sets for which to train and test a classifier 
        for the tweets and their change in approval rating. It also sets a list of all data
        for the sentiment analysis work
        """
        self.read_tweets()
        self.read_ratings()
        self.combine_tweets_ratings()

        labels = list(self.combined_data.values())
        docs = list(self.combined_data.keys())

        #split into 80% test and 20% train
        tweet_train, tweet_test, label_train, label_test = train_test_split(docs, labels, test_size=0.2, random_state=42)

        self.train_data = list(zip(tweet_train, label_train))
        self.test_data = list(zip(tweet_test, label_test))
        self.all_data = list(zip(docs, labels))
        