from process_labels import TweetProcessing
import spacy
from tqdm.auto import tqdm
from spacy.tokens import DocBin
nlp = spacy.blank('en')


class TweetClassifier:
    """
    much of this is based on the spacy documentation: https://spacy.io/usage/training#quickstart and this article on training a spaCy classifier: https://catherinebreslin.medium.com/text-classification-with-spacy-3-0-d945e2e8fc44
    """
    def __init__(self, train, test):
        self.train_data = train
        self.test_data = test


    def make_docs(self, data, outfile):

        doc_bin = DocBin()
        categories = ['increase', 'decrease', 'nochange']
    #   for (tweet, label) in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        for (tweet,label) in data:
            doc = nlp.make_doc(tweet)
            doc.cats = {category: 0 for category in categories} #set all categories to 0
            doc.cats[label] = 1 #set the correct one to 1
            doc_bin.add(doc)
        doc_bin.to_disk(outfile)

    def convert(self):
        #convert train data and store in binary file to disc (for spacy to process)
        train_docs = self.make_docs(self.train_data,"./data/train.spacy")
        #convert test data and store 
        test_docs = self.make_docs(self.test_data,"./data/test.spacy")


tweet_labeler = TweetProcessing()
tweet_labeler.preprocessing()

classifier = TweetClassifier(tweet_labeler.train_data, tweet_labeler.test_data)
classifier.convert()


"""
after preparing data for classification, I train the data with these terminal commands:
python -m spacy init config --pipeline textcat config.cfg
python -m spacy train config.cfg --paths.train ./data/train.spacy  --paths.dev ./data/test.spacy --output textcat_model

and then 
python -m spacy evaluate textcat_model/model-best/ --output metrics.json ./data/test.spacy
"""


