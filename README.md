# CompLing-Twitter-Analysis
To run the classification model, first run the classification.py file, and then run: python -m spacy train config.cfg --paths.train ./data/train.spacy --paths.dev ./data/test.spacy --output textcat_model and
python -m spacy evaluate textcat_model/model-best/ --output metrics.json ./data/test.spacy To run the sentiment analysis, run the sentiment analysis file
