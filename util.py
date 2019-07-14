import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('english'))
from nltk import ngrams
import numpy as np
import json
import nltk
import json
import pandas as pd


class Utils:

    #cleans the text by removing external links and user mentions by '@' symbol
    def get_clean_text(self, text, keep_internal_punct=True):
        punctuation = string.punctuation
        text = str(text)
        text = re.sub(r'(@[A-Za-z0-9_]+)', '', text.lower())
        text = re.sub(r'\&\w*;', '', text.lower())
        text = re.sub(r'\$\w*', '', text.lower())
        text = re.sub(r'https?:\/\/.*\/\w*', '', text.lower())
        text = re.sub(r'#\w*', '', text.lower())
        text = re.sub(r'^RT[\s]+', '', text.lower())
        text = ''.join(c for c in text.lower() if c <= '\uFFFF')
        text = re.sub("[\(\[].*?[\)\]]", "", text.lower())
        if not keep_internal_punct:
            text = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', text.lower())
        return text.strip()

    #tokenizes the sentences
    def tokenize(self, text, keep_internal_punct=True):
        words = nltk.word_tokenize(text)
        if keep_internal_punct:
            return words
        else:
            words = [word.lower() for word in words if word.isalpha()]
            return words

    #removes the stop words
    def remove_stopwords(self, text):
        return [word for word in text if word not in stop_words]

    #lemmatizes the tokens
    def lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        lemmatized_token = []
        for word in text:
            lemmatized_token.append(lemmatizer.lemmatize(word))
        return lemmatized_token

    #stems the tokens
    def stemmer(self, text):
        stemmer = PorterStemmer()
        stemmed_tokens = []
        for word in text:
            stemmed_tokens.append(stemmer.stem(word))
        return stemmed_tokens

    #tokenizes the text by cleaning and processing as per the input
    def get_text_tokens(self, text, lemmatizing=True, stemming=True, keep_punctuation=True):
        cleaned_text = self.get_clean_text(text, keep_punctuation)
        text_tokens = self.tokenize(cleaned_text, keep_punctuation)
        text_tokens = self.remove_stopwords(text_tokens)
        if lemmatizing:
            text_tokens = self.lemmatize(text_tokens)
        if stemming:
            text_tokens = self.stemmer(text_tokens)
        return text_tokens

    #reads the muse embedding vector file
    def read_muse_vecs(self, muse_file):
        with open(muse_file, 'r',  errors='ignore', encoding="utf-8") as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word_list = line[0: len(line) - 300]
                curr_word = ""
                for t in curr_word_list:
                    curr_word = curr_word + str(t) + " "
                curr_word = curr_word.strip()
                words.add(curr_word)
                try:
                    word_to_vec_map[curr_word] = np.array(line[-300:], dtype=np.float64)
                except:
                    print(line, len(line))

            i = 1
            words_to_index = {}
            index_to_words = {}

            words.add("nokey")
            word_to_vec_map["nokey"] = np.zeros((300,), dtype=np.float64)

            for w in sorted(words):
                words_to_index[w] = i
                index_to_words[i] = w
                i = i + 1
        return words_to_index, index_to_words, word_to_vec_map

    #reade the json formatted review file and return the dataframe
    def read_review_file(self, review_file):
        rows = []
        with open(review_file, encoding="utf-8") as json_file:
            data = json.load(json_file)
            reviews = data["items"]["item"]
            for review in reviews:
                row = []
                row.append(review["summary"])
                row.append(review["text"])
                row.append(review["rating"])
                rows.append(row)

        df = pd.DataFrame(rows, columns=['Summary', 'Text', "Rating"])
        return df

    #tokenizes the review dataframe as per the provided input options
    def tokenize_reviews(self, df, keep_text=True, lemmatizing=True, stemming=True, keep_punctuation=True):
        X = []
        y = []
        for index, row in df.iterrows():
            tokens = []
            if row["Summary"]:
                tokens += self.get_text_tokens(row["Summary"], lemmatizing=lemmatizing, stemming=stemming, keep_punctuation=keep_punctuation)
            if keep_text and (row["Text"] is not None):
                tokens += self.get_text_tokens(row["Text"], lemmatizing=lemmatizing, stemming=stemming, keep_punctuation=keep_punctuation)
            if (row["Rating"] is not None) and (len(tokens) > 0):
                X.append(tokens)
                if float(row["Rating"]) > 3:
                    y.append(1)
                else:
                    y.append(0)
        return X, y

