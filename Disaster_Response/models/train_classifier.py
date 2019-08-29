import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stopword_list = stopwords.words('english')
    tokens = [lemmatizer.lemmatize(words.lower(), 'v')
              for words in word_tokenize(text) 
              if not words.lower() in stopword_list]
    return tokens


def build_model():
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20)))])
    parameters = {'clf__estimator__n_estimators': [10, 15], 'clf__estimator__min_samples_split': [2, 4]}
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print(category_names[i], ':')
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        print('\n')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    database_filepath, model_filepath = 'sqlite:///DisasterResponse.db', '../models/classifier.pkl'
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

    
if __name__ == '__main__':
    main()