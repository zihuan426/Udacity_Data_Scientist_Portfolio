import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on='id')
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # rename the columns of `categories`
    row = categories.loc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
      categories[column] = categories[column].str.get(-1)
      categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)

def main():

    messages_filepath, categories_filepath, database_filepath = ('disaster_messages.csv', 'disaster_categories.csv', 'DisasterResponse.db')

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')
    
    


if __name__ == '__main__':
    main()