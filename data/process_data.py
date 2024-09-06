import pandas as pd
from sqlalchemy import create_engine
import sys

messages_filepath = "disaster_messages.csv"
categories_filepath = "disaster_categories.csv"
database_filepath = 'DisasterResponse.db'


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath: str. Filepath for the csv file containing messages dataset.
    categories_filepath: str. Filepath for the csv file containing categories dataset.

    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting values to binary, and removing duplicates.

    Args:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.

    Returns:
    df: dataframe. Dataframe containing cleaned version of the merged content.
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # Extract column names for the categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
        
        # Specifically convert 'related' values of 2 to 1
        if column == 'related':
            categories[column] = categories[column].replace(2, 1)

    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.

    Args:
    df: dataframe. Cleaned dataframe.
    database_filename: str. Filename for the output database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()



