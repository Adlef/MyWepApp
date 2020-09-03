import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''
	This function loads csv files into DataFrames
	Args:
		messages_filepath: filepath to disaster message (.csv file).
		categories_filepath: filepath of disaster categories (.csv file).
	returns:
		df: merged DataFrame of the 2 csv files.
		
	'''
	# READ MESSAGES CSV FILE
	messages = pd.read_csv(messages_filepath)
	
	# READ CATEGORIES
	categories = pd.read_csv(categories_filepath)
	
	# CREATING DATAFRAME
	df = pd.merge(messages, categories, on='id',  how='left')
	return df

def clean_data(df):
    '''
    This function cleans the DataFrame previously loaded.
    Args:
         DataFrame.
    returns:
         cleaned DataFrame.
    '''
    # CREATING A DATAFRAME OF THE 36 INDIVIDUAL CATEGORY COLUMNS
    categories = df['categories'].str.split(';', expand=True)
    
    # SELECTING FIRST ROW OF THE CATEGORIES DATAFRAME
    row = categories.iloc[0]

    # EXTRACTING A LIST OF NEW COLUMN NAMES FOR CATEGORIES
    category_colnames = [cat.split('-')[0] for cat in row.unique()]
	
    # RENAMING CATEGORIES COLUMNS
    categories.columns = category_colnames

    for column in categories:
        # SETTING EACH VALUE TO BE THE LAST CHARCATER OF THE STRING
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # CONVERTING COLUMN FROM STRING TO NUMERIC
        categories[column] = pd.to_numeric(categories[column])
		
    # DROPPING ORIGINAL CATEGORIES COLUMNS FROM DF
    df = df.drop('categories', axis=1)
	
    # CONCATENING ORIGINAL DATAFRAME WITH NEW CATEGORIES DATAFRAME
    df = pd.concat([df,categories], join='inner', axis=1)

    # DROPPING DUPLICATES
    df = df.drop_duplicates()
	
    return df

def save_data(df, database_filename):
    '''
    This function stores a DataFrame into a SQLLITE DataBase.
    Args:
        df: DataFrame to be stored.
        database_filename: SQLITE DataBase name.
    returns:
        /    
    '''
    # SAVING CLEANED DATASET INTO SQLITE DATABASE 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, if_exists='replace', index=False)  

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()