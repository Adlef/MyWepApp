import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import re

def load_data(database_filepath):
    '''
    this function loads the DataBase and returns 2 DataFrames.
    Args:
        database_filepath: path to the DataBase.
    returns:
        X: DataFrame input for the Machine Learning Model.
        y: DataFrame output for the Machine Learning Model.
        category_names: disaster categories.
	'''
	# LOADING DATABASE FROM SQLITE DATABASE
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterMessages', engine)
    
    # CATEGORY NAMES
    category_names = list(df.columns[4:])

    # ENSURING WE KEEP ONLY FEATURES WITH 2 VALUES (0, 1)
    df = df[(df.related != 2) & (df[category_names].sum(axis=1) != 0)]
    
    X = df['message']
    y = df.drop(['id','message', 'original','genre'], axis=1)
    
    return X, y, category_names

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def tokenize(text):
    '''
    Tokenize a text into a vector of cleaned words.
    Args:
        text: text returned as vector cleaned.
    returns:
        clean_tokens: vector tokenized of the text cleaned.
    '''
    text = text.lower()
    
    # ENSURING THAT NO MAIL, URLS OR IPS ARE IN OUR TEXT
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    emails_regex = '[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+'
    ips_regex = '(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})'
    
    # STOPWORDS IN ENGLISH AND FRENCH
    stopwords_list = stopwords.words('english')

    # DETECTING URLS, MAILS OR IPS.
    # IT MIGHT BE THAT IN 1 TEXT MANY URLS / IPS / MAILS EXIST

    detected_urls = re.findall(url_regex,text)
    detected_emails = re.findall(emails_regex,text)
    detected_emails = [email.split()[0] for email in detected_emails]
    detected_ips = re.findall(ips_regex,text)
    
    # NORMALIZING TEXT
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # REMOVING NUMBERS
    num_pattern = re.compile(r'[^a-zA-Z]')
    text = re.sub(num_pattern,' ',text)
    
    # REPLACING ELEMENTS WITH ' '
    for url in detected_urls:
        text = text.replace(url,' ')    
                 
    for email in detected_emails:
        text = text.replace(email,' ')
            
    for ip in detected_ips:
        text = text.replace(ip,' ')       
    
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, get_wordnet_pos(tok)).strip()
        
        #REMOVING STOP WORDS
        if(clean_tok not in stopwords_list):
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    This functions builds the Machine Learning Model.
    Args:
        /
    returns:
        cv: Machine Learning model
    '''
    # DEFINING MACHINE LEARNING PIPELINE
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier())))
    ])
    
    # APPLYING GRIDSEARCH TO OPTIMIZE MODEL

    parameters = {
        'clf__estimator__n_estimators': [50,100,200],
        #'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__base_estimator__max_depth': [1, 2],
		#'vect__ngram_range': ((1,1), (1,2)),
        'tfidf__use_idf': (True, False),    
        'vect__max_features': (None, 10000),
        
    }
    
    # GRIDSEARCHCV SCORER FUNCTION
    scorer = make_scorer(f1_score,average='micro')
    cv = GridSearchCV(pipeline, scoring=scorer, param_grid=parameters, cv=3, verbose=2, n_jobs=1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating the model by printing the f1 score, precision ,recall and the best parameters of the model.
    '''
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print("Feature: {}".format(category))
        print(classification_report(Y_test[category], y_pred[:,i]))
        print("\n")
    
    print("Best Parameters: {}\n".format(model.best_params_))

def get_feature_importance(model, category_names, database_filepath):
    '''collect important features from model and store in database
    Function get the weights of most important (words)
    features, their weights, and the category in database
    'word' table after training
    Args:
      model: name of model
      category_names: list of category name of array Y
      database_filepath: name of database containing data
    Returns:
      None
    ''' 
    # TAKE THE BEST ESTIMATOR FROM MODEL (FROM GRIDSEARCHCV)
    best_pipeline = model.best_estimator_
    #best_pipeline = model
    col_name = []
    imp_value = []
    imp_word = []
    # List vocabulary
    x_name = best_pipeline.named_steps['vect'].get_feature_names()
    # GET FEATURE IMPORTANCES FROM THE LEARNING MODEL AND FOR A SPECIFIC CATEGORY
    for j, col in enumerate(category_names):
        x_imp = best_pipeline.named_steps['clf'].estimators_[j].feature_importances_
        # LIMIT FOR WEIGHT OF FEATURES SET TO MINMUM OF VALUE
        #value_max = x_imp.max() / 2.0
        value_max = 0.017
        
        # GET FEATURES NOT LESS THAN THE MINIMUM WEIGHT SET PER COLUMN - NO POINT TO DISPLAY ALL FEATURES
        for i,value in enumerate(x_imp):
            if(value > value_max):
                col_name.append(col)
                imp_value.append(value)
                imp_word.append(x_name[i])
                if col == 'cold':
                    print("great")

    # PREPARING DATAFRAME
    col_name = np.array(col_name).reshape(-1, 1)
    imp_value = np.array(imp_value).reshape(-1, 1)
    imp_word = np.array(imp_word).reshape(-1, 1)
    imp_array = np.concatenate((col_name, imp_value, imp_word), axis=1)
    df_imp = pd.DataFrame(imp_array, columns=['category_name', 'importance_value', 'important_word'])  
    
    # IMPORTANCE VALUE SHOULD BE A FLOAT
    df_imp.importance_value = pd.to_numeric(df_imp.importance_value, downcast='float')

    # CREATING SQL ENGINE
    engine = create_engine('sqlite:///' + database_filepath)

    # SAVING DATAFRAME INTO A TABLE
    df_imp.to_sql('Words', engine, if_exists='replace', index=False) 
    df_imp = pd.read_sql("SELECT * FROM Words", engine)
    
    print('Sample feature importance...')
    print(df_imp.head())

def save_model(model, model_filepath):
    '''
    Storing the model into a file.
	Args:
		model: the model to be stored.
		model_filepath: path / name given to the file which will contain the model.
	returns:
		/
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('\n')
        print('Saving feature importance...')
        get_feature_importance(model, category_names, database_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()