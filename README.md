### Table of Contents

1. [Installation](#installation)
2. [My WebApp](#mywebapp)
3. [Structure of the Project](#structure)
4. [Running the code](#running)

## Installation <a name="installation"></a>

All the necessary python libraries are available in the requirements.txt file. The application has been built based on the Anaconda distribution.

## MyWepApp <a name="mywebapp"></a>

This project implements a Web Application available at Heroku: https://my-data-science-app.herokuapp.com/
Currently, it has:
- links to my linkedin & github.
- links to my articles.
- a first WorldHappiness page which has some initial graphics.

The content will be updated time to time.

## Structure of the Project <a name="structure"></a>

```
|   myapp.py
|   nltk.txt
|   Procfile
|   README.md
|   requirements.txt
|   
+---data
|   +---DisasterMessages
|   |       DisasterResponse.db
|   |       disaster_categories.csv
|   |       disaster_messages.csv
|   |       process_data.py
|   |       
|   \---WorldHappiness
|           2015.csv
|           2016.csv
|           2017.csv
|           2018.csv
|           2019.csv
|           
+---myapp
|   |   routes.py
|   |   __init__.py
|   |   
|   +---static
|   |   \---img
|   |           githublogo.png
|   |           linkedinlogo.png
|   |           
|   \---templates
|       |   disasters_1_2.html
|       |   disasters_2_2.html
|       |   happiness.html
|       |   index.html
|       |   
|       \---includes
|               _navbar.html
|               
\---wrangling_scripts
    +---DisasterMessages
    |   |   disasters_messages.py
    |   |   train_classifier.py
    |   |   __init__.py
    |   |   
    |   \---model
    |           model_ada.sav
    |           model_ada_v1.sav
    |           model_ada_v2.sav
    |           
    \---WorldHappiness
            wrangle_data_happiness.py
            __init__.py
```
The project is based on the following structure:
- a data folder where data documents are placed for the different html pages.
- wrangling_script folder where the data is being cleaned and prepared to be visualized in the html code.
- myapp folder which is the core of the app. It has the html code to make the website.
- requirements.txt and nltk.txt are the list of tools necessary for the app.

## Running the code <a name="running"></a>

0. Clone the repository use: `git@github.com:Adlef/MyWepApp.git`, and pip install `requirement.txt`
```
conda update python
python3 -m venv name_of_your_choosing
source name_of_your_choosing/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                      # install packages in requirement
```

1. To update the Disaster Messages Project included in the WebApp if necessary

The whole project can be found there: `https://github.com/Adlef/DisasterAnalysis`

    - To run ETL pipeline that cleans data and stores in database
        `python data/DisasterMessages/process_data.py data/DisasterMessages/disaster_messages.csv data/DisasterMessages/disaster_categories.csv data/DisasterMessages/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves it (make sure if don't have gpu available, in build_model function pipeline remove n_jobs=-1)
        `python wrangling_scripts/DisasterMessages/train_classifier.py data/DisasterMessages/DisasterResponse.db wrangling_scripts/DisasterMessages/models/model_ada.sav`

2. To run the WebPage: `python myapp.py`

3. Go to `http://0.0.0.0:3001/`

## Some discussion about the unbalanced categories:

- The original csv files after merging, contained about 20% samples that did not have labels. After investigating content of those messages, it seemed that the no-label is not part of a one-hot coding strategy. Many of those messages were related to actual events (e.g. fire, aid, ...) but were not labeled. Those samples as well as ~200 rows of data with label `2` belonging to `related` category were removed. Some of these messages also showed translation from `original` message column in English to `message` column in Spanish!, keeping those messages when no translation algorithms is not applied to them before pipeline would impact the algorithm performance considering their sample size.
- The multioutput method in the pipeline, applies Adaboost algorithm to each category. Considering its nature Adaboost puts more weight on mislabeled samples during the training. This seems suitable over methods such is Random Forest that are less accurate and slower.
- The data for several categories is highly unbalanced with even less than 2% sample for positive class. Therefore using accuracy metrics is inappropriate for the optimization task. Since accuracy will not penalize the class with low sample size. For example for a very important category `missing` the positive label was less than `5%` of the data. Mis-identifying the missing people messages (high False negative rate) is extremely consequential. Another side of the story is identifying events incorrectly, for example algorithm predict `fire` incorrectly, or have high false positive rate, which can lead to sending resource to places that are not affected. This is also costly. So a delicate balance between handling `FN` and `FP` is needed. For this analysis, recall metrics to catch the `FN` and precision to get `FP`, Fscore (F1, F_beta), or roc_auc_score are more appropriate. In order to ensure better performance, algorithm is optimized using f1_score (combination of recall and precision) but also tested with roc_auc score, which showed not significant improvement over f1_score.
- Even after considering different metrics to deal with unbalanced categories, prediction is not ideal for certain cases. More obvious remedy is collecting more data in those specific scenarios. Down sampling the negative class is also an option when the vocabulary integrity will not get jeopardized.

<p align="center"> 
<img src="https://github.com/ania4data/Disaster_response_pipeline/blob/master/app/static/category_selection_app.png" style="width:30%">
</p>

