### Table of Contents

1. [Installation](#installation)
2. [My WebApp](#mywebapp)
3. [Structure of the Project](#structure)

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

'''
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
'''
The project is based on the following structure:
- a data folder where data documents are placed.
- wrangling_script folder where the data is being cleaned and prepared to be visualized in the html code.
- myapp folder which is the core of the app. It has the html code to make the website.