from myapp import app
import json, plotly
from flask import render_template, request
from wrangling_scripts.WorldHappiness.wrangle_data_happiness import return_figures
from wrangling_scripts.DisasterMessages.disasters_messages import *
from flask_navigation import Navigation

nav = Navigation(app)

nav.Bar('top', [
    nav.Item('Happiness in the World', 'happiness'),
	nav.Item('Disaster Messages Classification', 'disasters_1_2'),
])


@app.route('/')

@app.route('/happiness')

def happiness():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('happiness.html',
                           ids=ids,
                           figuresJSON=figuresJSON)



graphs, model, df = return_figures_disaster()
    
@app.route('/disasters_1_2')



def disasters_1_2():
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('disasters_1_2.html', ids=ids, graphJSON=graphJSON)

@app.route('/disasters_2_2')

def disasters_2_2():

    # save user text input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.best_estimator_.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the disaster_2_2.html Please see that file. 
    return render_template(
        'disasters_2_2.html',
        query=query,
        classification_result=classification_results
    )