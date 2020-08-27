from myapp import app
import json, plotly
from flask import render_template
from wrangling_scripts.wrangle_data_happiness import return_figures
from flask_navigation import Navigation

nav = Navigation(app)

nav.Bar('top', [
    nav.Item('Happiness in the World', 'happiness'),
])


@app.route('/')
#@app.route('/index')

#def index():

#    return render_template('index.html')

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