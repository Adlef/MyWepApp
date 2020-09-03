from wrangling_scripts.DisasterMessages.train_classifier import tokenize

from myapp import app

app.run(host='0.0.0.0', port=3001, debug=True)