from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/training_state')
def get_training_state():
    try:
        with open('static/training_state.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'epoch': 0, 'batch': 0, 'losses': []})

if __name__ == '__main__':
    app.run(debug=True) 