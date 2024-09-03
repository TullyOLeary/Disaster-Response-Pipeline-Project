import json
import plotly
import pandas as pd
from urllib.parse import quote as url_quote

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.figure_factory as ff
import joblib
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import string
from collections import Counter

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql_table('DisasterMessages', engine)
print(f"original shape: {df.shape}")
# Drop the 'child_alone' column
df = df.drop(columns=['child_alone'])
print("Shape of DataFrame after dropping 'child_alone':", df.shape)
print("Columns in DataFrame:", df.columns.tolist())
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # Distribution of Message Categories
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    graphs.append(
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45  # Rotate x-axis labels for better readability
                }
            }
        }
    )

    # Bar Chart for the Top N Words in Messages without Stopwords and Punctuation
    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    custom_stopwords = set(["'s", "ha", "wa", "n't", "would", "said", "http", "u", "''","like", "know", '..',"also"])

    # Tokenize all messages and filter out stopwords and punctuation
    all_tokens = [token for message in df['message'] for token in tokenize(message)]
    filtered_tokens = [token for token in all_tokens if token not in stop_words and token not in punctuation and token not in custom_stopwords]

    top_n = 20
    common_words = Counter(filtered_tokens).most_common(top_n)
    words, counts = zip(*common_words)

    graphs.append(
        {
            'data': [
                Bar(
                    x=words,
                    y=counts
                )
            ],
            'layout': {
                'title': f'Top {top_n} Most Common Words in Messages (excluding stopwords & punctuation)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    )

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render go.html with the classification result
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    # Test the model with a sample query
    query = "This is an example disaster message"
    prediction = model.predict([query])
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction for '{query}': {prediction}")
    main()
