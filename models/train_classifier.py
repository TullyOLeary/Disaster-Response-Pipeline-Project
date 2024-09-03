import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])
nltk.download(['punkt_tab', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.drop(columns=['message', 'original', 'genre'])

    # Drop 'child_alone' column if it exists in Y
    if 'child_alone' in Y.columns:
        Y = Y.drop(columns=['child_alone'])

    print("Columns in Y (target):", Y.columns)  # Debugging statement
    print("Sample data in Y:\n", Y.head())  # Debugging statement

    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # Normalize text: Convert to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words and perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    return tokens


def build_model():
    # Build a new pipeline with TF-IDF + SVM
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight='balanced')))
    ])

    # Optional: Use GridSearchCV for hyperparameter tuning
    param_grid = {
        'clf__estimator__C': [0.01, 0.1, 1, 10],  # Regularization parameter
    }

    grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, verbose=2, n_jobs=1)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    # Print classification reports for each category
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test[col], Y_pred[:, i]))
        print('------------------------------------------------------------')


def save_model(model, model_filepath):
    import joblib
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # Sample a smaller portion of the training data for quicker tuning
        print("Sampling 5% of the training data for quicker tuning...")
        X_train_sampled = X_train.sample(frac=0.05, random_state=42)  # 5% of the training data
        Y_train_sampled = Y_train.loc[X_train_sampled.index]
        print(f"Sampled {len(X_train_sampled)} instances from the training set.")

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train_sampled, Y_train_sampled)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
