
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords

app = Flask(__name__)

# Load data
df = pd.read_csv('sample_sentiment_data.csv')
df['processed_text'] = df['text'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stopwords.words('english')]))

# Create model pipeline
pipeline = make_pipeline(CountVectorizer(), RandomForestClassifier())

# Train model
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = ' '.join([word for word in text.lower().split() if word not in stopwords.words('english')])
        sentiment = pipeline.predict([processed_text])[0]
        return render_template('index.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    