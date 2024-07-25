import pandas as pd
import numpy as np
import nltk
import re
import string
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
df = pd.read_csv("stress.csv")
df.head()
df. describe()
df.isnull().sum()
import nltk
import re
import string
nltk.download('stopwords')
from nltk.corpus import stopwords 
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["text"] = df["text"].apply(clean)
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(i for i in df.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
df["label"] = df["label"].map({0: "No Stress", 1: "Stress"})
df = df[["text", "label"]]
print(df.head())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, url_for

x = np.array(df["text"])
y = np.array(df["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,   test_size=0.33, random_state=42)
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(xtrain, ytrain)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form.get('text', '')
        result = process_input(input_text)
        return render_template('result.html', result=result)
    else:
        return render_template('index.html', css=url_for('static', filename='css/styles.css'))

app = Flask(__name__)

df = pd.read_csv("stress.csv")

# Rest of your code...
from flask import Flask, send_from_directory

# @app.route('/static/images/background-image.jpg')
# def serve_image():
#     return send_from_directory('/static/images', 'background-image.jpg')
   
# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)
        # output = "No Stress" if output == 0 else "Stress"
        return render_template('index.html', output=output)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
