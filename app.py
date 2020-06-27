import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from flask import Flask, render_template, url_for, request,redirect #importing Flask framework
from textblob import TextBlob
from nltk.corpus import stopwords
import re
import csv
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk                                                         # impoting NLTK to use for stopwords removal
nltk.download('stopwords')


app = Flask(__name__)
model = pickle.load(open('irony.pkl', 'rb'))
stop = stopwords.words('english')                                   #downloading stopwords
vocab_size = 20000

@app.route('/')                 
def index():                                                        #sends data to index.html
    return render_template('index.html')
@app.route('/about')            
def about():                                                        #sends data to about.html
   return render_template("about.html")

@app.route('/sample')           
def sample():                                                       #sends data to sample.html
   return render_template("sample.html")
        
        
        
@app.route('/predict', methods=['GET', 'POST'])                      #enables to bidirectional data passing between python and html
def predict():                                                       #gets text from html, posts prediction 
    if request.method == "GET":             
        return render_template("index.html", predicted_text=text)    #gets data from index.html if available
    data=request.form["text_input"]
    f=data
    data=data.lower()
    data=re.sub('[^A-Za-z0-9]+', ' ', data)                          #preprocessing starts
    word_tokens = word_tokenize(data) 
    data = [w for w in word_tokens if not w in stop] 
    data = []
    for w in word_tokens: 
        if w not in stop: 
            data.append(w) 
    end=[one_hot(str(data), vocab_size)]
    padded_docs = pad_sequences(end,maxlen=20,padding='post')
    pred = model.predict_classes(padded_docs,batch_size=32)          #preprocessing ends
    if pred==1:
        return render_template('index.html', prediction_text="Irony",text_input=f)      
    else:
        return render_template('index.html', prediction_text="Not Irony",text_input=f)
 

@app.route("/feedback", methods=["GET", "POST"])       
def feedback():                                                      #posts feedback
  if request.method == "GET":
    return render_template("feedback.html")  
  elif request.method == "POST":
    userdata = dict(request.form)
    sentence = userdata["sentence"]
    a=sentence
    feed = userdata["feed"]
    with open('feedback.csv', mode='a') as csv_file:
      data = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      data.writerow([sentence, feed])                               #dataset is updated
  return render_template('feedback.html',text="Thank you!",text1=a)

if __name__ == '__main__': 
   app.run(threaded=False)
