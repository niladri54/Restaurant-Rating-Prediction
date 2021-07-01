import pickle
import numpy as np
from flask import Flask, request, render_template
import joblib
import gzip

app=Flask(__name__)
model=joblib.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[int(x) for x in request.form.values()]
    final_features=[np.array(features)]
    output=model.predict(final_features)
    op=np.round(output[0],1)
    return render_template('secondpage.html',value=op)

if __name__=='__main__':
    app.run(debug=True)
