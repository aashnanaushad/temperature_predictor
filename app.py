from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("pred_temp.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    ctemp=int_features[0]
    features = [int_features[1],int_features[2],int_features[3]]
    final=[np.array(features)]
    print(features)
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    newtemp = ctemp + float(output)*ctemp
    print(output)
    print(newtemp)
    if output>str(0.5):
        return render_template('pred_temp.html',pred='Probability of temperature increase is more than 50% \n Possible temperature after 15 mins is {}'.format(newtemp))
    else:
        return render_template('pred_temp.html',pred='Probability of temperature increase is less than 50% \nPossible temperature after 15 mins is {}'.format(newtemp))
   

if __name__ == '__main__':
    app.run(debug=True)
