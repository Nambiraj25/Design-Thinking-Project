from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
file_path_model='random_forest_Regression_model.pkl'
file_path_dataframe='random_forest_Regression_dataframe.pkl'
model = joblib.load(file_path_model)
dataframe=pd.read_pickle(file_path_dataframe)
print(type(model))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features = {"vehicletype":str(data['vehicletype']).lower(),
                    "gearbox":str(data['gearbox']).lower(),
                    "powerPS":float(data['powerPS']),
                    "model":str(data['model']).lower(),
                   "kilometer":int(data['kilometer']),
                   "fuelType":str(data['fueltype']).lower(),
                   "notRequiredDamage":str(data['notRequiredDamage']).lower(),
                   "Age":float(data['age']) 
        } 
        user_input=pd.DataFrame(features,index=[0])
        missing_col=list(set(dataframe.columns)-set(user_input.columns))
        user_input=pd.get_dummies(user_input,drop_first=True)
        missing_data=pd.DataFrame(0,columns=missing_col,index=user_input.index)
        user_input=pd.concat([user_input,missing_data],axis=1)
        user_input=user_input[dataframe.columns]
        user_prediction = model.predict(user_input)
        predicted_price=round(user_prediction[0],2)
        return render_template('index.html', predicted_price=predicted_price)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
