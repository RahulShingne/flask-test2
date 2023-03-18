from flask import Flask, jsonify
import os
import pandas as pd

app=Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict/')
def predict():
    '''income = request.form.get('Annual Income (Rs)')
    spending = request.form.get('Spending Score (1-100)')

    input_query=np.array([[income,spending]])'''

    data=pd.read_csv("Dataset.csv")

    df1=data[["CustomerID","Gender","Age","Annual Income (Rs)","Spending Score (1-100)"]]

    X=df1[["Annual Income (Rs)","Spending Score (1-100)"]]
    X.head()

    from sklearn.cluster import KMeans
    wcss=[]

    for i in range(1,11):
        km=KMeans(n_clusters=i)
        km.fit(X)
        wcss.append(km.inertia_)

    km1=KMeans(n_clusters=5)

    km1.fit(X)

    y=km1.predict(X)

    df1["label"] = y
    df1.head()
    cust1=df1[df1["label"]==1]
    cust2=df1[df1["label"]==2]
    cust3=df1[df1["label"]==0]
    cust4=df1[df1["label"]==3]
    cust5=df1[df1["label"]==4]
    result = {
        
        "A":cust1["CustomerID"].values,
        "B":cust2["CustomerID"].values,
        "C":cust3["CustomerID"].values,
        "D":cust4["CustomerID"].values,
        "E":cust5["CustomerID"].values


    }
    return jsonify(str(result))


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
