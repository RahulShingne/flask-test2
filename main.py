from flask import Flask, jsonify
import os
import pandas as pd
import requests
from io import StringIO

app=Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/read/')
def read_dataset():
    return jsonify(fetch_latest_dataset())

@app.route('/view/')
def view_data():
    column_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    num_bins = 5
    d=fetch_latest_dataset()
    df = pd.read_csv(StringIO(d))
    histograms = {}
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a Pandas DataFrame")
    for col_name in column_names:
        histograms[col_name], bins = pd.cut(df[col_name], bins=num_bins, retbins=True, include_lowest=True)
        result=""
    for col_name in column_names:
        histogram_data = histograms[col_name].value_counts().sort_index().rename_axis('Bin').reset_index(name='Number of data')
        result = result + "#" + histogram_data.to_string(index=False)
    return jsonify(result)

@app.route('/scaler/')
def data_normalization():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    d=fetch_latest_dataset()
    df = pd.read_csv(StringIO(d))
    scaler = MinMaxScaler()
    df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    return str(df.head())  

@app.route('/test/')
def testing():
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering

    d=fetch_latest_dataset()
    data = pd.read_csv(StringIO(d))
    X = data.drop(['CustomerID', 'Gender'], axis=1).values

    # KMeans
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    kmeans_labels = kmeans.labels_
    kmeans_score = silhouette_score(X, kmeans_labels)

    # DBSCAN
    dbscan = DBSCAN(eps=3, min_samples=2).fit(X)
    dbscan_labels = dbscan.labels_
    if len(set(dbscan_labels)) > 1:  
        dbscan_score = silhouette_score(X, dbscan_labels)
    else:
        dbscan_score = -1  


    agg = AgglomerativeClustering(n_clusters=5)
    agg.fit(X)
    labels = agg.labels_
    silhouette_avg = silhouette_score(X, labels)
    return jsonify(str(round(kmeans_score,4))+" "+str(round(dbscan_score,4))+" "+str(round(silhouette_avg,4)))

@app.route('/missing/')
def fill_missing_values():
    d=fetch_latest_dataset()
    data = pd.read_csv(StringIO(d))
    missing_values = data.isnull().sum()
    return jsonify(str(missing_values))

@app.route('/process_data/', methods=['GET'])
def process_data():
    input_data = request.form['input_data']
    # do something with the input_data, such as store it in a database or run some calculations
    response_data = "Received input data: {}".format(input_data)
    return response_data

@app.route('/outlier/')
def outlier():
    d=fetch_latest_dataset()
    df = pd.read_csv(StringIO(d))
    # Check for outliers and remove them
    outliers_age = df[(df['Age'] < 18) | (df['Age'] > 80)]
    outliers_income = df[(df['Annual Income (k$)'] < 10) | (df['Annual Income (k$)'] > 150)]
    outliers_spending = df[(df['Spending Score (1-100)'] < 0) | (df['Spending Score (1-100)'] > 100)]
    outliers_removed = pd.concat([outliers_age, outliers_income, outliers_spending]).drop_duplicates()
    df = df[(df['Age'] >= 18) & (df['Age'] <= 80)]
    df = df[(df['Annual Income (k$)'] >= 10) & (df['Annual Income (k$)'] <= 150)]
    df = df[(df['Spending Score (1-100)'] >= 0) & (df['Spending Score (1-100)'] <= 100)]
    return str(len(outliers_removed))


@app.route('/predict/')
def predict():
    '''income = request.form.get('Annual Income (Rs)')
    spending = request.form.get('Spending Score (1-100)')

    input_query=np.array([[income,spending]])'''

    d=fetch_latest_dataset()
    data = pd.read_csv(StringIO(d))

    df1=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
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


def fetch_latest_dataset():
    owner = "RahulShingne"
    repo = "flask-test2"
    branch = "main"


    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
    response.raise_for_status()

    latest_file = None
    for file in response.json()["tree"]:
        if file["path"].endswith(".csv") and (latest_file is None or file["sha"] < latest_file["sha"]):
            latest_file = file

    if latest_file is None:
        raise Exception("No CSV file found in the repository")

    response = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{latest_file['path']}")
    response.raise_for_status()
    return response.text



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
