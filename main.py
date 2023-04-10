from flask import Flask, jsonify,request
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

from io import StringIO

app=Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/read/')
def read_dataset():
    return jsonify(fetch_latest_dataset().to_dict())

@app.route('/view/')
def view_data():
    column_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    num_bins = 5
  
    df = fetch_latest_dataset()
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

@app.route('/output/')
def analysis():
    
    data = fetch_latest_dataset()

    df1 = data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
    X = df1[["Annual Income (k$)","Spending Score (1-100)"]]

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
    y = km.predict(X)
    df1["label"] = y

    result = {}
    for i in range(5):
        cluster = df1[df1["label"]==i]
        result[str(i)] = {
        "num_data_items": len(cluster),
        "mean_annual_income": cluster["Annual Income (k$)"].mean(),
        "mean_spending_score": cluster["Spending Score (1-100)"].mean()
    }

    return jsonify(result)

@app.route('/c1/')
def show_cluster1():
  
    data = fetch_latest_dataset()

    df1 = data[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X = df1[["Annual Income (k$)", "Spending Score (1-100)"]]

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
    df1["label"] = km.predict(X)

    cluster_df = df1[df1["label"] == 0]
    result_str = cluster_df.to_string(index=False)
    return jsonify(result_str)

@app.route('/c2/')
def show_cluster2():
    
    data = fetch_latest_dataset()

    df1 = data[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X = df1[["Annual Income (k$)", "Spending Score (1-100)"]]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
    df1["label"] = km.predict(X)
    cluster_df = df1[df1["label"] == 1]
    result_str = cluster_df.to_string(index=False)
    return jsonify(result_str)

@app.route('/c3/')
def show_cluster3():
    data = fetch_latest_dataset()

    df1 = data[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X = df1[["Annual Income (k$)", "Spending Score (1-100)"]]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
    df1["label"] = km.predict(X)
    cluster_df = df1[df1["label"] == 2]
    result_str = cluster_df.to_string(index=False)
    return jsonify(result_str)

@app.route('/c4/')
def show_cluster4():
    data = fetch_latest_dataset()

    df1 = data[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X = df1[["Annual Income (k$)", "Spending Score (1-100)"]]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
    df1["label"] = km.predict(X)
    cluster_df = df1[df1["label"] == 3]
    result_str = cluster_df.to_string(index=False)
    return jsonify(result_str)

@app.route('/c5/')
def show_cluster5():
  
    data = fetch_latest_dataset()

    df1 = data[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X = df1[["Annual Income (k$)", "Spending Score (1-100)"]]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
    df1["label"] = km.predict(X)
    cluster_df = df1[df1["label"] == 4]
    result_str = cluster_df.to_string(index=False)
    return jsonify(result_str)

@app.route('/scaler/')
def data_normalization():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    df = fetch_latest_dataset()
    scaler = MinMaxScaler()
    df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    return str(df.head())  

@app.route('/test/')
def testing():
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering

    data = fetch_latest_dataset()
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
  
    data = fetch_latest_dataset()
    missing_values = data.isnull().sum()
    return jsonify(str(missing_values))

@app.route('/process_data/', methods=['POST'])
def process_data():
    # Load the dataset
    data = pd.read_csv('Mall_Customers_KNN.csv')

    # Set the features and labels
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    y = data['Cluster'].values

    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    # Get the input data from the request
    data = request.get_json()
    age = data['input1']
    income = data['input2']
    score = data['input3']
    
    # Make the prediction
    prediction = knn.predict([[age, income, score]])
    
    # Return the prediction as a JSON response
    return jsonify({'cluster': prediction[0]})



@app.route('/outlier/')
def outlier():
    df = fetch_latest_dataset()
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

    data = fetch_latest_dataset()

    df1=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
    X=df1[["Annual Income (k$)","Spending Score (1-100)"]]
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
    '''owner = "RahulShingne"
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
    response.raise_for_status()'''
    data=pd.read_csv('customers_data.csv')
    return data

df = pd.read_csv('Mall_Customers_KNN.csv')

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model with the data
knn.fit(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], df['Cluster'])

@app.route('/knn', methods=['POST'])
def knn_method():
    # Get the request data
    data = request.get_json()
    age = data['Age']
    income = data['Annual Income (k$)']
    spending = data['Spending Score (1-100)']
    
    # Predict the cluster label using KNN
    prediction = knn.predict([[age, income, spending]])[0]
    
    # Return the predicted value as a JSON response
    return jsonify({'Cluster': int(prediction)})



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
