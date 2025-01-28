from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, precision_recall_fscore_support

def train_kmeans(data, n_clusters):
    """Train a k-Means clustering model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    silhouette_avg = silhouette_score(data, labels)
    return kmeans, labels, silhouette_avg

def train_dbscan(data, eps, min_samples):
    """Train a DBSCAN clustering model."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    if len(set(labels)) > 1:
        db_score = davies_bouldin_score(data, labels)
    else:
        db_score = None  # No valid clustering
    return dbscan, labels, db_score

def train_isolation_forest(data, contamination):
    """Train an Isolation Forest anomaly detection model."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(data)
    scores = iso_forest.decision_function(data)
    labels = iso_forest.predict(data)
    labels = [1 if label == -1 else 0 for label in labels]  # Convert to binary labels
    return iso_forest, labels, scores

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using precision, recall, and F1-score."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1
