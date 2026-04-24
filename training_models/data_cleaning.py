def run_dbscan_1d(data, eps=0.6, min_samples=15):
    
    X = data.values.reshape(-1, 1)
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    
    return labels

def compute_centroids(data, labels):
    clusters = {}
    
    for label in set(labels):
        if label == -1: 
            continue
        
        cluster_points = data[labels == label]
        clusters[label] = {
            "centroid": cluster_points.mean(),
            "size": len(cluster_points)
        }
    
    return clusters

for col in columns:
    clusters = results[col]["clusters"]
    
    if len(clusters) > 0:
        centroids = [clusters[c]["centroid"] for c in clusters]
        final_value = np.mean(centroids)
    else:
        final_value = None
    
    results[col]["final_value"] = final_value
