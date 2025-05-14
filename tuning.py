import itertools
import pandas as pd
from DyDBSCAN import DynamicDBSCAN
from evaluator import ClusteringEvaluator
import time
from sklearn.datasets import make_blobs
from data_gen import generate_blob_data,  load_fashion_mnist_pca
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def parameter_grid_search(X, y_true, minP_list, t_list, eps_list):
    results = []
    for eps, t, minP in itertools.product(eps_list,t_list, minP_list):
        print(f"eps: {eps}, t: {t}, minP: {minP}")
        

        
        db = DynamicDBSCAN(minP, t, eps, X)
        
        predicted_labels = []
        
        for i in range(len(X)):
            predicted_labels.append(db.get_cluster(i))
        metrics = ClusteringEvaluator(y_true).evaluate(predicted_labels=predicted_labels)
        
        results.append({
            'minP': minP,
            "t": t,
            "eps": eps,
            "NMI": metrics["NMI"],
            "ARI": metrics["ARI"],
            
        })
        

    return pd.DataFrame(results)



# X, y_true = generate_kcc99cup()
# # X, y_true = generate_blob_data(n_samples=1000, n_features=5, center=10)
# # model=DynamicDBSCAN(minP=5, t=5, eps=0.5, initial_data=X)
# # y_pred = [model.get_cluster(i) for i in range(1000)]
# # print(f"ARI: {adjusted_rand_score(y_true, y_pred):.4f}, NMI: {normalized_mutual_info_score(y_true, y_pred):.4f}")
# # # X, y_true = generate_blob_data(n_samples=1000, n_features=5, center=10)
# # 
# # time.sleep(50)
# eps_list = [ 0.75, 1, 1.25, 1.5, 2, 3, 4, 5]
# min_samples_list = [5, 10, 15]
# hash_counts = [5, 10, 15]

# df_results = parameter_grid_search(X, y_true, min_samples_list,hash_counts, eps_list )

# df_results.to_csv("kddcup99_1.csv", index=False)

# # best score
# print(df_results.sort_values(by="ARI", ascending=False).head())