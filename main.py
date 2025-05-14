from tuning import parameter_grid_search
from data_gen import generate_blob_data, load_fashion_mnist_pca, generate_kddcup99
from DyDBSCAN import DynamicDBSCAN
from visualize import visualize_clusters_states

X, y_true = generate_kddcup99(n_samples=2000)


df = parameter_grid_search(X, y_true, minP_list=[5, 10, 15], t_list=[5, 10, 15], eps_list=[0.75, 1, 1.25, 1.5, 2, 3, 4, 5])
df.to_csv("evaluation_result.csv", index=False)
print(df.sort_values(by="ARI", ascending=False).head())
best_minP = df.sort_values(by="ARI", ascending=False).iloc[0]["minP"]
best_eps = df.sort_values(by="ARI", ascending=False).iloc[0]["eps"]
best_t = df.sort_values(by="ARI", ascending=False).iloc[0]["t"]

print(f"Best Params: eps={best_eps}, t={best_t}, minP={best_minP}")

# Re-run with best config, track labels over time
visualize_clusters_states(X, min_p=best_minP, t=best_t, eps=best_eps)
