import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

df.info()
print(df.describe())

plt.hist(df['Age'],bins= 12); plt.title("Age Distribution");plt.show()
plt.hist(df['Annual Income (k$)'], bins = 12); plt.title("Income k($)");plt.show()

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)']);
plt.xlabel('Annual Income (k$)'); plt.ylabel('Spending Score (1-100)');
plt.title('Annual Income vs Spending Score')
plt.show()

features = df[['Annual Income (k$)','Spending Score (1-100)']].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

inertia = []

K_range = range(1,11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features_scaled)
    inertia.append(km.inertia_)

plt.plot(list(K_range), inertia, marker = 'o')
plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow method"); plt.xticks(list(K_range))
plt.show()

k_final = 4

kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_scaled)
df['Cluster'] = labels

df['Cluster'].value_counts().sort_index()

centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centers_df.index.name = 'Cluster'


plt.figure(figsize=(7,5))
for c in range(k_final):
    subset = df[df['Cluster'] == c]
    plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'], label=f"Cluster {c}", s=40)
plt.scatter(centers_original[:,0], centers_original[:,1], marker='X', s=150, label='Centers')
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score")
plt.title("Clusters: Income vs Spending Score"); plt.legend(); plt.show()


income_med = df['Annual Income (k$)'].median()
score_med = df['Spending Score (1-100)'].median()

for i, row in centers_df.iterrows():
    income, score = row['Annual Income (k$)'], row['Spending Score (1-100)']
    if income >= income_med and score >= score_med:
        label = "High income, high spenders (Premium)"
    elif income >= income_med and score < score_med:
        label = "High income, low spenders (Potential upsell)"
    elif income < income_med and score >= score_med:
        label = "Low income, high spenders (Value buyers)"
    else:
        label = "Low income, low spenders (Budget-conscious)"
    print(f"Cluster {i}: Income={income:.1f}k$, Score={score:.1f} -> {label}")