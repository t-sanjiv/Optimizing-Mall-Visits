import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,:].values
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()
from sklearn.cluster import AgglomeractiveClustering
clustering=AgglomerativeClustering(n_clusters=5)
y_hc=clustering.fit_predict(x)
plt.scattering(X[y_hc==0,0], X[y_hc==0,1], c="red", label="C1")
plt.scattering(X[y_hc==1,0], X[y_hc==1,1], c="blue", label="C2")
plt.scattering(X[y_hc==2,0], X[y_hc==2,1], c="green", label="C3")
plt.scattering(X[y_hc==3,0], X[y_hc==3,1], c="orange", label="C4")
plt.scattering(X[y_hc==4,0], X[y_hc==4,1], c="black", label="C5")
plt.title("Cluster of Customers")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()