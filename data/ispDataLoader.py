import csv
import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy

orig_dd = pd.read_csv("AOA_1539000552401_1_ISP-Data-For-Anomaly-Detection/all_data.csv", encoding="UTF-8")
orig_dd = orig_dd.iloc[:, 2:]
orig_dd = orig_dd.astype(np.float32)


orig_data_tensor = torch.FloatTensor(orig_dd.values)
orig_data_tensor= orig_data_tensor.reshape([orig_data_tensor.shape[0], orig_data_tensor.shape[1],1,1])
orig_data_tensor.shape

train_data = orig_data_tensor[: orig_data_tensor.shape[0] // 7 * 5, :,:,:]
val_data = orig_data_tensor[orig_data_tensor.shape[0] // 7 * 5: orig_data_tensor.shape[0] // 7 * 6, :,:,:]
test_data = orig_data_tensor[orig_data_tensor.shape[0] // 7 * 6:, :,:,:]

G = nx.random_regular_graph(4, 40, seed=2050)
A = nx.to_scipy_sparse_matrix(G, format='csr')
n, m = A.shape
diags = A.sum(axis=1)
D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')

A=A.astype(dtype='float32')
obj_matrix = torch.FloatTensor(A.toarray())



results = [obj_matrix, train_data, val_data, test_data]

with open("isp_data.pickle", 'wb') as f:
    pickle.dump(results, f)
