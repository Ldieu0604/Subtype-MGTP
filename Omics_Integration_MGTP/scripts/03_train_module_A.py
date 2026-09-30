import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# CÁC HÀM TẠO ĐỒ THỊ CHUẨN (TỪ UTILS.PY BẢN GỐC)
# ==========================================
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def graph_from_dist_tensor(dist, num_class, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    # Logic của Subtype-MGTP
    k=int(dist.shape[0]/num_class+1)  
    new_matrix = np.ones_like(dist.cpu().numpy())
    dist_np = dist.cpu().numpy()
    for idx, row in enumerate(dist_np):
        kth_smallest = np.partition(row, k-1)[:k]
        new_row = np.where(np.isin(row, kth_smallest), row, 1)
        new_matrix[idx] = new_row
    return new_matrix

def gen_adj_mat_tensor(data, num_class, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)   
    g = graph_from_dist_tensor(dist, num_class, self_dist=True) 

    adj = 1-g    
    diag_idx = np.diag_indices(adj.shape[0])
    adj[diag_idx[0], diag_idx[1]] = 0

    row_sums = adj.sum(axis=1)                          
    row_sums_expanded = np.expand_dims(row_sums, axis=1)
    # Xử lý chia cho 0
    row_sums_expanded[row_sums_expanded == 0] = 1
    adj = adj / row_sums_expanded      
    
    adj_T=adj.T
    adj=adj+adj_T                         

    adj = F.normalize(torch.from_numpy(adj).float(), p=1)  

    I = torch.eye(adj.shape[0])
    adj=adj+I
    adj = to_sparse(adj)    
    return adj

# ==========================================
# ĐỊNH NGHĨA MÔ HÌNH (SỬ DỤNG torch.sparse.mm GIỐNG BẢN GỐC)
# ==========================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) 
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight) 
        # Đã đổi lại dùng sparse.mm y như bản gốc
        output = torch.sparse.mm(adj, support) 
        if self.bias is not None:
            return output + self.bias
        return output

class GraphGCN(nn.Module):
    def __init__(self, nfeat, nhid, dim_final, dropout):
        super(GraphGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, dim_final)

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class GraphOmicsMultiGCNEncoder(nn.Module):
    def __init__(self, dim_list, dim_hid_list, dim_final, dropout):
        super(GraphOmicsMultiGCNEncoder, self).__init__()
        self.dropout = dropout
        self.Ge = GraphGCN(dim_list[0], dim_hid_list[0], dim_final, dropout)
        self.MUT = GraphGCN(dim_list[1], dim_hid_list[1], dim_final, dropout)
        self.Meth = GraphGCN(dim_list[2], dim_hid_list[2], dim_final, dropout)
        
    def forward(self, x_ge, x_mut, x_meth, adj_ge, adj_mut, adj_meth):
        z_ge = self.Ge(x_ge, adj_ge)
        z_mut = self.MUT(x_mut, adj_mut)
        z_meth = self.Meth(x_meth, adj_meth)
        z_out = (z_ge + z_mut + z_meth) / 3.0
        return z_out

# ==========================================
# HÀM MAIN
# ==========================================
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")

    # Tải dữ liệu đặc trưng (Features)
    print("1. Đang tải dữ liệu...")
    x_ge = torch.FloatTensor(np.load('../data/processed/Ge_features.npy')).to(device)
    x_mut = torch.FloatTensor(np.load('../data/processed/MUT_features.npy')).to(device)
    x_meth = torch.FloatTensor(np.load('../data/processed/Meth_features.npy')).to(device)

    # Tính toán đồ thị KNN Sparse theo chuẩn PyTorch của bản gốc
    # num_class trong bài báo gốc là 32 (cho Module A)
    print("2. Đang tạo ma trận kề (Sparse) trên thiết bị...")
    num_class = 32
    adj_ge = gen_adj_mat_tensor(x_ge, num_class).to(device)
    adj_mut = gen_adj_mat_tensor(x_mut, num_class).to(device)
    adj_meth = gen_adj_mat_tensor(x_meth, num_class).to(device)

    protein_df = pd.read_csv('../data/processed/Matched_Proteomics_for_GraphOmicDRP.csv')
    protein_matrix = protein_df.iloc[:, 3:].values
    
    valid_mask = ~np.isnan(protein_matrix[:, 0])
    protein_matrix = np.nan_to_num(protein_matrix)
    Y_target = torch.FloatTensor(protein_matrix).to(device)
    valid_mask_tensor = torch.BoolTensor(valid_mask).to(device)

    n_ge, n_mut, n_meth = x_ge.shape[1], x_mut.shape[1], x_meth.shape[1]
    dim_final = Y_target.shape[1] 
    dim_list = [n_ge, n_mut, n_meth]
    dim_hid_list = [256, 256, 256] 
    
    model = GraphOmicsMultiGCNEncoder(dim_list, dim_hid_list, dim_final, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.MSELoss()

    print("\n3. Bắt đầu huấn luyện...")
    epochs = 200
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_ge, x_mut, x_meth, adj_ge, adj_mut, adj_meth)
        loss = criterion(output[valid_mask_tensor], Y_target[valid_mask_tensor])
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch: {epoch+1:04d}, Loss (MSE): {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        final_embeddings = model(x_ge, x_mut, x_meth, adj_ge, adj_mut, adj_meth)
        final_embeddings_np = final_embeddings.cpu().numpy()
        
    np.save('../data/processed/Predicted_Proteomics_Embeddings.npy', final_embeddings_np)
    print("\n[THÀNH CÔNG] Đã lưu vector nhúng Protein hoàn chỉnh vào 'Predicted_Proteomics_Embeddings.npy'")

if __name__ == '__main__':
    main()
