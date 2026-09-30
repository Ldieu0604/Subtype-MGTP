import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        output = torch.sparse.mm(adj, support) 
        if self.bias is not None:
            return output + self.bias
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_channels, out_channels*2)
        self.gc2 = GraphConvolution(out_channels*2, out_channels)

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_channels, in_channels*2)
        self.gc2 = GraphConvolution(in_channels*2, out_channels)

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class Model(nn.Module):
    def __init__(self, encoder, decoder, num_sample):
        super(Model, self).__init__()
        self.n = num_sample
        self.encoder = encoder
        self.decoder = decoder
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(self.n, self.n, dtype=torch.float32), requires_grad=True)

    def forward(self, x, edge_index):
        H = self.encoder(x, edge_index) 
        Coefficient = self.Coefficient
        CH = torch.matmul(Coefficient, H) 
        X_ = self.decoder(CH, edge_index) 
        return H, CH, Coefficient, X_

# ==========================================
# HÀM CONTRASTIVE LOSS (BẢN GỐC)
# ==========================================
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x / 1.0)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def instanceloss(z1: torch.Tensor, z2: torch.Tensor):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    return ((l1 + l2) * 0.5).mean()

# ==========================================
# HÀM MAIN
# ==========================================
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")

    print("1. Tải đặc trưng Protein ảo (Đầu ra từ Module A)...")
    protein_features = np.load('../data/processed/Predicted_Proteomics_Embeddings.npy')
    data = torch.FloatTensor(protein_features).to(device)
    
    print("2. Xây dựng đồ thị KNN theo kiểu PyTorch Sparse (Bản gốc)...")
    num_class = 6 # Theo args.cluster_num mặc định của bài gốc B_main.py
    graph = gen_adj_mat_tensor(data, num_class).to(device)

    num_samples = len(data)
    in_channels = data.shape[1]
    out_channels = 128 
    
    print("3. Khởi tạo Autoencoder và Contrastive Model (Module B)...")
    encoder = Encoder(in_channels, out_channels, dropout=0.1).to(device)
    decoder = Decoder(out_channels, in_channels, dropout=0.1).to(device)
    model = Model(encoder, decoder, num_samples).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    print("\n4. Bắt đầu huấn luyện...")
    epochs = 600
    model.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        H, CH, Coefficient, X_ = model(data, graph) 
        
        rec_loss = torch.sum(torch.pow(data - X_, 2))
        loss_instance = instanceloss(H, CH)
        loss_coef = torch.sum(torch.pow(Coefficient, 2))
      
        loss = 1.0 * loss_instance + 1.0 * loss_coef + 0.01 * rec_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 30 == 0:
            print(f'Epoch={epoch:03d}, Tổng Loss={loss:.4f}, Instance={loss_instance:.4f}, Rec={rec_loss:.4f}, Coef={loss_coef:.4f}')

    model.eval()
    with torch.no_grad():
        H, CH, Coefficient, X_ = model(data, graph) 
        final_cell_line_embeddings = H.cpu().numpy()
        
    np.save('../data/processed/Final_CellLine_Embeddings_for_DRP.npy', final_cell_line_embeddings)
    print("\n[THÀNH CÔNG] Đã lưu vector nhúng tinh tuý nhất của Cell Line vào 'Final_CellLine_Embeddings_for_DRP.npy'")

if __name__ == '__main__':
    main()
