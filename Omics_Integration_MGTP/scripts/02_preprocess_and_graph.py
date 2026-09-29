import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

def main():
    # 1. Đọc danh sách chuẩn 1074 Cell line (Lấy từ cột 'Name' vì cột này chứa Cosmic ID)
    print("1. Đang tải danh sách Cell line chuẩn...")
    cell_list = pd.read_csv('../data/raw/Cell_list.csv')
    master_ids = cell_list['Name'].astype(str).str.strip().values
    print(f"  => Tổng số cell line chuẩn: {len(master_ids)}")

    # ==========================================
    # 2. Xử lý dữ liệu Đột biến (MUT)
    # ==========================================
    print("\n2. Đang xử lý dữ liệu Đột biến (MUT)...")
    mut_df = pd.read_csv('../data/raw/PANCANCER_Genetic_feature.csv')
    # Xoay trục (Pivot)
    mut_pivot = mut_df.pivot_table(index='cosmic_sample_id', columns='genetic_feature', values='is_mutated', fill_value=0)
    mut_pivot.index = mut_pivot.index.astype(str).str.strip()
    
    # Căn chỉnh (Align) theo master_ids
    mut_aligned = mut_pivot.reindex(master_ids).fillna(0).values
    print(f"  => Kích thước ma trận đặc trưng MUT: {mut_aligned.shape}")
    np.save('../data/processed/MUT_features.npy', mut_aligned)

    # ==========================================
    # 3. Xử lý dữ liệu Methyl hóa (Meth)
    # ==========================================
    print("\n3. Đang xử lý dữ liệu Methyl hóa (Meth)...")
    meth_df = pd.read_csv('../data/raw/METH_CELLLINES_BEMs_PANCAN2.csv')
    # Xoay trục (Pivot)
    meth_pivot = meth_df.pivot_table(index='V1', columns='V2', values='V3', fill_value=0)
    meth_pivot.index = meth_pivot.index.astype(float).astype(int).astype(str).str.strip()
    
    # Căn chỉnh
    meth_aligned = meth_pivot.reindex(master_ids).fillna(0).values
    print(f"  => Kích thước ma trận đặc trưng Meth: {meth_aligned.shape}")
    np.save('../data/processed/Meth_features.npy', meth_aligned)

    # ==========================================
    # 4. Xử lý dữ liệu Gene Expression (Ge)
    # ==========================================
    print("\n4. Đang xử lý dữ liệu Biểu hiện Gen (Ge)... (Quá trình này hơi lâu do file 300MB)")
    ge_df = pd.read_csv('../data/raw/Cell_line_RMA_proc_basalExp.txt', sep='\t')
    
    # Set index là tên Gen, bỏ cột GENE_title nếu có
    ge_df.set_index('GENE_SYMBOLS', inplace=True)
    if 'GENE_title' in ge_df.columns:
        ge_df.drop(columns=['GENE_title'], inplace=True)
        
    # Transpose (xoay) để Cell line thành hàng, Gen thành cột
    ge_pivot = ge_df.T
    ge_pivot.index = ge_pivot.index.str.replace('DATA.', '', regex=False).str.strip()
    
    # Căn chỉnh
    ge_aligned = ge_pivot.reindex(master_ids).fillna(0).values
    print(f"  => Kích thước ma trận Ge ban đầu: {ge_aligned.shape}")
    
    # Dùng chuẩn PCA của scikit-learn
    print("  - Đang chạy scikit-learn PCA để giảm chiều Ge xuống 512 chiều...")
    pca = PCA(n_components=512)
    ge_reduced = pca.fit_transform(ge_aligned)
    print(f"  => Kích thước ma trận Ge sau PCA: {ge_reduced.shape}")
    np.save('../data/processed/Ge_features.npy', ge_reduced)

    print("\n[HOÀN TẤT] Đã lưu 3 ma trận đặc trưng thành công!")
    print("Các đồ thị KNN (Adjacency Matrix) sẽ được tính trực tiếp bằng hàm Torch trong lúc Train giống hệt bản gốc.")

if __name__ == '__main__':
    main()
