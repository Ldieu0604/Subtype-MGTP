import pandas as pd

def map_proteomics():
    # Đọc danh sách Cell line gốc
    print("1. Đang đọc Cell_list.csv...")
    try:
        cell_list_df = pd.read_csv('../data/raw/Cell_list.csv')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file Cell_list.csv")
        return
        
    cell_list_df['Name'] = cell_list_df['Name'].astype(str).str.strip()

    # Đọc file từ điển Model.csv từ DepMap
    print("2. Đang đọc Model.csv...")
    try:
        model_df = pd.read_csv('../data/raw/Model.csv')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file Model.csv")
        return
        
    # Chỉ lấy 2 cột cần thiết và loại bỏ các dòng không có COSMICID
    mapping_df = model_df[['CCLEName', 'COSMICID']].dropna().copy()
    # Chuyển đổi COSMICID (có thể có đuôi .0) về số nguyên rồi thành chuỗi để map
    mapping_df['COSMICID'] = mapping_df['COSMICID'].astype(float).astype(int).astype(str).str.strip()

    # Ánh xạ Cosmic ID sang CCLEName
    print("3. Đang ánh xạ Cosmic ID sang CCLEName...")
    merged_cell_list = pd.merge(cell_list_df, mapping_df, 
                                left_on='Name', 
                                right_on='COSMICID', 
                                how='left')

    # Đọc dữ liệu Proteomics
    print("4. Đang đọc Proteomics.csv (Dữ liệu RPPA)...")
    try:
        protein_df = pd.read_csv('../data/raw/Proteomics.csv')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file Proteomics.csv.")
        return
        
    # Cột đầu tiên của file RPPA 2018 là CCLEName (vd: DMS53_LUNG)
    protein_df.rename(columns={protein_df.columns[0]: 'CCLEName'}, inplace=True)

    # Lọc và lấy dữ liệu Protein cho các Cell line của bạn
    print("5. Đang trích xuất dữ liệu protein phù hợp...")
    final_protein_data = pd.merge(merged_cell_list[['Cosmic Id', 'Name', 'CCLEName']], 
                                  protein_df, 
                                  on='CCLEName', 
                                  how='left')

    # Kiểm tra xem có bao nhiêu Cell line thiếu dữ liệu Protein
    # Cột 0 là Cosmic Id, 1 là Name, 2 là ModelID, cột thứ 3 trở đi là protein
    if final_protein_data.shape[1] > 3:
        missing_protein = final_protein_data.iloc[:, 3].isna().sum()
        total_cells = len(final_protein_data)
        print("="*40)
        print(f"Tổng số cell line của bạn: {total_cells}")
        print(f"Số cell line KHÔNG CÓ dữ liệu protein: {missing_protein}")
        print(f"Số cell line CÓ dữ liệu protein: {total_cells - missing_protein}")
        print("="*40)

    # Lưu kết quả ra file mới
    output_file = '../data/processed/Matched_Proteomics_for_GraphOmicDRP.csv'
    final_protein_data.to_csv(output_file, index=False)
    print(f"Hoàn tất! Dữ liệu đã được lưu vào {output_file}")

if __name__ == '__main__':
    map_proteomics()
