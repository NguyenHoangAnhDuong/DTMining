import pandas as pd
import os

# Đường dẫn file input là file Excel gốc, output là thư mục muốn lưu CSV
INPUT_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\data_root\Gold Dataset (Integrated) (2).xlsx"
OUTPUT_DIR = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\data_root"
OUTPUT_NAME = "dataset_root"

# Hàm này đọc file Excel rồi lưu thành CSV, nếu có nhiều sheet thì mỗi sheet ra một file riêng
def convert_excel_to_csv(input_path: str, output_dir: str, output_name: str) -> None:
    xl = pd.ExcelFile(input_path)
    sheet_names = xl.sheet_names

    if len(sheet_names) == 1:
        df = pd.read_excel(input_path, sheet_name=sheet_names[0])
        output_path = os.path.join(output_dir, f"{output_name}.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {output_path}  ({len(df)} rows x {len(df.columns)} cols)")
    else:
        for sheet in sheet_names:
            df = pd.read_excel(input_path, sheet_name=sheet)
            # Tên sheet có thể có ký tự '/' hay '\' không hợp lệ cho tên file, thay bằng '-'
            safe_sheet = sheet.replace("/", "-").replace("\\", "-")
            output_path = os.path.join(output_dir, f"{output_name}_{safe_sheet}.csv")
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {output_path}  ({len(df)} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    convert_excel_to_csv(INPUT_PATH, OUTPUT_DIR, OUTPUT_NAME)
