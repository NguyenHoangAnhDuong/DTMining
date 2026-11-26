# Bước 3b — xử lý outlier bằng IQR Capping (Winsorization)
# Thay vì xóa hàng, giá trị vượt ngưỡng sẽ bị kéo về đúng ngưỡng đó
# Ngưỡng tính theo: Q1 - 1.5*IQR (dưới) và Q3 + 1.5*IQR (trên)
# Chỉ áp dụng cho cột float64, bỏ qua cột nhị phân 0/1
# Lưu ý: giá dầu âm tháng 4/2020 là hợp lệ và đã được giữ nguyên từ bước trước

import os

import pandas as pd

INPUT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_missing.csv"
OUTPUT_PATH = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_outlier.csv"


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5):
    # Tính ngưỡng dưới và ngưỡng trên dựa trên IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return lower, upper


def cap_outliers(df: pd.DataFrame, columns: list[str], multiplier: float = 1.5):
    df = df.copy()
    report = []

    for col in columns:
        lower, upper = detect_outliers_iqr(df[col], multiplier)
        n_lower = (df[col] < lower).sum()
        n_upper = (df[col] > upper).sum()
        n_total = n_lower + n_upper

        # clip kéo giá trị về đúng ngưỡng, không xóa hàng
        df[col] = df[col].clip(lower=lower, upper=upper)

        if n_total > 0:
            report.append({
                "column"        : col,
                "lower_bound"   : round(lower, 4),
                "upper_bound"   : round(upper, 4),
                "outliers_below": int(n_lower),
                "outliers_above": int(n_upper),
                "total_capped"  : int(n_total),
            })

    return df, pd.DataFrame(report)


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Doc du lieu: {df.shape[0]} hang x {df.shape[1]} cot")

    # Chỉ lấy cột float, cột nhị phân 0/1 không cần xử lý outlier
    float_cols = df.select_dtypes(include="float64").columns.tolist()
    print(f"So cot float can xu ly: {len(float_cols)}")

    df_out, report = cap_outliers(df, float_cols, multiplier=1.5)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Luu tai: {OUTPUT_PATH}")
    print(f"Kich thuoc dau ra: {df_out.shape}")

    if report.empty:
        print("Khong phat hien ngoai lai trong bat ky cot nao.")
    else:
        print(f"\nBao cao cap ngoai lai ({len(report)} cot bi anh huong):")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
