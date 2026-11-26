# Bước 5 — kiểm định tính dừng và chuyển đổi chuỗi thời gian
# Quy trình:
#   1. Kiểm định ADF + KPSS trên chuỗi gốc xem có dừng không
#   2. Chuyển đổi chuỗi không dừng:
#        - Cột giá (price)  -> log-return  : ln(P_t / P_{t-1})
#        - Cột lãi suất     -> sai phân b1 : P_t - P_{t-1}
#   3. Kiểm định lại sau chuyển đổi để xác nhận đã dừng
#   4. Vẽ ACF/PACF (Gold Price log-return) để xác định bậc p, q cho ARIMA
#   5. Phân rã STL (Gold Price gốc) để thấy xu hướng + mùa vụ + phần dư
# Kết quả ra 3 loại file: CSV dữ liệu, CSV báo cáo kiểm định, và ảnh biểu đồ
# Lưu ý: Fedral Funds Rate_diff vẫn còn xu hướng sau sai phân vì có cấu trúc break
#         ở giai đoạn lãi suất bằng 0 (zero lower bound) — chấp nhận, không xử lý thêm

import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

INPUT_PATH  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_time_series.csv"
OUTPUT_CSV  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\processed\after_stationarity.csv"
REPORT_CSV  = r"D:\University\Data Mining\Project_DTMining\DTMining\dataset\reports\stationarity_report.csv"
FIG_DIR     = r"D:\University\Data Mining\Project_DTMining\DTMining\figures\stationarity"

# Cột giá dùng log-return vì tất cả đều dương sau khi xử lý outlier
PRICE_COLS = [
    "Gold Price", "Oil", "S&P 500", "DJI", "U.S Treasury",
    "U.S dollar index", "CPI Rate", "Silver", "Copper",
    "Iron Price", "Platinum", "Palladium",
]

# Cột lãi suất dùng sai phân bậc 1 vì có thể âm nên không dùng được log
RATE_COLS = ["Inflation Rate", "Fedral Funds Rate"]

# Cột mục tiêu dự báo — bỏ qua khi kiểm định, không tạo thêm cột transform
TARGET_SKIP = {"Tomorrow gold price", "Next week gold price", "Next Month gold price"}

# Chu kỳ STL = ~252 ngày giao dịch mỗi năm
STL_PERIOD = 252


def _test_one(series: pd.Series, col_name: str) -> dict:
    # Chạy ADF và KPSS trên một chuỗi, trả về kết quả dạng dict
    # ADF: H0 là có đơn vị gốc (không dừng), bác bỏ H0 khi p < 0.05 => dừng
    # KPSS: H0 là dừng, bác bỏ H0 khi p < 0.05 => không dừng
    s = series.dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, _, _            = kpss(s, regression="c", nlags="auto")

    adf_reject  = adf_p  < 0.05
    kpss_reject = kpss_p < 0.05

    if adf_reject and not kpss_reject:
        ket_luan = "Dung"
    elif adf_reject and kpss_reject:
        ket_luan = "Dung theo xu huong"
    else:
        ket_luan = "Khong dung"

    return {
        "Cot"      : col_name,
        "ADF_stat" : round(adf_stat,  4),
        "ADF_p"    : round(adf_p,     4),
        "ADF_1pct" : round(adf_crit["1%"], 4),
        "KPSS_stat": round(kpss_stat, 4),
        "KPSS_p"   : round(kpss_p,    4),
        "Ket_luan" : ket_luan,
    }


def scan_stationarity(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    # Kiểm định ADF + KPSS cho danh sách cột, in bảng kết quả ra log
    rows = [_test_one(df[c], c) for c in cols if c in df.columns]
    report = pd.DataFrame(rows)
    logger.info("\n=== %s ===\n%s", label, report.to_string(index=False))
    return report


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Thêm cột {col}_logret = ln(P_t / P_{t-1}), hàng đầu tiên sẽ là NaN vì không có P_{t-1}
    for col in PRICE_COLS:
        if col not in df.columns:
            continue
        new_col = col + "_logret"
        df[new_col] = np.log(df[col] / df[col].shift(1))
        logger.info("Them cot %-35s (log-return)", f"'{new_col}'")
    return df


def add_first_diffs(df: pd.DataFrame) -> pd.DataFrame:
    # Thêm cột {col}_diff = P_t - P_{t-1}, hàng đầu tiên cũng NaN
    for col in RATE_COLS:
        if col not in df.columns:
            continue
        new_col = col + "_diff"
        df[new_col] = df[col].diff()
        logger.info("Them cot %-35s (sai phan bac 1)", f"'{new_col}'")
    return df


def plot_acf_pacf(series: pd.Series, title: str, save_path: str, lags: int = 50) -> None:
    # Vẽ ACF và PACF để xác định bậc p, q cho ARIMA
    # ACF: nhìn vào lag cuối có ý nghĩa thống kê -> bậc MA(q)
    # PACF: nhìn vào lag cuối có ý nghĩa thống kê -> bậc AR(p)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    plot_acf(series.dropna(),  lags=lags, ax=ax1, alpha=0.05, zero=False)
    ax1.set_title("ACF (xac dinh bac MA q)", fontsize=11)
    ax1.axhline(0, color="black", linewidth=0.5)

    plot_pacf(series.dropna(), lags=lags, ax=ax2, alpha=0.05, method="ywm", zero=False)
    ax2.set_title("PACF (xac dinh bac AR p)", fontsize=11)
    ax2.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Luu ACF/PACF: %s", save_path)


def plot_stl(series: pd.Series, period: int, title: str, save_path: str) -> None:
    # Phân rã STL tách chuỗi thành 3 phần: xu hướng + mùa vụ + phần dư
    # robust=True giảm ảnh hưởng của outlier lên trend
    stl = STL(series.dropna(), period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(series.index, series.values, linewidth=0.7, color="#2176ae")
    axes[0].set_ylabel("Goc (Original)", fontsize=9)

    axes[1].plot(series.index, res.trend, linewidth=1.0, color="#e63946")
    axes[1].set_ylabel("Xu huong (Trend)", fontsize=9)

    axes[2].plot(series.index, res.seasonal, linewidth=0.7, color="#2a9d8f")
    axes[2].set_ylabel("Mua vu (Seasonal)", fontsize=9)

    axes[3].plot(series.index, res.resid, linewidth=0.7, color="#6c757d")
    axes[3].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[3].set_ylabel("Phan du (Residual)", fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Luu STL: %s", save_path)


def process_stationarity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Trả về (df_out, report_combined)
    # df_out: data gốc + thêm các cột _logret và _diff
    # report_combined: bảng kết quả kiểm định trước và sau chuyển đổi ghép lại
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Lấy danh sách cột cần kiểm định, bỏ qua target và cột cờ nhị phân
    all_test_cols = [c for c in PRICE_COLS + RATE_COLS if c in df.columns]

    logger.info("Kiem dinh tinh dung tren chuoi GOC...")
    report_before = scan_stationarity(df, all_test_cols, "Ket qua tren chuoi GOC")
    report_before["Giai_doan"] = "Truoc chuyen doi"

    df = add_log_returns(df)
    df = add_first_diffs(df)

    transformed_cols = (
        [c + "_logret" for c in PRICE_COLS if c in df.columns] +
        [c + "_diff"   for c in RATE_COLS  if c in df.columns]
    )
    logger.info("Kiem dinh tinh dung tren chuoi DA CHUYEN DOI...")
    report_after = scan_stationarity(df, transformed_cols, "Ket qua tren chuoi DA CHUYEN DOI")
    report_after["Giai_doan"] = "Sau chuyen doi"

    report_combined = pd.concat([report_before, report_after], ignore_index=True)

    os.makedirs(FIG_DIR, exist_ok=True)

    gold_logret = df["Gold Price_logret"].dropna()
    gold_logret.index = df.loc[gold_logret.index, "Date"]
    plot_acf_pacf(
        gold_logret,
        title="ACF / PACF -- Gold Price Log-Return",
        save_path=os.path.join(FIG_DIR, "acf_pacf_gold_logret.png"),
    )

    gold_orig = df["Gold Price"].copy()
    gold_orig.index = pd.to_datetime(df["Date"])
    plot_stl(
        gold_orig,
        period=STL_PERIOD,
        title="Phan ra STL -- Gold Price (goc)",
        save_path=os.path.join(FIG_DIR, "stl_gold_price.png"),
    )

    return df, report_combined


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = pd.read_csv(INPUT_PATH)
    logger.info("Doc du lieu: %d dong x %d cot", *df.shape)

    df_out, report = process_stationarity(df)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Luu du lieu: %s", OUTPUT_CSV)

    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    logger.info("Luu bao cao: %s", REPORT_CSV)

    new_cols = [c for c in df_out.columns if c.endswith("_logret") or c.endswith("_diff")]
    after_rows = report[report["Giai_doan"] == "Sau chuyen doi"]
    n_stationary = (after_rows["Ket_luan"] == "Dung").sum()

    print("\n=== Tom tat Buoc 5 ===")
    print(f"  So hang            : {len(df_out)}")
    print(f"  So cot goc         : {len(df.columns)}")
    print(f"  So cot them moi    : {len(new_cols)}")
    print(f"  Tong so cot        : {df_out.shape[1]}")
    print(f"  Cot dung sau TT    : {n_stationary} / {len(after_rows)}")
    print(f"  Bieu do luu tai    : {FIG_DIR}")
    print(f"  Bao cao luu tai    : {REPORT_CSV}")
    print(f"  Du lieu luu tai    : {OUTPUT_CSV}")
    print()
    print("Cac cot them moi (_logret / _diff):")
    for c in new_cols:
        nan_count = df_out[c].isna().sum()
        print(f"  {c:<40} NaN={nan_count} (hang dau tien)")


if __name__ == "__main__":
    main()
