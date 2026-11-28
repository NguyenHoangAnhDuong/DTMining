"""Biểu đồ Plotly dùng chung — một theme, một bảng màu cho toàn app."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from source.services.constants import BRANCH_LABELS, COLORS, REGIME_COLORS, REGIME_LABELS, REGIMES

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def theme(fig: go.Figure, height: int = 420, **layout) -> go.Figure:
    """Theme sáng kiểu biểu đồ kinh tế: chỉ lưới ngang, trục dưới, chữ nhỏ."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Helvetica Neue, Arial, sans-serif", size=12, color=COLORS["text"]),
        margin=dict(l=4, r=8, t=36 if layout.get("title") else 12, b=4),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=COLORS["axis"], font=dict(color=COLORS["text"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        colorway=[COLORS["actual"], COLORS["base"], COLORS["kmeans"], COLORS["gold"], "#4A7C3A", "#7A4E9A"],
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=COLORS["axis"],
                     ticks="outside", tickcolor=COLORS["axis"], ticklen=4)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, showline=False,
                     title_font=dict(size=11, color=COLORS["muted"]))
    fig.update_layout(**layout)
    return fig


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, key=key)


def _range_selector(fig):
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=6, label="6T", step="month", stepmode="backward"),
                dict(count=1, label="1N", step="year", stepmode="backward"),
                dict(count=5, label="5N", step="year", stepmode="backward"),
                dict(step="all", label="Tất cả"),
            ],
            bgcolor="#F1EFE9", activecolor="#DAD6CC", font=dict(color=COLORS["text"], size=11),
            x=1, xanchor="right", y=1.02, yanchor="bottom",
        )
    )


# ── Giá + giai đoạn thị trường ─────────────────────────────────────────────
def price_by_regime(df: pd.DataFrame, height: int = 440) -> go.Figure:
    """Giá vàng tô màu theo giai đoạn (mỗi giai đoạn 1 trace WebGL → nhẹ với ~6k điểm)."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["Date"], y=df["Gold Price"], mode="lines", name="Giá vàng",
        line=dict(color="rgba(138,134,128,0.35)", width=1), hoverinfo="skip", showlegend=False,
    ))
    for r in REGIMES:
        y = df["Gold Price"].where(df["market_phase"] == r)
        fig.add_trace(go.Scattergl(
            x=df["Date"], y=y, mode="lines", name=REGIME_LABELS[r], connectgaps=False,
            line=dict(color=REGIME_COLORS[r], width=1.4),
            hovertemplate="%{y:,.1f} USD<extra>" + REGIME_LABELS[r] + "</extra>",
        ))
    theme(fig, height, yaxis_title="USD/oz")
    _range_selector(fig)
    return fig


def regime_share(df: pd.DataFrame) -> go.Figure:
    counts = df["market_phase"].value_counts().reindex(REGIMES).fillna(0)
    fig = go.Figure(go.Pie(
        labels=[REGIME_LABELS[r] for r in REGIMES], values=counts.values, hole=0.62,
        marker=dict(colors=[REGIME_COLORS[r] for r in REGIMES], line=dict(color=COLORS["bg"], width=2)),
        sort=False, textinfo="percent", hovertemplate="%{label}: %{value} phiên<extra></extra>",
    ))
    return theme(fig, 300, hovermode=False, legend=dict(orientation="v", x=1, y=0.5))


# ── Dự đoán ────────────────────────────────────────────────────────────────
def forecast_fan(history: pd.DataFrame, results: list[dict], lookback: int = 120) -> go.Figure:
    """Lịch sử giá + nón dự đoán (80%/95%) cho từng nhánh."""
    first = results[0]
    hist = history[history["Date"] <= first["date"]].tail(lookback)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Gold Price"], name="Giá lịch sử", mode="lines",
        line=dict(color=COLORS["actual"], width=1.4),
        hovertemplate="%{y:,.1f} USD<extra>Lịch sử</extra>",
    ))

    for r in results:
        color = COLORS[r["branch"]]
        x = [r["date"], r["target_date"]]
        rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        for lv, alpha in (("0.95", 0.12), ("0.8", 0.22)):
            if lv not in r["intervals"]:
                continue
            lo, hi = r["intervals"][lv]
            fig.add_trace(go.Scatter(
                x=x + x[::-1], y=[r["price_t"], hi, lo, r["price_t"]], mode="lines", fill="toself",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})", line=dict(width=0),
                name=f"{int(float(lv) * 100)}% · {BRANCH_LABELS[r['branch']]}", hoverinfo="skip",
                legendgroup=r["branch"], showlegend=len(results) == 1,
            ))
        fig.add_trace(go.Scatter(
            x=x, y=[r["price_t"], r["pred_price"]], mode="lines+markers",
            name=f"Dự đoán · {BRANCH_LABELS[r['branch']]}", legendgroup=r["branch"],
            line=dict(color=color, width=2, dash="dot"), marker=dict(size=[0, 9], color=color),
            hovertemplate="%{y:,.1f} USD<extra>Dự đoán</extra>",
        ))

    if first.get("actual") is not None:
        fig.add_trace(go.Scatter(
            x=[first["target_date"]], y=[first["actual"]], mode="markers", name="Giá thực tế",
            marker=dict(symbol="x-thin", size=12, color=COLORS["actual"], line=dict(color=COLORS["actual"], width=2.5)),
            hovertemplate="%{y:,.1f} USD<extra>Thực tế</extra>",
        ))
    fig.add_vline(x=first["date"], line=dict(color=COLORS["rule"], dash="dot", width=1))
    return theme(fig, 440, yaxis_title="USD/oz")


# ── Kết quả backtest ───────────────────────────────────────────────────────
def predictions_chart(df: pd.DataFrame, branches: list[str]) -> go.Figure:
    fig = go.Figure()
    first = df[df["branch"] == branches[0]]
    fig.add_trace(go.Scattergl(
        x=first["Date"], y=first["y_true"], name="Thực tế", mode="lines",
        line=dict(color=COLORS["actual"], width=1.3),
    ))
    for b in branches:
        d = df[df["branch"] == b]
        fig.add_trace(go.Scattergl(
            x=d["Date"], y=d["y_pred"], name=BRANCH_LABELS[b], mode="lines",
            line=dict(color=COLORS[b], width=1.3),
        ))
    theme(fig, 430, yaxis_title="USD/oz")
    _range_selector(fig)
    return fig


def cumulative_advantage(df: pd.DataFrame) -> go.Figure:
    """Σ(|e_base| − |e_kmeans|): đi lên = K-Means sai số nhỏ hơn."""
    wide = df.pivot_table(index="Date", columns="branch", values="abs_err")
    adv = (wide["base"] - wide["kmeans"]).cumsum()
    fig = go.Figure(go.Scattergl(
        x=adv.index, y=adv.values, mode="lines", fill="tozeroy",
        line=dict(color=COLORS["kmeans"], width=1.5), fillcolor="rgba(192,86,43,0.10)",
        hovertemplate="%{y:,.1f} USD<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=COLORS["rule"], width=1))
    return theme(fig, 300, yaxis_title="USD tích lũy")


def grouped_bars(df: pd.DataFrame, x: str, cols: dict, height: int = 340, **layout) -> go.Figure:
    """cols = {tên cột: (nhãn, màu)}."""
    fig = go.Figure()
    for col, (label, color) in cols.items():
        fig.add_trace(go.Bar(
            x=df[x].astype(str), y=df[col], name=label, marker_color=color,
            hovertemplate="%{y:,.3f}<extra>" + label + "</extra>",
        ))
    return theme(fig, height, barmode="group", bargap=0.35, bargroupgap=0.05, **layout)


def delta_bars(df: pd.DataFrame, x: str, y: str, height: int = 300, **layout) -> go.Figure:
    """Cột Δ%: xanh = K-Means tốt hơn (âm), đỏ = kém hơn."""
    colors = [COLORS["good"] if v < 0 else COLORS["bad"] for v in df[y]]
    fig = go.Figure(go.Bar(
        x=df[x].astype(str), y=df[y], marker_color=colors,
        text=[f"{v:+.2f}%" for v in df[y]], textposition="outside", textfont=dict(family="Source Code Pro, monospace", size=11),
        hovertemplate="%{y:+.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=COLORS["rule"], width=1))
    return theme(fig, height, hovermode=False, **layout)


def feature_importance_chart(df: pd.DataFrame, top: int = 20) -> go.Figure:
    d = df.nlargest(top, "importance_gain").iloc[::-1]
    regime = d["feature"].str.contains("kmeans_|regime_id")
    fig = go.Figure(go.Bar(
        x=d["importance_gain"], y=d["feature"].str.replace("f_", "", regex=False), orientation="h",
        marker_color=[COLORS["kmeans"] if r else COLORS["base"] for r in regime],
        hovertemplate="%{x:.4f}<extra></extra>",
    ))
    return theme(fig, max(360, 22 * len(d)), hovermode=False, xaxis_title="Gain (chuẩn hoá)")


# ── EDA / phân cụm ─────────────────────────────────────────────────────────
def multi_line(df: pd.DataFrame, cols: list[str], normalize: bool) -> go.Figure:
    fig = go.Figure()
    for c in cols:
        s = df[c]
        if normalize:
            s = s / s.dropna().iloc[0] * 100
        fig.add_trace(go.Scattergl(x=df["Date"], y=s, name=c, mode="lines", line=dict(width=1.4)))
    theme(fig, 440, yaxis_title="Chỉ số (kỳ đầu = 100)" if normalize else "Giá trị")
    _range_selector(fig)
    return fig


def heatmap(corr: pd.DataFrame, height: int = 560) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0, "#2F5D8C"], [0.5, "#FFFFFF"], [1, "#9A6B12"]],
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    return theme(fig, height, hovermode=False, yaxis=dict(autorange="reversed"))


def histogram(s: pd.Series, name: str, color: str = COLORS["gold"]) -> go.Figure:
    fig = go.Figure(go.Histogram(x=s, nbinsx=120, marker_color=color, marker_line=dict(width=0), name=name))
    return theme(fig, 320, hovermode=False, bargap=0.02, xaxis_title=name)


def k_scan(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["k"], y=df["silhouette"], name="Silhouette ↑", mode="lines+markers",
                             line=dict(color=COLORS["gold"])))
    fig.add_trace(go.Scatter(x=df["k"], y=df["davies_bouldin"], name="Davies-Bouldin ↓", mode="lines+markers",
                             line=dict(color=COLORS["base"]), yaxis="y2"))
    fig.add_vline(x=4, line=dict(color=COLORS["rule"], dash="dot"), annotation_text="k = 4")
    return theme(fig, 320, xaxis_title="k", yaxis=dict(title="Silhouette"),
                 yaxis2=dict(title="Davies-Bouldin", overlaying="y", side="right", showgrid=False))
