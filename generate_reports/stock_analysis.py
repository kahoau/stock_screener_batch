import yfinance as yf
import matplotlib.pyplot as plt
import http.client
import json
import asyncio
import os
import requests
import yaml
from telegram import Bot
from telegram.constants import ParseMode
from datetime import datetime
import numpy as np


# ------------------------------------------------------------------------------
# ✅ Final Fix: 100% Windows-Native Font Configuration (No Office Dependencies)
# ------------------------------------------------------------------------------
def setup_matplotlib_font():
    """
    Configure Matplotlib to use ONLY built-in Windows Chinese fonts (SimHei/Microsoft YaHei)
    and fully suppress all font-related warnings (including findfont messages)
    """
    import matplotlib
    import warnings

    # 1. Suppress ALL font-related warnings (findfont + Glyph missing)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # Only built-in fonts
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

    # 2. Hardcode SimHei font path (GUARANTEED on Windows) to bypass font lookup
    try:
        from matplotlib.font_manager import FontProperties
        simhei_path = "C:/Windows/Fonts/simhei.ttf"
        if os.path.exists(simhei_path):
            font_prop = FontProperties(fname=simhei_path)
            matplotlib.rcParams['font.family'] = font_prop.get_name()
    except Exception:
        pass  # Fallback to font list if path lookup fails


# ------------------------------------------------------------------------------
# ✅ Core Initialization (Option 2: Market → Category → Date Folder Structure)
# ------------------------------------------------------------------------------
def init():
    """Initialize config with market → category → date folder structure"""
    # First setup font to eliminate ALL font warnings
    setup_matplotlib_font()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Core variables
    market_type = config["MARKET_TYPE"]  # HK/US
    today = datetime.now().strftime("%Y%m%d")
    base_output_dir = config["BASE_OUTPUT_DIR"]  # Root: ./stock_analysis_report

    # Category mapping (folder names match your requirement: holdings/strong_trend/market/watch)
    category_mapping = {
        "market": {"cn_name": "市场指数", "folder_name": "market"},  # Exact folder name
        "strong_trend": {"cn_name": "强势趋势股", "folder_name": "strong_trend"},
        "watch": {"cn_name": "观察列表", "folder_name": "watch"},
        "holding": {"cn_name": "持仓列表", "folder_name": "holdings"}  # Renamed to "holdings" (match your req)
    }

    # Load stock lists (HK/US)
    if market_type == "HK":
        stock_lists = {
            "market": config["HK_STOCK_MARKET_LIST"],
            "strong_trend": config["HK_STOCK_STRONG_TREND_LIST"],
            "watch": config["HK_STOCK_WATCH_LIST"],
            "holding": config["HK_STOCK_HOLDING_LIST"]
        }
        market_cn = "港股"
        market_folder = "HK"  # Exact market folder name (uppercase)
    else:
        stock_lists = {
            "market": config["US_STOCK_MARKET_LIST"],
            "strong_trend": config["US_STOCK_STRONG_TREND_LIST"],
            "watch": config["US_STOCK_WATCH_LIST"],
            "holding": config["US_STOCK_HOLDING_LIST"]
        }
        market_cn = "美股"
        market_folder = "US"

    # ✅ Create Option 2 Folder Structure: BASE → MARKET → CATEGORY → DATE → img
    category_folders = {}
    for cat, cat_info in category_mapping.items():
        # Step 1: Market + Category folder (e.g., ./stock_analysis_report/HK/holdings)
        category_root = os.path.join(base_output_dir, market_folder, cat_info["folder_name"])
        # Step 2: Date subfolder (e.g., ./stock_analysis_report/HK/holdings/20260216)
        date_folder = os.path.join(category_root, today)
        # Step 3: Image folder inside date folder (e.g., ./stock_analysis_report/HK/holdings/20260216/img)
        img_folder = os.path.join(date_folder, "img")

        # Create all folders (exist_ok=True to avoid errors if folders already exist)
        for folder in [category_root, date_folder, img_folder]:
            os.makedirs(folder, exist_ok=True)

        category_folders[cat] = {
            "category_root": category_root,  # e.g., ./stock_analysis_report/HK/holdings
            "report_folder": date_folder,  # e.g., ./stock_analysis_report/HK/holdings/20260216 (where MD is saved)
            "img_folder": img_folder,  # e.g., ./stock_analysis_report/HK/holdings/20260216/img
            "cn_name": cat_info["cn_name"],
            "folder_name": cat_info["folder_name"]
        }

    # Color config (unchanged)
    if market_type == "HK":
        up_color = config["HK_COLOR_CONFIG"]["up_color"]
        down_color = config["HK_COLOR_CONFIG"]["down_color"]
        ma_colors = config["HK_COLOR_CONFIG"]["ma_colors"]
    else:
        up_color = "#00B300"
        down_color = "#FF3333"
        ma_colors = {5: "#FF9900", 10: "#FF33CC", 20: "#0066FF", 60: "#9933FF"}

    # Final config dict
    init_config = {
        "market_type": market_type,
        "market_cn": market_cn,
        "market_folder": market_folder,  # HK/US (folder name)
        "today": today,
        "base_output_dir": base_output_dir,
        "category_mapping": category_mapping,
        "category_folders": category_folders,
        "stock_lists": stock_lists,
        "up_color": up_color,
        "down_color": down_color,
        "ma_colors": ma_colors,
        "current_price_color": "#FFCC00",
        "accum_color": "#9900CC",
        "accum_threshold_color": "#0099FF",
        "period": config["PERIOD"],
        "interval": config["INTERVAL"],
        "ma_periods": config["MA_PERIODS"],
        "support_resistance_lookback": config["SUPPORT_RESISTANCE_LOOKBACK"],
        "rsi_period": config["RSI_PERIOD"],
        "rsi_overbought": config["RSI_OVERBOUGHT"],
        "rsi_oversold": config["RSI_OVERSOLD"],
        "doubao_api_key": config["DOUBAO_API_KEY"],
        "api_host": config["API_HOST"],
        "api_path": config["API_PATH"],
        "model": config["MODEL"],
        "tencent_api_headers": config["TENCENT_API_HEADERS"],
        "telegram_bot_token": config["TELEGRAM_BOT_TOKEN"],
        "telegram_chat_id": config["TELEGRAM_CHAT_ID"]
    }

    return init_config


# ------------------------------------------------------------------------------
# Data Fetching (Unchanged)
# ------------------------------------------------------------------------------
def get_data(stock, config):
    df = yf.Ticker(stock).history(
        period=config["period"],
        interval=config["interval"],
        auto_adjust=False
    )
    return df


# ------------------------------------------------------------------------------
# Main Force Accumulation (Unchanged)
# ------------------------------------------------------------------------------
def add_main_force_accumulation(df):
    df = df.copy()
    df["HL"] = df["High"] - df["Low"]
    df["HL"] = df["HL"].replace(0, 0.001)
    df["mf_raw"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / df["HL"] * df["Volume"]
    df["mf_ma10"] = df["mf_raw"].abs().rolling(window=10).mean()
    df["strength"] = df["mf_raw"] / df["mf_ma10"] * 8
    df["MainForceAccum"] = np.where(
        (df["Close"] < df["Open"]) & (df["strength"] > 0.2) | (df["strength"] > 0.4),
        df["strength"],
        0
    )
    df["MainForceAccum"] = df["MainForceAccum"].clip(0, 100)
    last3 = df.tail(3).index
    df.loc[last3, "MainForceAccum"] *= 0.7
    df = df.drop(columns=["HL", "mf_raw", "mf_ma10", "strength"])
    df["MainForceAccum"] = df["MainForceAccum"].fillna(0)
    return df


# ------------------------------------------------------------------------------
# Data Cleaning (Unchanged)
# ------------------------------------------------------------------------------
def clean_data(df):
    if df.empty:
        return df
    keep = ["Open", "High", "Low", "Close", "Volume", "MainForceAccum"]
    df = df.copy()[keep].dropna()
    df = df[(df["Volume"] > 0) & (df["High"] > df["Low"])]
    return df


# ------------------------------------------------------------------------------
# MA & RSI (Unchanged)
# ------------------------------------------------------------------------------
def add_indicators(df, config):
    if df.empty:
        return df
    df = df.copy()
    for n in config["ma_periods"]:
        df[f"MA{n}"] = df["Close"].rolling(window=n).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=config["rsi_period"]).mean()
    avg_loss = loss.rolling(window=config["rsi_period"]).mean()
    avg_loss = avg_loss.replace(0, 0.0001)
    df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))
    return df


# ------------------------------------------------------------------------------
# Support/Resistance (Unchanged)
# ------------------------------------------------------------------------------
def calculate_swing_support_resistance(df, config):
    recent = df.tail(config["support_resistance_lookback"])
    c = recent["Close"].iloc[-1]
    h = recent["High"].mean()
    l = recent["Low"].mean()
    close_avg = recent["Close"].mean()
    pp = (h + l + close_avg) / 3
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    res = [x for x in [r1, r2] if c < x <= c * 1.05]
    sup = [x for x in [s1, s2] if c * 0.95 <= x < c]
    resistance = min(res) if res else c * 1.03
    support = max(sup) if sup else c * 0.97
    return round(support, 2), round(resistance, 2)


# ------------------------------------------------------------------------------
# Trend Analysis (Unchanged)
# ------------------------------------------------------------------------------
def get_trend(df, config):
    sub = df.tail(50)
    c = sub["Close"].iloc[-1]
    ma20 = sub["MA20"].iloc[-1]
    ma60 = sub["MA60"].iloc[-1]
    if c > ma20 > ma60:
        return "上升趋势"
    elif c < ma20 < ma60:
        return "下降趋势"
    else:
        return "横盘整理"


def rsi_status(v, config):
    if v >= config["rsi_overbought"]:
        return "超买"
    elif v <= config["rsi_oversold"]:
        return "超卖"
    else:
        return "中性"


# ------------------------------------------------------------------------------
# ✅ FIXED: get_ai_analysis Function (Previously Missing)
# ------------------------------------------------------------------------------
def get_ai_analysis(stock, trend, close, ma20, ma60, rsi, rsi_text, support, resistance, config):
    """
    Call AI API to generate stock analysis (Cantonese, short-term 1-5 days)
    """
    market = config["market_cn"]
    prompt = f"""
你是專業{market}波段交易員，只用廣東話分析，專注1-5日超短線。
股票：{stock}
現價：{close:.2f}
趨勢：{trend}
MA20：{ma20:.2f} MA60：{ma60:.2f}
RSI：{rsi:.1f} ({rsi_text})
支撐：{support:.2f} 阻力：{resistance:.2f}

格式：
1. 趨勢總結
2. 勝率
3. 情景分析
4. 交易建議
"""
    headers = {"Authorization": f"Bearer {config['doubao_api_key']}", "Content-Type": "application/json"}
    payload = json.dumps({
        "model": config["model"], "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}]
    })
    try:
        conn = http.client.HTTPSConnection(config["api_host"])
        conn.request("POST", config["api_path"], payload, headers)
        resp = json.loads(conn.getresponse().read().decode())
        conn.close()
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ AI分析失敗 for {stock}: {str(e)}")
        return f"AI分析失敗（請檢查API配置）: {str(e)}"


# ------------------------------------------------------------------------------
# Plotting (Save to Option 2 Structure: DATE/img folder)
# ------------------------------------------------------------------------------
def plot_stock_chart(stock, df, support, resistance, config, category):
    plot_df = df.tail(90).reset_index(drop=True)
    current = plot_df["Close"].iloc[-1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                        gridspec_kw={"height_ratios": [3, 1, 1.5]})
    width = 0.6

    # K-line plotting
    for idx, row in plot_df.iterrows():
        o = row["Open"]
        h = row["High"]
        l = row["Low"]
        c = row["Close"]
        is_up = c >= o
        color = config["up_color"] if is_up else config["down_color"]
        ax1.plot([idx, idx], [l, h], color=color, linewidth=1.0)
        ax1.bar(idx, abs(c - o), bottom=min(o, c), color=color, width=width)

    # Moving Averages
    for n in config["ma_periods"]:
        ax1.plot(plot_df.index, plot_df[f"MA{n}"], color=config["ma_colors"][n], label=f"MA{n}", linewidth=1.5)

    # Support/Resistance/Current Price
    ax1.axhline(support, color=config["up_color"], linestyle="--", linewidth=2)
    ax1.axhline(resistance, color=config["down_color"], linestyle="--", linewidth=2)
    ax1.axhline(current, color=config["current_price_color"], linestyle=":", linewidth=3, alpha=0.8)

    # Annotations (Chinese text now renders correctly)
    ax1.text(0.02, 1.02, f"{stock}  日线图",
             fontsize=14, fontweight='bold',
             ha='left', va='bottom', transform=ax1.transAxes)
    ax1.text(0.98, 0.95, f"现价: {current:.2f}",
             color=config["current_price_color"], fontsize=11, fontweight="bold",
             va="top", ha="right", transform=ax1.transAxes)
    ax1.text(0.98, 0.90, f"支撑位: {support:.2f}",
             color=config["up_color"], fontsize=11, fontweight="bold",
             va="top", ha="right", transform=ax1.transAxes)
    ax1.text(0.98, 0.85, f"阻力位: {resistance:.2f}",
             color=config["down_color"], fontsize=11, fontweight="bold",
             va="top", ha="right", transform=ax1.transAxes)

    # Adaptive Y-axis
    price_min = min(plot_df["Low"].min(), support)
    price_max = max(plot_df["High"].max(), resistance)
    price_range = price_max - price_min
    if price_range < 50:
        y_step = 10
    elif price_range < 200:
        y_step = 50
    elif price_range < 1000:
        y_step = 100
    else:
        y_step = 200
    y_min = int(price_min // y_step * y_step - y_step)
    y_max = int(price_max // y_step * y_step + y_step)
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(range(y_min, y_max + 1, y_step))
    ax1.set_xticks([])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#F5F5F5")

    # Volume Plot
    up = plot_df[plot_df.Close >= plot_df.Open]
    dn = plot_df[plot_df.Close < plot_df.Open]
    ax2.bar(up.index, up.Volume * 0.15, color=config["up_color"], alpha=0.6)
    ax2.bar(dn.index, dn.Volume * 0.15, color=config["down_color"], alpha=0.6)
    ax2.text(0.02, 1.02, "成交量",
             fontsize=12, fontweight='bold',
             ha='left', va='bottom', transform=ax2.transAxes)
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.set_xticks([])
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#F5F5F5")

    # Main Force Accumulation
    ax3.set_ylim(0, 100)
    ax3.axhline(100, color=config["accum_threshold_color"], linewidth=2)
    for idx, val in zip(plot_df.index, plot_df["MainForceAccum"]):
        if val > 0:
            ax3.bar(idx, val, color=config["accum_color"], alpha=0.8, width=0.4)
    ax3.text(0.02, 1.02, "主力吸筹",
             fontsize=12, fontweight='bold',
             ha='left', va='bottom', transform=ax3.transAxes)
    ax3.set_xticks([])
    ax3.grid(alpha=0.3)
    ax3.set_facecolor("#F5F5F5")

    # ✅ Save to Option 2 Structure: DATE/img folder (e.g., ./stock_analysis_report/HK/holdings/20260216/img/0005.HK_chart_20260216.png)
    img_folder = config["category_folders"][category]["img_folder"]
    chart_filename = f"{stock}_chart_{config['today']}.png"
    path = os.path.join(img_folder, chart_filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F5F5F5")
    plt.close()
    return path


# ------------------------------------------------------------------------------
# Generate Report (Save to Option 2 Structure: DATE folder)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ✅ 圖片放最前，內容放後面 (Image first, then content)
# ------------------------------------------------------------------------------
def generate_category_md(reports, config, category):
    """Generate report saved to DATE subfolder (Option 2 structure)"""
    category_info = config["category_folders"][category]
    category_cn = category_info["cn_name"]
    date_folder = category_info["report_folder"]
    market_cn = config["market_cn"]
    today = config["today"]

    md = [
        f"# {market_cn} - {category_cn} 分析报告",
        f"**生成时间**: {today}",
        f"**分析股票数量**: {len(reports)}",
        "\n---\n"
    ]

    for r in reports:
        if not r:
            continue
        img_filename = os.path.basename(r["image"])
        img_relative_path = f"img/{img_filename}"

        # 👇 重點：先放圖，再放內容
        md.extend([
            f"## {r['stock']}",
            f"![{r['stock']} 日线图]({img_relative_path})",  # 圖放最頂
            "",
            f"- 最新價格: {r['close']:.2f}",
            f"- 技術趨勢: {r['trend']}",
            f"- MA20: {r['ma20']:.2f} | MA60: {r['ma60']:.2f}",
            f"- RSI({config['rsi_period']}): {r['rsi']:.1f} ({r['rsi_text']})",
            f"- 支撑位: {r['support']} | 阻力位: {r['resistance']}",
            "\n### 📝 AI超短線分析",
            r['ai'],
            "\n---\n"
        ])

    report_filename = f"{config['market_folder'].lower()}_{category_info['folder_name']}_report_{today}.md"
    report_path = os.path.join(date_folder, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"✅ 生成{category_cn}报告: {report_path}")
    return report_path


# ------------------------------------------------------------------------------
# Analyze Category (Unchanged Logic)
# ------------------------------------------------------------------------------
def analyze_category(category, config):
    category_cn = config["category_folders"][category]["cn_name"]
    stocks = config["stock_lists"][category]
    print(f"\n📈 開始分析 {category_cn} (共{len(stocks)}只股票): {', '.join(stocks)}")
    reports = []

    for stock in stocks:
        print(f"  📊 分析 {stock}...")
        try:
            df = get_data(stock, config)
            if len(df) < 50:
                print(f"  ⚠️ {stock} 数据不足，跳过")
                continue

            df = add_main_force_accumulation(df)
            df = clean_data(df)
            df = add_indicators(df, config)
            sup, res = calculate_swing_support_resistance(df, config)
            trend = get_trend(df, config)
            latest = df.iloc[-1]
            rsi_val = latest["RSI"]
            rsi_txt = rsi_status(rsi_val, config)
            img_path = plot_stock_chart(stock, df, sup, res, config, category)
            # ✅ Call get_ai_analysis (now defined)
            ai = get_ai_analysis(stock, trend, latest.Close, latest.MA20, latest.MA60,
                                 rsi_val, rsi_txt, sup, res, config)

            reports.append({
                "stock": stock,
                "close": latest.Close,
                "ma20": latest.MA20,
                "ma60": latest.MA60,
                "rsi": rsi_val,
                "rsi_text": rsi_txt,
                "trend": trend,
                "support": sup,
                "resistance": res,
                "image": img_path,
                "ai": ai
            })
            print(f"  ✅ {stock} 分析完成")

        except Exception as e:
            print(f"  ❌ 分析 {stock} 出错: {str(e)}")
            continue

    return reports


# ------------------------------------------------------------------------------
# Telegram (Commented Out)
# ------------------------------------------------------------------------------
# async def send_telegram(reports, config):
#     pass
# def escape_markdown_v2(t):
#     pass

# ------------------------------------------------------------------------------
# Main Program (Unchanged Logic)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    config = init()
    market_cn = config["market_cn"]
    today = config["today"]
    base_output_dir = config["base_output_dir"]

    print(f"🚀 開始{market_cn}全類別分析 ({today})")
    print(f"📁 輸出目錄: {base_output_dir}")

    all_categories = ["market", "strong_trend", "watch", "holding"]
    for category in all_categories:
        category_reports = analyze_category(category, config)
        if category_reports:
            generate_category_md(category_reports, config, category)
        else:
            category_cn = config["category_folders"][category]["cn_name"]
            print(f"⚠️ {category_cn} 無有效分析結果")

    print(f"\n✅ 所有{market_cn}類別分析完成！")
    print(f"📄 報告位置: {base_output_dir}/{config['market_folder']}")