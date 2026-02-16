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
# ✅ 新增：音频生成相关库
import pyttsx3
import edge_tts
import aiofiles


# ------------------------------------------------------------------------------
# ✅ NEW: Audio Generation Functions (Edge TTS - Better Quality than Google)
# ------------------------------------------------------------------------------
async def generate_audio(text, lang, output_path):
    """
    Generate audio using Edge TTS (better quality than Google TTS)
    Supported langs:
    - Cantonese: zh-HK (Hong Kong Cantonese)
    - Mandarin: zh-CN (Mainland Mandarin)
    - English: en-US (American English)
    """
    # ✅ 修正：使用 Microsoft 官方有效聲源
    voice_map = {
        "zh-HK": "zh-HK-WanLungNeural",  # 香港粵語（正確）
        "zh-CN": "zh-CN-XiaoxiaoNeural",  # 普通話
        "en-US": "en-US-AriaNeural"  # 英語
    }

    try:
        communicate = edge_tts.Communicate(text, voice_map[lang])
        await communicate.save(output_path)
        print(f"✅ Audio generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Audio generation failed for {lang}: {str(e)}")
        return None


def clean_text_for_tts(text):
    """Clean special characters from text to avoid TTS errors"""
    # Remove markdown symbols and special chars
    clean_text = text.replace("#", "").replace("*", "").replace("`", "").replace("---", "")
    clean_text = clean_text.replace("📊", "").replace("📝", "").replace("：", ":").replace("｜", "|")
    # Simplify line breaks for better TTS flow
    clean_text = clean_text.replace("\n\n", ". ").replace("\n", ". ")
    return clean_text


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
        "market": {"cn_name": "市场指数", "en_name": "Market Index"},
        "strong_trend": {"cn_name": "强势趋势股", "en_name": "Strong Trend Stocks"},
        "watch": {"cn_name": "观察列表", "en_name": "Watch List"},
        "holding": {"cn_name": "持仓列表", "en_name": "Holding List"}
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
        market_en = "Hong Kong Stocks"
        market_folder = "HK"  # Exact market folder name (uppercase)
    else:
        stock_lists = {
            "market": config["US_STOCK_MARKET_LIST"],
            "strong_trend": config["US_STOCK_STRONG_TREND_LIST"],
            "watch": config["US_STOCK_WATCH_LIST"],
            "holding": config["US_STOCK_HOLDING_LIST"]
        }
        market_cn = "美股"
        market_en = "US Stocks"
        market_folder = "US"

    # ✅ Create Option 2 Folder Structure: BASE → MARKET → CATEGORY → DATE → img + audio
    category_folders = {}
    for cat, cat_info in category_mapping.items():
        # Step 1: Market + Category folder (e.g., ./stock_analysis_report/HK/holdings)
        category_root = os.path.join(base_output_dir, market_folder, cat)
        # Step 2: Date subfolder (e.g., ./stock_analysis_report/HK/holdings/20260216)
        date_folder = os.path.join(category_root, today)
        # Step 3: Image + Audio folders inside date folder
        img_folder = os.path.join(date_folder, "img")
        audio_folder = os.path.join(date_folder, "audio")  # ✅ New: Audio folder

        # Create all folders (exist_ok=True to avoid errors if folders already exist)
        for folder in [category_root, date_folder, img_folder, audio_folder]:
            os.makedirs(folder, exist_ok=True)

        category_folders[cat] = {
            "category_root": category_root,
            "report_folder": date_folder,
            "img_folder": img_folder,
            "audio_folder": audio_folder,  # ✅ Add audio folder path
            "cn_name": cat_info["cn_name"],
            "en_name": cat_info["en_name"]
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
        "market_en": market_en,
        "market_folder": market_folder,
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
        return {"cn": "上升趋势", "en": "Uptrend"}
    elif c < ma20 < ma60:
        return {"cn": "下降趋势", "en": "Downtrend"}
    else:
        return {"cn": "横盘整理", "en": "Sideways Trend"}


def rsi_status(v, config):
    if v >= config["rsi_overbought"]:
        return {"cn": "超买", "en": "Overbought"}
    elif v <= config["rsi_oversold"]:
        return {"cn": "超卖", "en": "Oversold"}
    else:
        return {"cn": "中性", "en": "Neutral"}


# ------------------------------------------------------------------------------
# ✅ UPDATED: get_ai_analysis (Dual Language: Chinese + English)
# ------------------------------------------------------------------------------
def get_ai_analysis(stock, trend, close, ma20, ma60, rsi, rsi_text, support, resistance, config):
    """
    Call AI API to generate stock analysis in BOTH Cantonese and English (short-term 1-5 days)
    """
    market_cn = config["market_cn"]
    market_en = config["market_en"]

    # Chinese Prompt
    prompt_cn = f"""
你是專業{market_cn}波段交易員，只用廣東話分析，專注1-5日超短線，語言簡潔。
股票：{stock}
現價：{close:.2f}
趨勢：{trend['cn']}
MA20：{ma20:.2f} MA60：{ma60:.2f}
RSI：{rsi:.1f} ({rsi_text['cn']})
支撐：{support:.2f} 阻力：{resistance:.2f}

格式：
1. 趨勢總結
2. 勝率
3. 情景分析
4. 交易建議
"""

    # English Prompt
    prompt_en = f"""
You are a professional {market_en} swing trader, analyze in English only, focus on 1-5 days short-term, concise language.
Stock: {stock}
Current Price: {close:.2f}
Trend: {trend['en']}
MA20: {ma20:.2f} | MA60: {ma60:.2f}
RSI: {rsi:.1f} ({rsi_text['en']})
Support: {support:.2f} | Resistance: {resistance:.2f}

Format:
1. Trend Summary
2. Win Rate
3. Scenario Analysis
4. Trading Advice
"""

    # Get Chinese Analysis
    headers = {"Authorization": f"Bearer {config['doubao_api_key']}", "Content-Type": "application/json"}
    ai_cn = "AI分析失敗（請檢查API配置）"
    ai_en = "AI analysis failed (please check API configuration)"

    try:
        # Chinese Analysis
        payload_cn = json.dumps({
            "model": config["model"], "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt_cn}]
        })
        conn = http.client.HTTPSConnection(config["api_host"])
        conn.request("POST", config["api_path"], payload_cn, headers)
        resp_cn = json.loads(conn.getresponse().read().decode())
        ai_cn = resp_cn["choices"][0]["message"]["content"]
        conn.close()

        # English Analysis
        payload_en = json.dumps({
            "model": config["model"], "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt_en}]
        })
        conn = http.client.HTTPSConnection(config["api_host"])
        conn.request("POST", config["api_path"], payload_en, headers)
        resp_en = json.loads(conn.getresponse().read().decode())
        ai_en = resp_en["choices"][0]["message"]["content"]
        conn.close()

    except Exception as e:
        print(f"❌ AI分析失敗 for {stock}: {str(e)}")
        ai_cn = f"AI分析失敗：{str(e)}"
        ai_en = f"AI analysis failed: {str(e)}"

    return {"cn": ai_cn, "en": ai_en}


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

    # ✅ Save to Option 2 Structure: DATE/img folder
    img_folder = config["category_folders"][category]["img_folder"]
    chart_filename = f"{stock}_chart_{config['today']}.png"
    path = os.path.join(img_folder, chart_filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F5F5F5")
    plt.close()
    return path


# ------------------------------------------------------------------------------
# ✅ UPDATED: Generate Single Stock Report (With Audio Embedding)
# ------------------------------------------------------------------------------
async def generate_single_stock_report(stock_report, config, category):
    """Generate pure Markdown report (no HTML) with web-ready audio links"""
    # Get basic info
    stock = stock_report["stock"]
    today = config["today"]
    market_folder = config["market_folder"]  # HK/US
    category_info = config["category_folders"][category]
    date_folder = category_info["report_folder"]
    audio_folder = category_info["audio_folder"]

    # Clean stock code for filename (replace . with empty, e.g., 0700.HK → 0700HK)
    clean_stock_code = stock.replace(".", "")

    # 1. Generate Audio Files
    cn_analysis_text = f"{stock} 技術分析報告。{stock_report['ai']['cn']}"
    en_analysis_text = f"{stock} Technical Analysis Report. {stock_report['ai']['en']}"
    clean_cn_text = clean_text_for_tts(cn_analysis_text)
    clean_en_text = clean_text_for_tts(en_analysis_text)

    # Audio filenames (web-friendly: no spaces/special chars)
    cantonese_audio_filename = f"{clean_stock_code}_cantonese_{today}.mp3"
    mandarin_audio_filename = f"{clean_stock_code}_mandarin_{today}.mp3"
    english_audio_filename = f"{clean_stock_code}_english_{today}.mp3"

    cantonese_audio_path = os.path.join(audio_folder, cantonese_audio_filename)
    mandarin_audio_path = os.path.join(audio_folder, mandarin_audio_filename)
    english_audio_path = os.path.join(audio_folder, english_audio_filename)

    # Generate audio (Edge TTS)
    await generate_audio(clean_cn_text, "zh-HK", cantonese_audio_path)
    await generate_audio(clean_cn_text, "zh-CN", mandarin_audio_path)
    await generate_audio(clean_en_text, "en-US", english_audio_path)

    # 2. Web-optimized relative paths (critical for stock_screener_web)
    # Path format: /{market}/{category}/{date}/audio/{filename} (absolute for web)
    # OR relative path (works if MD/HTML is in /{market}/{category}/{date}/):
    img_relative_path = f"img/{os.path.basename(stock_report['image'])}"
    cantonese_audio_rel = f"audio/{cantonese_audio_filename}"  # Web-ready relative path
    mandarin_audio_rel = f"audio/{mandarin_audio_filename}"
    english_audio_rel = f"audio/{english_audio_filename}"

    # 3. Pure Markdown content (NO HTML)
    md_content = [
        # Chinese Section (Pure Markdown)
        f"# {config['market_cn']} - {category_info['cn_name']}",
        f"## {stock} 技術分析報告",
        f"**生成時間**: {today}",
        "",
        f"![{stock} 日线图]({img_relative_path})",
        "",
        "### 📊 核心技術指標",
        f"- 最新價格: {stock_report['close']:.2f}",
        f"- 技術趨勢: {stock_report['trend']['cn']}",
        f"- MA20: {stock_report['ma20']:.2f} | MA60: {stock_report['ma60']:.2f}",
        f"- RSI({config['rsi_period']}): {stock_report['rsi']:.1f} ({stock_report['rsi_text']['cn']})",
        f"- 支撑位: {stock_report['support']} | 阻力位: {stock_report['resistance']}",
        "",
        "### 📝 AI超短線分析 (1-5日)",
        stock_report['ai']['cn'],
        "",
        "### 🎧 語音版本",
        f"- 粵語版: [{cantonese_audio_filename}]({cantonese_audio_rel})",  # Pure MD link
        f"- 普通話版: [{mandarin_audio_filename}]({mandarin_audio_rel})",
        "",
        "---",
        "",
        # English Section (Pure Markdown)
        f"# {config['market_en']} - {category_info['en_name']}",
        f"## {stock} Technical Analysis Report",
        f"**Generated Time**: {today}",
        "",
        f"![{stock} Daily Chart]({img_relative_path})",
        "",
        "### 📊 Core Technical Indicators",
        f"- Current Price: {stock_report['close']:.2f}",
        f"- Technical Trend: {stock_report['trend']['en']}",
        f"- MA20: {stock_report['ma20']:.2f} | MA60: {stock_report['ma60']:.2f}",
        f"- RSI({config['rsi_period']}): {stock_report['rsi']:.1f} ({stock_report['rsi_text']['en']})",
        f"- Support Level: {stock_report['support']} | Resistance Level: {stock_report['resistance']}",
        "",
        "### 📝 AI Short-Term Analysis (1-5 Days)",
        stock_report['ai']['en'],
        "",
        "### 🎧 Audio Version",
        f"- English Version: [{english_audio_filename}]({english_audio_rel})",  # Pure MD link
    ]

    # 4. Save pure Markdown file
    report_filename = f"{market_folder}_{category}_{clean_stock_code}_{today}.md"
    report_path = os.path.join(date_folder, report_filename)
    async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
        await f.write("\n".join(md_content))

    print(f"✅ 生成純Markdown報告(網頁優化): {report_path}")
    return report_path


# ------------------------------------------------------------------------------
# ✅ UPDATED: Analyze Category (Async for Audio Generation)
# ------------------------------------------------------------------------------
async def analyze_category_async(category, config):
    category_cn = config["category_folders"][category]["cn_name"]
    stocks = config["stock_lists"][category]
    print(f"\n📈 開始分析 {category_cn} (共{len(stocks)}只股票): {', '.join(stocks)}")
    generated_reports = []

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

            # Get dual-language AI analysis
            ai_analysis = get_ai_analysis(stock, trend, latest.Close, latest.MA20, latest.MA60,
                                          rsi_val, rsi_txt, sup, res, config)

            # Build stock report dict
            stock_report = {
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
                "ai": ai_analysis
            }

            # Generate single stock report (with audio)
            report_path = await generate_single_stock_report(stock_report, config, category)
            generated_reports.append(report_path)

            print(f"  ✅ {stock} 分析完成，報告+音頻已生成")

        except Exception as e:
            print(f"  ❌ 分析 {stock} 出错: {str(e)}")
            continue

    return generated_reports


# ------------------------------------------------------------------------------
# ✅ UPDATED: Main Program (Async Entry Point)
# ------------------------------------------------------------------------------
async def main():
    config = init()
    market_cn = config["market_cn"]
    today = config["today"]
    base_output_dir = config["base_output_dir"]

    print(f"🚀 開始{market_cn}全類別分析 ({today})")
    print(f"📁 輸出目錄: {base_output_dir}")

    all_categories = ["market", "strong_trend", "watch", "holding"]
    for category in all_categories:
        generated_reports = await analyze_category_async(category, config)
        if generated_reports:
            category_cn = config["category_folders"][category]["cn_name"]
            print(f"✅ {category_cn} 共生成 {len(generated_reports)} 個單股報告(含音頻)")
        else:
            category_cn = config["category_folders"][category]["cn_name"]
            print(f"⚠️ {category_cn} 無有效分析結果")

    print(f"\n✅ 所有{market_cn}類別分析完成！")
    print(f"📄 報告位置: {base_output_dir}/{config['market_folder']} (每只股票單獨文件+音頻)")


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())