import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from yahooquery import Screener

# ======================
# 你的筛选参数
# ======================
MIN_MARKET_CAP = 30000000000    # 300亿美金
MIN_AVG_VOLUME = 1000000        # 日均成交量 > 100万
MIN_PRICE      = 10
MIN_DAYS       = 7             # 期权到期 ≥7天
MAX_DAYS       = 30            # 期权到期 ≤30天
VIX_SAFE       = 25

# ======================
# 1. 本周大市判断（Short Put 能否开仓）
# ======================
def is_week_good_for_short_put():
    print("=" * 70)
    print("📊 本周大市风险分析 · 是否适合做 Short Put")
    print("=" * 70)
    try:
        vix = yf.Ticker("^VIX")
        vix_price = vix.history(period="1d")["Close"].iloc[-1]

        spx = yf.Ticker("^GSPC")
        h = spx.history(period="5d")
        week_ret = (h["Close"].iloc[-1] / h["Close"].iloc[0]) - 1

        ok_vix = vix_price < VIX_SAFE
        ok_trend = week_ret > -0.015
        suitable = ok_vix and ok_trend

        print(f"VIX恐慌指数：{vix_price:.1f}")
        print(f"标普500 本周表现：{week_ret:.1%}")
        print("-"*70)
        if suitable:
            print("🟢 结论：本周适合做 Short Put ✅")
        else:
            print("🔴 结论：本周不建议做 Short Put ❌")
        print("="*70)
        return suitable
    except:
        return False

# ======================
# 2. 正版 yahooquery 攞全美股
# ======================
def get_all_us_stocks():
    s = Screener()
    data = s.get_screeners(["all_us_stocks"], count=10000)
    quotes = data["all_us_stocks"]["quotes"]
    return [q["symbol"] for q in quotes if "symbol" in q]

# ======================
# 3. 【你最想要】个股 Put/Call Ratio + 建议 Short Put 价位
# ======================
def analyze_put_call_ratio_and_safe_strike(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.info.get("currentPrice", 0)
        if price < MIN_PRICE:
            return None

        # 拿到所有 7~30 天到期的期权
        exp_list = tk.options
        today = datetime.now()
        valid_exps = []
        for exp in exp_list:
            d = datetime.strptime(exp, "%Y-%m-%d")
            days = (d - today).days
            if MIN_DAYS <= days <= MAX_DAYS:
                valid_exps.append(exp)

        if not valid_exps:
            return None

        # 汇总所有近月期权的 Call / Put 成交量
        total_call_vol = 0
        total_put_vol = 0
        all_puts = []

        for exp in valid_exps:
            opt = tk.option_chain(exp)
            calls = opt.calls
            puts = opt.puts

            total_call_vol += calls["volume"].sum()
            total_put_vol += puts["volume"].sum()
            all_puts.append(puts)

        all_puts = pd.concat(all_puts)
        put_vol_sum = total_put_vol
        call_vol_sum = total_call_vol

        if call_vol_sum == 0:
            return None

        # 个股 Put/Call Ratio
        put_call_ratio = put_vol_sum / call_vol_sum
        has_put_support = put_call_ratio < 0.7  # 愈细愈多人买Call → 个股强

        # 筛选 OTM Put，流动性好，建议最稳价位
        valid_puts = all_puts[
            (all_puts["strike"] < price * 0.90)  # 至少 10% OTM
            & (all_puts["volume"] > 50)
            & (all_puts["openInterest"] > 100)
        ].copy()

        if valid_puts.empty:
            return None

        # 选 OpenInterest 最大的 Put 做建议价
        best = valid_puts.sort_values("openInterest", ascending=False).iloc[0]
        suggest_strike = best["strike"]

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "put_call_ratio": round(put_call_ratio, 2),
            "put_support": "🟢 强" if has_put_support else "🔴 弱",
            "suggest_short_put_below": round(suggest_strike, 2),
            "otm_pct": round((1 - suggest_strike/price)*100, 1)
        }
    except:
        return None

# ======================
# 4. 主筛选
# ======================
def screen_best_short_put_stocks(symbols):
    result = []
    print("\n开始扫描个股 Put/Call Ratio + 建议价位...\n")
    for sym in symbols[:2000]:  # 大盘股范围
        data = analyze_put_call_ratio_and_safe_strike(sym)
        if data:
            result.append(data)
            print(
                f"{data['symbol']} | 价 ${data['price']} | "
                f"P/C Ratio {data['put_call_ratio']} | {data['put_support']} | "
                f"建议 Short Put ≤ {data['suggest_short_put_below']} "
                f"({data['otm_pct']}% OTM)"
            )
    df = pd.DataFrame(result)
    return df.sort_values("put_call_ratio")

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    all_symbols = get_all_us_stocks()
    df = screen_best_short_put_stocks(all_symbols)
    if not df.empty:
        df.to_csv("ultra_short_put_list.csv", index=False)
        print("\n✅ 清单已保存 ultra_short_put_list.csv")
    else:
        print("\n⚠️ 暂时没有符合条件的标的")