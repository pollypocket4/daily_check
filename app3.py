import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.title("📈 股票 K 線圖 + MA / EMA 分析工具（互動版）")

symbol_input = st.text_input("請輸入股票代碼（例如 AAPL、tsla、nvda）", value = "AAPL")
symbol = symbol_input.strip().upper()

today = pd.to_datetime("today")
start_date = st.date_input("選擇開始日期", today - pd.DateOffset(years=1))
end_date = st.date_input("選擇結束日期", today - pd.Timedelta(days=1))

if st.button("取得資料並顯示 K 線圖"):
    try:
        max_window = 120
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=max_window * 2)

        df = yf.download(symbol, start=extended_start, end=end_date, group_by='ticker')

        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(0):
                df = df.xs(symbol, axis=1, level=0)
            else:
                st.error(f"❌ 多層欄位結構中找不到 '{symbol}'，請確認代碼正確")
                st.write("實際欄位名稱：", df.columns.tolist())
                st.stop()

        if df.empty or 'Close' not in df.columns:
            st.error("❌ 找不到 Close 資料，請確認股票代碼是否正確")
            st.write("實際欄位名稱：", df.columns.tolist())
            st.stop()

        df['Close'] = df['Close'].fillna(method='ffill')
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean()
        df['MA120'] = df['Close'].rolling(window=120, min_periods=1).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
        df['EMA120'] = df['Close'].ewm(span=120, adjust=False).mean()

        df['golden_cross'] = (df['EMA20'] > df['EMA60']) & (df['EMA20'].shift(1) <= df['EMA60'].shift(1))
        df['death_cross'] = (df['EMA20'] < df['EMA60']) & (df['EMA20'].shift(1) >= df['EMA60'].shift(1))

        def calc_slope(series):
            return np.rad2deg(np.arctan((series - series.shift(5)) / 5))

        df['EMA20_slope'] = calc_slope(df['EMA20'])
        df['EMA60_slope'] = calc_slope(df['EMA60'])
        df['EMA120_slope'] = calc_slope(df['EMA120'])

        df['ema20_pullback'] = (df['Close'] > df['EMA20']) & (df['Close'].shift(1) > df['EMA20'].shift(1)) & (abs(df['Close'] - df['EMA20']) / df['EMA20'] < 0.01)

        df['bias_ema20'] = (df['Close'] - df['EMA20']) / df['EMA20'] * 100

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').astype('float64')

        plot_df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            row_heights=[0.8, 0.2], vertical_spacing=0.02)

        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'],
            name='K線'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], line=dict(color='#CCCCCC', width=1, dash='dash'), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA60'], line=dict(color='#FF9999', width=1, dash='dash'), name='MA60'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA120'], line=dict(color='#46A3FF', width=1, dash='dash'), name='MA120'), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA20'], line=dict(color='white', width=1), name='EMA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA60'], line=dict(color='red', width=1), name='EMA60'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA120'], line=dict(color='blue', width=1), name='EMA120'), row=1, col=1)

        golden_crosses = df[df['golden_cross'] & (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        fig.add_trace(go.Scatter(
            x=golden_crosses.index,
            y=golden_crosses['Close'],
            mode='markers',
            marker=dict(color='lime', size=10, symbol='circle'),
            name='黃金交叉'
        ), row=1, col=1)

        death_crosses = df[df['death_cross'] & (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        fig.add_trace(go.Scatter(
            x=death_crosses.index,
            y=death_crosses['Close'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='死亡交叉'
        ), row=1, col=1)

        pullback_points = df[df['ema20_pullback'] & (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        fig.add_trace(go.Scatter(x=pullback_points.index, y=pullback_points['Close'], mode='markers', name='Pullback EMA20',
                                 marker=dict(color='orange', size=10, symbol='circle')), row=1, col=1)

        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color='lightgray', name='Volume'), row=2, col=1)

        fig.update_layout(
            title=f"{symbol} K 線圖（互動式）",
            xaxis_title="日期",
            yaxis_title="價格",
            xaxis_rangeslider_visible=False,
            height=800,
            hovermode="x unified",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            showlegend=True
        )

        fig.update_xaxes(showgrid=True, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridcolor='gray')

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 最近資料")
        st.dataframe(df.tail(10))

        st.subheader("📐 均線角度 (近5日斜率)")
        latest = df.iloc[-1]
        st.write(f"EMA20 斜率：{latest['EMA20_slope']:.2f}°")
        st.write(f"EMA60 斜率：{latest['EMA60_slope']:.2f}°")
        st.write(f"EMA120 斜率：{latest['EMA120_slope']:.2f}°")

        st.subheader("📉 乖離率（收盤價對 EMA20）")
        bias_df = plot_df[['Close', 'EMA20', 'bias_ema20']].copy()
        bias_df.rename(columns={'bias_ema20': '乖離率 (%)'}, inplace=True)
        bias_df['乖離率 (%)'] = bias_df['乖離率 (%)'].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
        st.dataframe(bias_df.tail(10))

        cross_df = plot_df[plot_df['golden_cross'] | plot_df['death_cross']]

        if not cross_df.empty:
            st.subheader("📊 交叉點後漲跌幅分析")

            analysis_results = []
            for date, row in cross_df.iterrows():
                cross_type = "黃金交叉" if row['golden_cross'] else "死亡交叉"
                close_price = row['Close']

                current_idx = plot_df.index.get_loc(date)
                future_5_price = plot_df['Close'].iloc[current_idx + 5] if current_idx + 5 < len(plot_df) else None
                future_10_price = plot_df['Close'].iloc[current_idx + 10] if current_idx + 10 < len(plot_df) else None

                pct_5 = (future_5_price - close_price) / close_price * 100 if future_5_price else None
                pct_10 = (future_10_price - close_price) / close_price * 100 if future_10_price else None

                analysis_results.append({
                    "交叉日期": date.date(),
                    "交叉類型": cross_type,
                    "當日價格": close_price,
                    "5日後價格": future_5_price,
                    "5日後漲跌幅 (%)": pct_5,
                    "10日後價格": future_10_price,
                    "10日後漲跌幅 (%)": pct_10,
                })

            analysis_df = pd.DataFrame(analysis_results)
            analysis_df.sort_values("交叉日期", ascending=False, inplace=True)

            def format_change(x):
                if pd.isna(x):
                    return ""
                return f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%"

            analysis_df["5日後漲跌幅 (%)"] = analysis_df["5日後漲跌幅 (%)"].apply(format_change)
            analysis_df["10日後漲跌幅 (%)"] = analysis_df["10日後漲跌幅 (%)"].apply(format_change)

            for col in ["當日價格", "5日後價格", "10日後價格"]:
                analysis_df[col] = analysis_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

            def highlight_cross_type_only(row):
                style = [''] * len(row)
                col_idx = row.index.get_loc("交叉類型")
                if row["交叉類型"] == "黃金交叉":
                    style[col_idx] = 'background-color: #228B22; color: white'
                elif row["交叉類型"] == "死亡交叉":
                    style[col_idx] = 'background-color: #8B0000; color: white'
                return style

            st.dataframe(analysis_df.style.apply(highlight_cross_type_only, axis=1), use_container_width=True)

        else:
            st.info("尚未偵測到黃金交叉或死亡交叉")

    except Exception as e:
        import traceback
        st.error(f"發生錯誤：{e}")
        st.text(traceback.format_exc())