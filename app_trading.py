import streamlit as st
import vectorbt as vbt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from weasyprint import HTML
import base64
import io

# Streamlit interface

st.set_page_config(page_title='Backtesting', layout='wide')
st.title("Backtesting")

if 'strategies' not in st.session_state:
    st.session_state.strategies = []

class TradingStrategy:
    def __init__(self, symbol, start_date, end_date, strategy_name, params):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_name = strategy_name
        self.params = params
        self.close_data = None
        self.open_data = None
        self.high_data = None
        self.low_data = None
        self.entries = None
        self.exits = None

    def fetch_data(self):
        start_date_tz = pd.Timestamp(self.start_date, tz='UTC')
        end_date_tz = pd.Timestamp(self.end_date, tz='UTC')
        self.close_data = vbt.YFData.download(self.symbol, start=start_date_tz, end=end_date_tz).get('Close')
        self.open_data = vbt.YFData.download(self.symbol, start=start_date_tz, end=end_date_tz).get('Open')
        self.high_data = vbt.YFData.download(self.symbol, start=start_date_tz, end=end_date_tz).get('High')
        self.low_data = vbt.YFData.download(self.symbol, start=start_date_tz, end=end_date_tz).get('Low')

    def calculate_signals(self):
        if self.strategy_name == "MA Crossover":
            short_ema = vbt.MA.run(self.close_data, self.params['short_period'], short_name='fast', ewm=True)
            long_ema = vbt.MA.run(self.close_data, self.params['long_period'], short_name='slow', ewm=True)
            self.entries = short_ema.ma_crossed_above(long_ema)
            self.exits = short_ema.ma_crossed_below(long_ema)
            
        elif self.strategy_name == "RSI Strategy":
            rsi = vbt.RSI.run(self.close_data, self.params['rsi_period'])
            self.entries = rsi.rsi_crossed_below(self.params['rsi_oversold'])
            self.exits = rsi.rsi_crossed_above(self.params['rsi_overbought'])
        
        elif self.strategy_name == "Bollinger Bands":
            bb = vbt.BBANDS.run(self.close_data, self.params['bb_period'], self.params['bb_std'])
            self.entries = self.close_data < bb.lower
            self.exits = self.close_data > bb.upper
            
        elif self.strategy_name == "MACD":
            macd = vbt.MACD.run(self.close_data, self.params['macd_fast'], self.params['macd_slow'], self.params['macd_signal'])
            self.entries = macd.macd_above(macd.signal)
            self.exits = macd.macd_below(macd.signal)
            
        else:
            raise ValueError(f"Unknown strategy {self.strategy_name}")

    def run_backtest(self, initial_equity, size, size_type, fees, direction):
        if size_type == 'percent':
            size_value = float(size) / 100.0
        else:
            size_value = float(size)

        portfolio = vbt.Portfolio.from_signals(
            self.close_data, self.entries, self.exits,
            direction=direction,
            size=size_value,
            size_type=size_type,
            fees=fees/100,
            init_cash=initial_equity,
            freq='1D',
            min_size=1,
            size_granularity=1
        )
        
        return portfolio

# Sidebar for inputs
with st.sidebar:
    st.header("Strategy Controls")
    
    # Inputs for the symbol, start and end dates
    symbol = st.text_input("Enter the symbol (e.g., 'AAPL')", value="HDFCBANK.NS")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    # Container for dynamically adding strategies
    strategies_container = st.container()
    strategy_params = []

    with strategies_container:
        num_strategies = st.number_input("Number of Strategies", min_value=1, max_value=10, value=1)

        for i in range(num_strategies):
            st.subheader(f"Strategy {i + 1}")
            strategy = st.selectbox(f"Select Strategy {i + 1}", ["MA Crossover","RSI Strategy","Bollinger Bands","MACD"], key=f"strategy_{i}")

            params = {}
            if strategy == "MA Crossover":
                params['short_period'] = st.number_input(f"Short MA Period {i + 1}", value=10, min_value=1, key=f"short_ma_{i}")
                params['long_period'] = st.number_input(f"Long MA Period {i + 1}", value=20, min_value=1, key=f"long_ma_{i}")
            elif strategy == "RSI Strategy":
                params['rsi_period'] = st.number_input(f"RSI Period {i + 1}", value=14, min_value=1, key=f"rsi_period_{i}")
                params['rsi_oversold'] = st.number_input(f"RSI Oversold Level {i + 1}", value=30, min_value=1, max_value=50, key=f"rsi_oversold_{i}")
                params['rsi_overbought'] = st.number_input(f"RSI Overbought Level {i + 1}", value=70, min_value=50, max_value=100, key=f"rsi_overbought_{i}")
            elif strategy == "Bollinger Bands":
                params['bb_period'] = st.number_input(f"BB Period {i + 1}", value=20, min_value=1, key=f"bb_period_{i}")
                params['bb_std'] = st.number_input(f"BB Standard Deviations {i + 1}", value=2.0, min_value=0.1, key=f"bb_std_{i}")
            elif strategy == "MACD":
                params['macd_fast'] = st.number_input(f"MACD Fast Period {i + 1}", value=12, min_value=1, key=f"macd_fast_{i}")
                params['macd_slow'] = st.number_input(f"MACD Slow Period {i + 1}", value=26, min_value=1, key=f"macd_slow_{i}")
                params['macd_signal'] = st.number_input(f"MACD Signal Period {i + 1}", value=9, min_value=1, key=f"macd_signal_{i}")
            
            strategy_params.append((strategy, params))

    st.header("Backtesting Controls")

    # Backtesting controls
    initial_equity = st.number_input("Initial Equity", value=100000)
    size = st.text_input("Position Size", value='50')  # Text input for size
    size_type = st.selectbox("Size Type", ["amount", "value", "percent"], index=2)  # Dropdown for size type
    fees = st.number_input("Fees (as %)", value=0.12, format="%.4f")
    direction = st.selectbox("Direction", ["longonly", "shortonly", "both"], index=0)

    combine_method = st.selectbox("Combine Method", ["AND", "OR"], index=0)

    # Button to perform backtesting
    backtest_clicked = st.button("Backtest")

# Main area for results
if backtest_clicked:
    combined_entries = None
    combined_exits = None
    trading_strategies = []
    
    for strategy, params in strategy_params:
        trading_strategy = TradingStrategy(symbol, start_date, end_date, strategy, params)
        trading_strategy.fetch_data()
        trading_strategy.calculate_signals()
        trading_strategies.append(trading_strategy)
        
        if combined_entries is None:
            combined_entries = trading_strategy.entries
            combined_exits = trading_strategy.exits
        else:
            if combine_method == "AND":
                combined_entries = combined_entries & trading_strategy.entries
                combined_exits = combined_exits & trading_strategy.exits
            elif combine_method == "OR":
                combined_entries = combined_entries | trading_strategy.entries
                combined_exits = combined_exits | trading_strategy.exits

    # Use combined signals to run backtest
    final_strategy = trading_strategies[0]
    final_strategy.entries = combined_entries
    final_strategy.exits = combined_exits
    portfolio = final_strategy.run_backtest(initial_equity, size, size_type, fees, direction)
    
    trades_df = portfolio.trades.records_readable.round(2)
    trades_df.index.name = 'Trade No' 
    trades_df.drop(trades_df.columns[[0, 1]], axis=1, inplace=True)
    trades_df.drop(columns=['Position Id'], inplace = True)
    
    # Store the results in session state
    st.session_state.strategies.append({
        "strategy": "Combined Strategy",
        "params": {
            "strategies": strategy_params,
            "combine_method": combine_method,
            "start_date": start_date,
            "end_date": end_date,
            "initial_equity": initial_equity,
            "size": size,
            "size_type": size_type,
            "fees": fees,
            "direction": direction,
        },
        "portfolio": portfolio,
        "stats": pd.DataFrame(portfolio.stats(), columns=['Value']),
        "trades": trades_df,
        "plot": portfolio.plot()
    })

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Backtesting Stats", "List of Trades", "Portfolio Plot"])

    with tab1:
        st.markdown(f"**Backtesting Stats for Combined Strategy:**")
        st.dataframe(st.session_state.strategies[-1]['stats'], height=800)

    with tab2:
        st.markdown(f"**List of Trades for Combined Strategy:**")
        st.dataframe(trades_df, width=800, height=600)

    with tab3:
        st.markdown(f"**Portfolio Plot for Combined Strategy:**")
        st.plotly_chart(st.session_state.strategies[-1]['plot'])
        

# Function to generate Excel file
def generate_excel_file(strategies):
    output = io.BytesIO()

    # Iterate over all backtest results
    i = 0
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for strat in st.session_state.strategies:
            trades_df = strat['trades']

            trades_df['Entry Timestamp'] = trades_df['Entry Timestamp'].apply(lambda a: pd.to_datetime(a).date())
            trades_df['Exit Timestamp'] = trades_df['Exit Timestamp'].apply(lambda a: pd.to_datetime(a).date())

            # Write DataFrame to a new sheet
            trades_df.to_excel(writer, sheet_name=f"Strategy_{i+1}", index=True)
            i = i+1

    processed_data = output.getvalue()
    return processed_data


# Function to generate HTML content for all strategies
def generate_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        table-layout: fixed;
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 4px;
        font-size: 12px;
        word-wrap: break-word;
        width: 48px;
    }
    th {
        background-color: #f2f2f2;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
    </head>
    <body>
    """
    for strat in st.session_state.strategies:
        html_content += f"<h1>Backtesting Results - {strat['strategy']}</h1>"
        html_content += "<h2>Strategy Parameters</h2><ul>"
        for key, value in strat['params'].items():
            html_content += f"<li><strong>{key}:</strong> {value}</li>"
        html_content += "</ul>"
        html_content += "<h2>Backtesting Stats</h2>"
        html_content += strat['stats'].to_html()
        # html_content += "<h2>List of Trades</h2>"
        # trades_df = strat['trades']
        # html_content += trades_df.to_html()
        img_bytes = pio.to_image(strat['plot'], format="png")
        img_str = base64.b64encode(img_bytes).decode()
        html_content += f'<h2>Portfolio Plot</h2><img src="data:image/png;base64,{img_str}">'
    html_content += "</body></html>"
    return html_content

# Function to convert HTML content to PDF
def convert_html_to_pdf(html_content):
    pdf_filename = "backtest_results.pdf"
    HTML(string=html_content).write_pdf(pdf_filename)
    return pdf_filename


if st.button("Download Excel File"):
    # Generate Excel file with all backtest results
    excel_data = generate_excel_file(st.session_state.strategies)
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="backtest_results.xlsx">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Button to generate PDF with all results
if st.button("Generate PDF for All Strategies"):
    html_content = generate_html()
    pdf_filename = convert_html_to_pdf(html_content)
    st.success(f"PDF with all strategies saved as {pdf_filename}")

    # Provide download link for the PDF
    with open(pdf_filename, "rb") as f:
        pdf_data = f.read()
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)