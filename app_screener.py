import streamlit as st
import pandas as pd
import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import seaborn as sns
import datetime
import statistics
import requests
import io
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

####################---- Screener Page ----####################
def screener_page():
    # Define financial ratios to score
    CAT1_RATIOS = ['D2E', 'PEG', 'PE fwd', 'PB', 'Beta']
    CAT2_RATIOS = ['ROCE', 'ROE', 'FCFY', 'CR', 'QR', 'Asset TR', 'EPS fwd']

    # Filepath for storing the data
    CSV_FILE = "sp500_data.csv"

    # Function to fetch and store S&P 500 data
    @st.cache_data
    def fetch_and_store_sp500_data():
        sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]

        def fetch_stock_info(ticker):
            stock_info = yf.Ticker(ticker).info
            data = {}
            data['Symbol'] = stock_info.get('symbol', np.nan)
            data['Name'] = stock_info.get('longName', np.nan)
            data['Sector'] = stock_info.get('sector', np.nan)
            data['Market Cap'] = stock_info.get('marketCap', np.nan)
            data['Price'] = stock_info.get('currentPrice', np.nan)
            data['PB'] = stock_info.get('priceToBook', np.nan)
            data['EPS fwd'] = stock_info.get('forwardEps', np.nan)
            data['PE fwd'] = stock_info.get('forwardPE', np.nan)
            data['PEG'] = stock_info.get('pegRatio', np.nan)
            data['D2E'] = stock_info.get('debtToEquity', np.nan)
            data['ROE'] = stock_info.get('returnOnEquity', np.nan)
            data['ROCE'] = np.nan
            data['FCFY'] = (stock_info.get('freeCashflow', 0) / stock_info.get('marketCap', 1)) * 100
            data['CR'] = stock_info.get('currentRatio', np.nan)
            data['QR'] = stock_info.get('quickRatio', np.nan)
            data['Asset TR'] = np.nan
            data['DY'] = stock_info.get('dividendYield', np.nan) * 100
            data['Beta'] = stock_info.get('beta', np.nan)
            data['52w Low'] = stock_info.get('fiftyTwoWeekLow', np.nan)
            data['52w High'] = stock_info.get('fiftyTwoWeekHigh', np.nan)
            return data

        sp500_data = [fetch_stock_info(ticker) for ticker in sp500_tickers]
        df = pd.DataFrame(sp500_data)

        # Compute ROCE and Asset Turnover Ratio
        def compute_roce(ticker):
            try:
                income_stmt = yf.Ticker(ticker).financials
                balance_sheet = yf.Ticker(ticker).balance_sheet
                ebit = income_stmt.loc['EBIT'].iloc[0]
                total_assets = balance_sheet.loc['Total Assets'].iloc[0]
                current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]
                return ebit / (total_assets - current_liabilities)
            except:
                return np.nan

        def compute_asset_turnover(ticker):
            try:
                income_stmt = yf.Ticker(ticker).financials
                balance_sheet = yf.Ticker(ticker).balance_sheet
                avg_assets = (balance_sheet.loc['Total Assets'].iloc[0] + balance_sheet.loc['Total Assets'].iloc[1]) / 2
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                return revenue / avg_assets
            except:
                return np.nan

        df['ROCE'] = df['Symbol'].apply(compute_roce)
        df['Asset TR'] = df['Symbol'].apply(compute_asset_turnover)

        # Remove any stocks with NaN values
        df_cleaned = df.dropna().reset_index(drop=True)

        # Add 52-week range
        df_cleaned['52w Range'] = ((df_cleaned['Price'] - df_cleaned['52w Low']) / (df_cleaned['52w High'] - df_cleaned['52w Low'])) * 100

        # Calculate momentum score
        def momentum_score(ticker):
            try:
                prices = yf.Ticker(ticker).history(period="5y")['Close']
                p0 = prices.iloc[-1]
                p1 = prices.iloc[-21]
                p3 = prices.iloc[-63]
                p6 = prices.iloc[-126]
                p12 = prices.iloc[-252]
                return (12 * (p0/p1 - 1)) + (4 * (p0/p3 - 1)) + (2 * (p0/p6 - 1)) + (p0/p12 - 1)
            except:
                return np.nan

        df_cleaned['Momentum Score'] = df_cleaned['Symbol'].apply(momentum_score)

        # Save the data to CSV
        df_cleaned.to_csv(CSV_FILE, index=False)
        return df_cleaned

    # Function to load the data from CSV
    def load_sp500_data():
        if os.path.exists(CSV_FILE):
            return pd.read_csv(CSV_FILE)
        else:
            return fetch_and_store_sp500_data()

    # Scoring system for all ratios
    def score(values, value, cat):
        std = statistics.stdev(values)
        mean = statistics.mean(values)
        if cat == 1:
            if (mean + (-1 * std)) < value <= mean:
                return 1
            elif (mean + (-2 * std)) < value <= (mean + (-1 * std)):
                return 2
            elif value <= (mean + (-2 * std)):
                return 3
            elif mean < value <= (mean + (1 * std)):
                return -1
            elif (mean + (1 * std)) < value <= (mean + (2 * std)):
                return -2
            else:
                return -3
        else:
            if mean <= value < (mean + (1 * std)):
                return 1
            elif (mean + (1 * std)) <= value < (mean + (2 * std)):
                return 2
            elif value >= (mean + (2 * std)):
                return 3
            elif (mean + (-1 * std)) <= value < mean:
                return -1
            elif (mean + (-2 * std)) <= value < (mean + (-1 * std)):
                return -2
            else:
                return -3

    # Divide the data by sector and score each sector independently
    def score_by_sector(df_cleaned):
        sector_dfs = {}
        for sector, sector_df in df_cleaned.groupby('Sector'):
            df_score = sector_df.copy()

            # Apply scoring
            for col in CAT1_RATIOS:
                df_score[col + '_score'] = [score(sector_df[col], value, 1) for value in sector_df[col]]

            for col in CAT2_RATIOS:
                df_score[col + '_score'] = [score(sector_df[col], value, 2) for value in sector_df[col]]

            # Apply momentum scoring
            df_score['Momentum_Score_normalized'] = [score(sector_df['Momentum Score'], value, 2) for value in sector_df['Momentum Score']]

            # Normalize and calculate total score for each stock
            df_score['Total Score'] = df_score[[col + '_score' for col in CAT1_RATIOS + CAT2_RATIOS] + ['Momentum_Score_normalized']].sum(axis=1)
            df_score['Total Score'] = ((df_score['Total Score'] - df_score['Total Score'].min()) /
                                    (df_score['Total Score'].max() - df_score['Total Score'].min())) * 10

            sector_dfs[sector] = df_score
        return sector_dfs

    # Streamlit App Design
    #st.set_page_config(layout="wide")
    st.title("S&P 500 Stock Screener by Sector")

    # Button to download the data (if not already downloaded)
    if st.button("Download Stock Data"):
        df_cleaned = fetch_and_store_sp500_data()
        st.success("Data downloaded and stored successfully!")

    # Button to update the data
    if st.button("Update Stock Data"):
        df_cleaned = fetch_and_store_sp500_data()
        st.success("Data updated successfully!")

    # Load the data from CSV
    df_cleaned = load_sp500_data()

    # Score by sector
    sector_dfs = score_by_sector(df_cleaned)

    # User selection for the sector
    selected_sector = st.selectbox("Select a Sector", list(sector_dfs.keys()))

    # Display the top-performing stocks in the selected sector
    st.write(f"Top stocks in {selected_sector} sector:")
    sector_df = sector_dfs[selected_sector].sort_values('Total Score', ascending=False)

    # Styling options for the table
    def style_table(df):
        df_style = df.style.format({
            "Market Cap": "{:,.0f}",
            "Price": "${:.2f}",
            "PE fwd": "{:.2f}",
            "PB": "{:.2f}",
            "PEG": "{:.2f}",
            "D2E": "{:.2f}",
            "ROE": "{:.2f}",
            "ROCE": "{:.2f}",
            "FCFY": "{:.2f}",
            "Momentum Score": "{:.2f}",
            "Total Score": "{:.2f}"
        }).background_gradient(cmap="coolwarm", subset=['Total Score', 'PE fwd', 'PB', 'Momentum Score'])
        return df_style

    # Display the full table with scores in a bigger format and with styling
    st.dataframe(style_table(sector_df), height=800)

    # ---- Sector PE Ratios vs S&P 500 PE Ratio ----
    st.subheader("Sector PE Ratios vs S&P 500 PE Ratio")

    # Compute average PE ratio for each sector
    sector_pe_ratios = {}
    for sector, sector_df in sector_dfs.items():
        valid_pe_ratios = sector_df['PE fwd'].dropna()
        if not valid_pe_ratios.empty:
            sector_pe_ratios[sector] = valid_pe_ratios.mean()

    # Calculate the average PE ratio for the entire S&P 500
    sp500_pe_ratio = df_cleaned['PE fwd'].dropna().mean()

    # Prepare data for plotting
    sectors = list(sector_pe_ratios.keys())
    pe_ratios = list(sector_pe_ratios.values())

    # Plot comparison of sector PE ratios vs S&P 500 PE ratio using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart for sector PE ratios
    ax.bar(sectors, pe_ratios, color='#1f77b4', label='Sector PE Ratios')

    # Horizontal line for S&P 500 PE ratio
    ax.axhline(y=sp500_pe_ratio, color='red', linestyle='--', label=f'S&P 500 PE (Avg: {sp500_pe_ratio:.2f})')

    # Add labels and formatting
    ax.set_title('Sector PE Ratios vs S&P 500 PE Ratio', fontsize=16, weight='bold', color='#333333')
    ax.set_ylabel('Average Forward PE Ratio', fontsize=12)
    ax.set_xticklabels(sectors, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)
    # ---- Button to Analyze Sector vs Market ----
    # ---- Function to Calculate Equal-Weighted Performance ----
    # Function to calculate equal-weighted performance with start and end dates
    def calculate_equal_weighted_performance(symbols, start, end, interval='1d'):
        # Download historical prices for all symbols within the given date range
        prices = yf.download(symbols, start=start, end=end, interval=interval)['Adj Close']
        
        # Check if prices is a Series (single symbol) or DataFrame (multiple symbols)
        if isinstance(prices, pd.Series):
            # If it's a Series, we don't need to take the mean, just normalize
            equal_weighted_normalized = (prices / prices.iloc[0]) * 100
        else:
            # If it's a DataFrame, calculate the equally weighted price
            equal_weighted_price = prices.mean(axis=1)
            # Normalize the price to start at 100
            equal_weighted_normalized = (equal_weighted_price / equal_weighted_price.iloc[0]) * 100
        
        return equal_weighted_normalized

    # User selection for the sector
    sector_list = df_cleaned['Sector'].unique().tolist()
    selected_sector = st.selectbox("Select a Sector", sector_list)

    # Filter the sector's stocks
    sector_df = df_cleaned[df_cleaned['Sector'] == selected_sector]
    sector_symbols = sector_df['Symbol'].tolist()

    # Allow users to select start and end dates
    start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('today'))

    # ---- Button to Analyze Sector vs Market ----
    if st.button("Analyze Sector vs the Market"):
        st.subheader(f'Equal-Weighted {selected_sector} vs Equal-Weighted S&P 500')

        # Fetch S&P 500 data (using SPY as proxy for simplicity)
        spy_symbols = ['SPY']  # Alternatively, use actual S&P 500 component tickers

        # Calculate equal-weighted performance for the sector and the S&P 500 within the selected date range
        sector_equal_weighted = calculate_equal_weighted_performance(sector_symbols, start=start_date, end=end_date)
        sp500_equal_weighted = calculate_equal_weighted_performance(spy_symbols, start=start_date, end=end_date)

        # Plot the sector performance vs. S&P 500 using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))

        # Line chart for sector performance
        ax.plot(sector_equal_weighted, label=f'{selected_sector} (Equal Weighted)', color='#1f77b4')

        # Line chart for S&P 500 performance
        ax.plot(sp500_equal_weighted, label='S&P 500 (Equal Weighted)', linestyle='--', color='red')

        # Add labels and formatting
        ax.set_title(f'{selected_sector} vs S&P 500 (Equal-Weighted Performance)', fontsize=16, weight='bold', color='#333333')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price (Starting at 100)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Add explanation below the graph
        st.markdown(f"""
        **Explanation**:
        - The blue line represents the normalized, equal-weighted performance of the {selected_sector} sector over the selected date range.
        - The red dashed line represents the normalized equal-weighted performance of the S&P 500.
        """)

    # Button to Analyze Momentum
    if st.button("Analyze Momentum"):
        st.subheader(f'{selected_sector} Sector Momentum vs S&P 500')

        # Download historical data for the sector and S&P 500 over the last 6 years and 1 month
        sector_symbols = sector_dfs[selected_sector]['Symbol'].tolist()
        sector_prices = yf.download(sector_symbols, period='73mo', interval='1d')['Adj Close']
        sp500_hist = yf.download('^GSPC', period='73mo', interval='1d')['Adj Close']

        # Calculate average sector price (equally weighted)
        sector_avg_price = sector_prices.mean(axis=1)
        sector_avg_price = sector_avg_price.loc[sp500_hist.index]

        # Step 5: Add momentum score based on the 13612W formula
        def momentum_score(prices):
            try:
                p0 = prices.iloc[-1]
                p1 = prices.iloc[-21]
                p3 = prices.iloc[-63]
                p6 = prices.iloc[-126]
                p12 = prices.iloc[-252]
                return (12 * (p0/p1 - 1)) + (4 * (p0/p3 - 1)) + (2 * (p0/p6 - 1)) + (p0/p12 - 1)
            except:
                return np.nan

        # Calculate momentum scores for the entire price history
        def calculate_momentum_for_series(price_series):
            momentum_scores = [momentum_score(price_series.iloc[:i]) for i in range(252, len(price_series))]
            return pd.Series(momentum_scores, index=price_series.index[252:])

        # Apply momentum score calculation to the sector and S&P 500
        sector_momentum = calculate_momentum_for_series(sector_avg_price)
        sp500_momentum = calculate_momentum_for_series(sp500_hist)

        # Remove NaN values
        momentum_df = pd.DataFrame({
            'Sector Momentum': sector_momentum,
            'S&P 500 Momentum': sp500_momentum
        }).dropna()

        # Function to normalize data using MinMaxScaler between -1 and 1
        def minmax_normalize(series):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            series_values = series.values.reshape(-1, 1)
            scaled_series = scaler.fit_transform(series_values)
            return pd.Series(scaled_series.flatten(), index=series.index)

        # Normalize the momentum scores
        momentum_df['Normalized Sector Momentum'] = minmax_normalize(momentum_df['Sector Momentum'])
        momentum_df['Normalized S&P 500 Momentum'] = minmax_normalize(momentum_df['S&P 500 Momentum'])

        # Calculate the difference in momentum (Sector - S&P 500)
        momentum_df['Momentum Difference'] = momentum_df['Normalized Sector Momentum'] - momentum_df['Normalized S&P 500 Momentum']

        # Plot the comparison with subplots using Matplotlib (Enhanced Design)
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # Plot normalized momentum comparison
        ax[0].plot(momentum_df.index, momentum_df['Normalized Sector Momentum'], label=f'{selected_sector} Momentum (Normalized)', color='blue')
        ax[0].plot(momentum_df.index, momentum_df['Normalized S&P 500 Momentum'], label='S&P 500 Momentum (Normalized)', color='black', linestyle='--')
        ax[0].set_title(f'5-Year Momentum Comparison: {selected_sector} vs S&P 500 (Normalized)', fontsize=12, weight='bold', color='#333333')
        ax[0].set_ylabel('Normalized Momentum Score (-1 to 1)', fontsize=10)
        ax[0].legend(fontsize=8)
        ax[0].grid(True)

        # Plot the momentum difference
        ax[1].plot(momentum_df.index, momentum_df['Momentum Difference'], label=f'{selected_sector} - S&P 500 Momentum Difference', color='green')
        ax[1].axhline(0, color='red', linestyle='--')
        ax[1].set_title(f'Momentum Difference: {selected_sector} vs S&P 500', fontsize=12, weight='bold', color='#333333')
        ax[1].set_xlabel('Date', fontsize=10)
        ax[1].set_ylabel('Momentum Difference', fontsize=10)
        ax[1].legend(fontsize=8)
        ax[1].grid(True)

        # Show the plot
        plt.tight_layout()
        st.pyplot(fig)

    ####################---- Macro Indicator Page ----####################
def macro_indicator_page():
    st.title("Macro Indicator: Diffusion Index (DI) Strategy")

    # Explain Diffusion Index Calculation
    st.markdown("""
    The Diffusion Index is a metric used to gauge the overall direction of economic activity. It is calculated as the 
    proportion of countries (from a selected group) that are showing positive growth in their Composite Leading Indicators (CLI). 
    The CLI data is sourced from the OECD for various countries. A DI value greater than 0.5 indicates that more than 50% of the 
    countries are experiencing positive economic growth, while a value less than 0.5 suggests contraction.

    **DI calculation**
    1. For each country, calculate the month-to-month change in CLI.
    2. Count how many countries have positive CLI changes.
    3. Divide the count of countries with positive changes by the total number of countries to obtain the proportion.
    4. This proportion is the DI.
    """)

    # DI Investment Strategy Class
    class DIInvestmentStrategy:
        def __init__(self, countries, start_date='2010-01-01', end_date='2024-10-16'):
            self.countries = countries
            self.start_date = start_date
            self.end_date = end_date
            self.cli_data = None
            self.di_series = None
            self.sp500_data = None
            self.strategy_returns = None

        def get_oecd_data(self):
            """Fetch CLI data for the specified countries."""
            st.write("Fetching CLI data from OECD...")
            database = '@DF_CLI'
            frequency = 'M'
            indicator = 'LI..'
            unit_of_measure = 'AA...'
            start_period = '2009-01'

            country_code = "+".join(self.countries)
            query_text = f"{database}/{country_code}.{frequency}.{indicator}.{unit_of_measure}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
            url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"

            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8'
            }

            download = requests.get(url=url, headers=headers)
            if download.status_code != 200:
                raise Exception(f"Failed to fetch CLI data. Status code: {download.status_code}")

            self.cli_data = pd.read_csv(io.StringIO(download.text))
            self.cli_data['TIME_PERIOD'] = pd.to_datetime(self.cli_data['TIME_PERIOD'])
            self.cli_data.set_index('TIME_PERIOD', inplace=True)
            st.write("CLI data fetched successfully.")
            st.session_state.cli_data = self.cli_data  # Store in session state
            st.dataframe(self.cli_data)

        def calculate_di(self):
            """Calculate the Diffusion Index (DI) based on the CLI data."""
            if self.cli_data is None:
                raise ValueError("CLI data not loaded. Please run get_oecd_data() first.")
            
            # Ensure data is available
            if 'cli_data' in st.session_state:
                self.cli_data = st.session_state.cli_data  # Retrieve from session state
            else:
                raise ValueError("CLI data not found in session state.")

            pivot_data = self.cli_data.pivot(columns='REF_AREA', values='OBS_VALUE')
            pivot_data.fillna(method='ffill', inplace=True)
            pivot_data_change = pivot_data.diff()

            self.di_series = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
            st.line_chart(self.di_series)
            st.write("Diffusion Index calculated.")

        def fetch_sp500_data(self):
            """Fetch S&P 500 ETF data from Yahoo Finance."""
            st.write("Fetching S&P 500 data...")
            self.sp500_data = yf.download('SPY', start=self.start_date, end=self.end_date)['Adj Close']
            if self.sp500_data.empty:
                raise ValueError("Failed to fetch S&P 500 data.")
            st.write("S&P 500 data fetched successfully.")

        def execute_strategy(self):
            if self.di_series is None or self.sp500_data is None:
                raise ValueError("Data not loaded. Please run calculate_di() and fetch_sp500_data() first.")

            strategy_df = pd.DataFrame(index=self.sp500_data.index)
            strategy_df['DI'] = self.di_series
            strategy_df['SP500'] = self.sp500_data
            strategy_df['DI'].ffill(inplace=True)

            strategy_df['Trade Day'] = strategy_df.index.to_series().apply(lambda x: x.day == 15)
            strategy_df['Trade Day'] = strategy_df['Trade Day'].astype(bool)
            strategy_df['Trade Day'] = strategy_df['Trade Day'] | (strategy_df['Trade Day'].shift(-1, fill_value=False))

            strategy_df['Allocation'] = 0  
            strategy_df.loc[strategy_df['DI'] < 0.1, 'Allocation'] = 0  
            strategy_df.loc[(strategy_df['DI'] >= 0.1) & (strategy_df['DI'] < 0.35), 'Allocation'] = 0  
            strategy_df.loc[(strategy_df['DI'] >= 0.35) & (strategy_df['DI'] < 0.5), 'Allocation'] = 0  
            strategy_df.loc[(strategy_df['DI'] >= 0.5) & (strategy_df['DI'] < 0.6), 'Allocation'] = 1  
            strategy_df.loc[(strategy_df['DI'] >= 0.7) & (strategy_df['DI'] < 0.8), 'Allocation'] = 1  
            strategy_df.loc[strategy_df['DI'] >= 0.8, 'Allocation'] = 1.0  

            strategy_df['SP500 Returns'] = strategy_df['SP500'].pct_change()
            strategy_df['Strategy Returns'] = strategy_df['Allocation'].shift(1) * strategy_df['SP500 Returns']
            strategy_df['Cumulative Strategy Returns'] = (1 + strategy_df['Strategy Returns']).cumprod()
            strategy_df['Cumulative SP500 Returns'] = (1 + strategy_df['SP500 Returns']).cumprod()

            self.strategy_returns = strategy_df

        def plot_results(self):
            if self.strategy_returns is None:
                raise ValueError("Strategy not executed. Please run execute_strategy() first.")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax1.plot(self.strategy_returns.index, self.strategy_returns['Cumulative Strategy Returns'], label='Strategy Returns')
            ax1.plot(self.strategy_returns.index, self.strategy_returns['Cumulative SP500 Returns'], label='S&P 500 Returns', alpha=0.7)
            ax1.set_title('Cumulative Returns: Strategy vs. S&P 500')
            ax1.legend()
            ax1.grid()

            ax2.plot(self.strategy_returns.index, self.strategy_returns['DI'], label='Diffusion Index', color='green')
            ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='0.5 Threshold')
            ax2.set_title('Diffusion Index (DI)')
            ax2.legend()
            ax2.grid()

            st.pyplot(fig)

    # Instantiate the strategy
    # Define the countries for the DI strategy
    countries = [
        'AUS', 'CAN', 'FRA', 'DEU', 'ITA', 'JPN', 'KOR', 'MEX', 'ESP', 'TUR', 'GBR', 'USA',
        'BRA', 'CHN', 'IND', 'IDN', 'ZAF',
        'FRA', 'DEU', 'ITA', 'GBR',
        'CAN', 'MEX', 'USA',
        'CHN', 'IND', 'IDN', 'JPN', 'KOR'
    ]   
    
    strategy = DIInvestmentStrategy(countries)

    # Ensure CLI data is loaded before any actions that require it
    if 'cli_data_loaded' not in st.session_state:
        st.session_state.cli_data_loaded = False

    # Fetch CLI Data
    if st.button("Fetch CLI Data"):
        strategy.get_oecd_data()
        st.session_state.cli_data_loaded = True

    # Calculate Diffusion Index
    if st.button("Calculate DI"):
        #if not st.session_state.cli_data_loaded:
        st.warning("Fetching CLI data first...")
        strategy.get_oecd_data()
        st.session_state.cli_data_loaded = True
        strategy.calculate_di()

    # Backtest Strategy
    if st.button("Backtest Strategy"):
        #if not st.session_state.cli_data_loaded:
        st.warning("Fetching CLI data first...")
        strategy.get_oecd_data()
        st.session_state.cli_data_loaded = True
        strategy.calculate_di()  # Ensure DI is calculated
        strategy.fetch_sp500_data()
        strategy.execute_strategy()
        strategy.plot_results()


####################---- Research Page ----####################

# Funzione per calcolare il momentum basato sulla formula 13612W
def momentum_score(ticker):
    try:
        prices = yf.Ticker(ticker).history(period="5y")['Close']
        p0 = prices.iloc[-1]
        p1 = prices.iloc[-21]
        p3 = prices.iloc[-63]
        p6 = prices.iloc[-126]
        p12 = prices.iloc[-252]
        return (12 * (p0/p1 - 1)) + (4 * (p0/p3 - 1)) + (2 * (p0/p6 - 1)) + (p0/p12 - 1)
    except:
        return None

# Funzione per calcolare lo Z-score
def z_score(ticker):
    try:
        prices = yf.Ticker(ticker).history(period="1y")['Close']
        mean = prices.mean()
        std_dev = prices.std()
        latest_price = prices.iloc[-1]
        return (latest_price - mean) / std_dev
    except:
        return None

# Funzione per ottenere il prezzo di apertura per una data specifica
def get_open_price(ticker, start_date):
    stock_data = yf.download(ticker, start=start_date, end=start_date + timedelta(days=1))
    if not stock_data.empty:
        return stock_data['Open'][0]
    return None

# Funzione della pagina di ricerca
def research_page():
    st.title("Potenziali Idee ")

    # Sintesi della ricerca
    st.markdown("""

    1. **Solida performance degli ETF finanziari**:
       - Dopo un periodo di instabilità causato dal crollo delle banche regionali statunitensi nel 2023, il settore finanziario ha mostrato una ripresa significativa. Gli ETF come il **KRE** hanno beneficiato degli aumenti dei tassi d'interesse, migliorando i margini di interesse. 
       - **Risultati chiave**: La crescita degli utili delle banche regionali è stata del **3,5% nel Q3 2024** con margini di interesse netti in crescita dell'**1,2%** rispetto al 2023.
       - **Proiezioni future**: Gli utili delle banche regionali dovrebbero crescere del **5%-7%** nel 2025.

    2. **Resilienza delle materie prime, in particolare dell'oro**:
       - L'oro continua ad essere una delle materie prime più sicure contro l'inflazione. Le banche centrali hanno aumentato gli acquisti di oro, contribuendo a mantenere il prezzo elevato. 
       - **Acquisti delle banche centrali**: Nel 2024, le riserve auree delle banche centrali hanno raggiunto **2.200 tonnellate**, un aumento del **20%** rispetto all'anno precedente.
       - **Prezzo attuale dell'oro**: L'oro viene scambiato intorno a **1.900-2.000 dollari per oncia**.

    3. **Mercati emergenti**:
       - Nonostante le sfide geopolitiche e macroeconomiche, i mercati emergenti offrono opportunità di crescita a lungo termine. Paesi come Cina e Brasile stanno implementando politiche fiscali per stimolare la crescita.
       - **Valutazioni**: Gli ETF sui mercati emergenti sono scambiati a valutazioni basse, circa **14x gli utili futuri**.
       - **Proiezioni del PIL**: La crescita del PIL nei mercati emergenti dovrebbe raggiungere il **5%** in Asia e il **4,3%** in America Latina nel 2025.

    4. **Ripresa delle banche regionali**:
       - Il **KRE** rappresenta una ripresa post-crisi, con prospettive di crescita solide a causa dell'aumento dei tassi d'interesse. Con margini di interesse più elevati, le banche regionali stanno riportando utili in crescita e sono posizionate per beneficiare di ulteriori aumenti dei tassi.

    5. **Azioni Small-Cap**:
       - Le **small-cap**, rappresentate dall'ETF **IWM**, sono posizionate per beneficiare di una ripresa economica, con proiezioni di crescita degli utili a doppia cifra. Con valutazioni storicamente basse, l'IWM rappresenta una buona opportunità di acquisto per gli investitori alla ricerca di crescita a lungo termine.
       - **Valutazione attuale**: Il P/E attuale dell'IWM è **16x**, rispetto alla media di mercato di **19x**.
    """)

    # Impostazione della tabella dinamica per il trading
    today_date = date.today()
    trade_data = [
        {"ETF": "KRE", "Ticker": "KRE", "Start Date": date(2024, 10, 15)},
        {"ETF": "GLD", "Ticker": "GLD", "Start Date": date(2024, 10, 15)},
        {"ETF": "IYF", "Ticker": "IYF", "Start Date": date(2024, 10, 15)},
        {"ETF": "IWM", "Ticker": "IWM", "Start Date": date(2024, 10, 15)},
        {"ETF": "EEM", "Ticker": "EEM", "Start Date": date(2024, 10, 15)},
    ]

    df_trades = pd.DataFrame(trade_data)
    for index, row in df_trades.iterrows():
        start_date = row['Start Date']
        open_price = get_open_price(row['Ticker'], start_date)
        df_trades.loc[index, 'Open Price'] = open_price
        stock_today = yf.download(row['Ticker'], start=today_date, end=today_date + timedelta(days=1))
        today_price = stock_today['Close'][0] if not stock_today.empty else None
        df_trades.loc[index, 'Today Price'] = today_price
        df_trades.loc[index, '% Change'] = ((today_price - open_price) / open_price) * 100 if open_price and today_price else None
        df_trades.loc[index, 'Z-Score'] = z_score(row['Ticker'])
        df_trades.loc[index, 'Momentum'] = momentum_score(row['Ticker'])

    st.subheader("Suggerimenti di Trading")
    st.markdown("Regola la data di inizio del trade per ogni ETF. Il prezzo di apertura e la variazione percentuale verranno aggiornati automaticamente.")

    st.dataframe(df_trades)

    st.subheader("Analisi Approfondita dei Trade")
    st.markdown("""
    - **KRE**: Beneficia dai rialzi dei tassi d'interesse, con una forte crescita nei prossimi trimestri. Margini di interesse in aumento, con prospettive di crescita degli utili tra il **5% e l'8%**.
    - **GLD**: L'oro è stato un rifugio sicuro durante i periodi di alta inflazione. Le banche centrali continuano ad acquistare oro, sostenendo il prezzo.
    - **IYF**: Le banche globali beneficiano dell'aumento della domanda di servizi finanziari. Si prevede un aumento degli utili del **7%-9%** per il 2024.
    - **IWM**: Le small-cap sono ben posizionate per crescere con l'allentamento delle politiche monetarie. Valutazioni a livelli storicamente bassi.
    - **EEM**: Gli ETF dei mercati emergenti offrono opportunità di crescita a lungo termine, nonostante i rischi geopolitici.
    """)

# Call the research page function
#research_page()

####################---- Stock Research Page ----####################
# Function to fetch stock info from Yahoo Finance for Fundamental Analysis
def fetch_stock_info(ticker):
    stock_info = yf.Ticker(ticker).info
    data = {}
    data['Symbol'] = stock_info.get('symbol', np.nan)
    data['Name'] = stock_info.get('longName', np.nan)
    data['Sector'] = stock_info.get('sector', np.nan)
    data['Market Cap'] = stock_info.get('marketCap', np.nan)
    data['Price'] = stock_info.get('currentPrice', np.nan)
    data['PB'] = stock_info.get('priceToBook', np.nan)
    data['EPS fwd'] = stock_info.get('forwardEps', np.nan)
    data['PE fwd'] = stock_info.get('forwardPE', np.nan)
    data['PEG'] = stock_info.get('pegRatio', np.nan)
    data['D2E'] = stock_info.get('debtToEquity', np.nan)
    data['ROE'] = stock_info.get('returnOnEquity', np.nan)
    data['ROCE'] = np.nan
    data['FCFY'] = (stock_info.get('freeCashflow', 0) / stock_info.get('marketCap', 1)) * 100
    data['CR'] = stock_info.get('currentRatio', np.nan)
    data['QR'] = stock_info.get('quickRatio', np.nan)
    data['Asset TR'] = np.nan
    data['DY'] = stock_info.get('dividendYield', np.nan) * 100
    data['Beta'] = stock_info.get('beta', np.nan)
    data['52w Low'] = stock_info.get('fiftyTwoWeekLow', np.nan)
    data['52w High'] = stock_info.get('fiftyTwoWeekHigh', np.nan)
    return data

# Function for Bollinger Bands calculation
def calculate_bollinger_bands(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['SMA'] + 2 * df['Close'].rolling(window=window).std()
    df['Lower Band'] = df['SMA'] - 2 * df['Close'].rolling(window=window).std()
    return df

# Stock Research Page Function
def stock_research_page():
    st.title("Stock Research")

    # User Input for Stock Ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", value="AAPL").upper()
    
    # Fetch Stock Data from Yahoo Finance
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period="5y")
    
    # Fetch fundamental data for the stock
    stock_info = fetch_stock_info(ticker)
    st.subheader(f"{ticker} - Fundamental Analysis")
    
    # Display fundamental data in a clear format
    st.write(f"**Company Name**: {stock_info.get('Name', 'N/A')}")
    st.write(f"**Sector**: {stock_info.get('Sector', 'N/A')}")
    st.write(f"**Market Cap**: ${stock_info.get('Market Cap', 'N/A'):,}")
    st.write(f"**Price to Earnings Ratio (PE fwd)**: {stock_info.get('PE fwd', 'N/A')}")
    st.write(f"**Price to Book Ratio (PB)**: {stock_info.get('PB', 'N/A')}")
    st.write(f"**Debt to Equity Ratio (D/E)**: {stock_info.get('D2E', 'N/A')}")
    st.write(f"**Return on Equity (ROE)**: {stock_info.get('ROE', 'N/A') * 100:.2f}%")
    st.write(f"**Free Cash Flow Yield (FCFY)**: {stock_info.get('FCFY', 'N/A'):.2f}%")
    st.write(f"**Dividend Yield (DY)**: {stock_info.get('DY', 'N/A'):.2f}%")
    st.write(f"**Beta**: {stock_info.get('Beta', 'N/A')}")
    st.write(f"**52-Week Range**: ${stock_info.get('52w Low', 'N/A')} - ${stock_info.get('52w High', 'N/A')}")
    
    # Technical Analysis Section
    st.subheader(f"{ticker} - Technical Analysis")
    
    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df)
    
    # Plot Closing Price and Bollinger Bands
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f'{ticker} Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Band', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Band', line=dict(color='red', dash='dash')))
    
    # Layout and display chart
    fig.update_layout(title=f'{ticker} - Closing Price with Bollinger Bands', 
                      xaxis_title='Date', yaxis_title='Price', width=1800, height=800)
    st.plotly_chart(fig)
    
    # Moving Average Convergence Divergence (MACD)
    st.subheader(f"{ticker} - MACD Indicator")
    short_window = 12
    long_window = 26
    signal_window = 9

    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

    # Plot MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], mode='lines', name='Signal Line', line=dict(color='orange', dash='dash')))
    fig_macd.update_layout(title=f'{ticker} - MACD Indicator', xaxis_title='Date', yaxis_title='MACD', width=1800, height=400)
    st.plotly_chart(fig_macd)
    
    # Stock Comparison (optional)
    st.subheader("Stock Comparison")
    comparison_ticker = st.text_input("Enter Another Stock Ticker for Comparison", value="MSFT").upper()
    comparison_data = yf.Ticker(comparison_ticker).history(period="5y")

    # Normalize prices for comparison (using percentage returns)
    df['Pct Returns'] = (df['Close'] / df['Close'].iloc[0]) * 100
    comparison_data['Pct Returns'] = (comparison_data['Close'] / comparison_data['Close'].iloc[0]) * 100
    
    # Plot Comparison
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=df.index, y=df['Pct Returns'], mode='lines', name=f'{ticker} % Returns'))
    fig_compare.add_trace(go.Scatter(x=comparison_data.index, y=comparison_data['Pct Returns'], mode='lines', name=f'{comparison_ticker} % Returns', line=dict(dash='dash')))
    fig_compare.update_layout(title=f'{ticker} vs {comparison_ticker} - Percentage Returns Comparison', 
                              xaxis_title='Date', yaxis_title='Percentage Return (%)', width=1800, height=800)
    st.plotly_chart(fig_compare)


####################---- Main App ----####################

# Add the Research page to the PAGES dictionary
PAGES = {
    "S&P 500 Stock Screener by Sector": screener_page,
    "Macro Indicator": macro_indicator_page,
    "Research": research_page,
    "Stock ReResearch": stock_research_page
}

# Function to render the appropriate page
def render_page(page):
    page()  # Call the function directly

# Select page
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Call the selected page
render_page(PAGES[selection])