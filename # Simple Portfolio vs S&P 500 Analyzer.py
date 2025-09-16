# Simple Portfolio vs S&P 500 Analyzer
# Compatible with Python 3.6+
# No complex type hints or advanced features

import warnings
warnings.filterwarnings('ignore')

def install_package(package_name):
    """Install package if not available"""
    import subprocess
    import sys
    try:
        __import__(package_name.replace('-', '_'))
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install required packages if needed
required_packages = ['yfinance', 'pandas', 'numpy', 'matplotlib']
for package in required_packages:
    install_package(package)

# Now import everything
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    print("✅ All packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please manually install: pip install yfinance pandas numpy matplotlib")
    exit()

def create_portfolio_analyzer():
    """Simple function-based portfolio analyzer"""
    
    print("="*60)
    print("PORTFOLIO VS S&P 500 ANALYZER")
    print("="*60)
    
    # Example portfolio - MODIFY THIS WITH YOUR STOCKS
    portfolio = {
        'AAPL': 0.3,    # 30% Apple
        'MSFT': 0.25,   # 25% Microsoft  
        'GOOGL': 0.2,   # 20% Google
        'NVDA': 0.15,   # 15% NVIDIA
        'TSLA': 0.1     # 10% Tesla
    }
    
    # Date range - MODIFY THESE DATES
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    print(f"Portfolio: {portfolio}")
    print(f"Period: {start_date} to {end_date}")
    print("-"*60)
    
    # Step 1: Fetch stock data
    print("📊 Fetching stock data...")
    stock_data = {}
    
    # Get individual stock data
    for symbol in portfolio.keys():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                stock_data[symbol] = hist['Close']
                print(f"✅ {symbol}: {len(hist)} days of data")
            else:
                print(f"❌ {symbol}: No data available")
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
    
    # Get S&P 500 data (using SPY ETF as proxy)
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(start=start_date, end=end_date)
        stock_data['SPY'] = spy_hist['Close']
        print(f"✅ SPY (S&P 500): {len(spy_hist)} days of data")
    except Exception as e:
        print(f"❌ SPY: Error - {str(e)}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(stock_data)
    df = df.dropna()  # Remove any missing data
    
    if df.empty:
        print("❌ No data available for analysis")
        return
    
    print(f"📈 Final dataset: {len(df)} days of complete data")
    
    # Step 2: Calculate returns
    print("\n💰 Calculating returns...")
    
    # Daily returns
    daily_returns = df.pct_change().dropna()
    
    # Portfolio weights
    weights = []
    portfolio_stocks = []
    for symbol, weight in portfolio.items():
        weights.append(weight)
        portfolio_stocks.append(symbol)
    
    # Calculate portfolio daily returns
    portfolio_daily_returns = []
    for i in range(len(daily_returns)):
        daily_portfolio_return = 0
        for j, symbol in enumerate(portfolio_stocks):
            if symbol in daily_returns.columns:
                daily_portfolio_return += weights[j] * daily_returns[symbol].iloc[i]
        portfolio_daily_returns.append(daily_portfolio_return)
    
    # Convert to pandas Series
    portfolio_returns = pd.Series(portfolio_daily_returns, index=daily_returns.index)
    spy_returns = daily_returns['SPY']
    
    # Cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    spy_cumulative = (1 + spy_returns).cumprod()
    
    # Step 3: Calculate metrics
    print("\n📊 Performance Metrics:")
    print("-"*40)
    
    # Total returns
    portfolio_total_return = portfolio_cumulative.iloc[-1] - 1
    spy_total_return = spy_cumulative.iloc[-1] - 1
    
    # Annualized returns (assuming 252 trading days per year)
    days = len(portfolio_returns)
    portfolio_annual_return = (1 + portfolio_total_return) ** (252/days) - 1
    spy_annual_return = (1 + spy_total_return) ** (252/days) - 1
    
    # Volatility (annualized)
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    spy_volatility = spy_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility > 0 else 0
    spy_sharpe = spy_annual_return / spy_volatility if spy_volatility > 0 else 0
    
    # Beta
    covariance = np.cov(portfolio_returns, spy_returns)[0][1]
    spy_variance = np.var(spy_returns)
    beta = covariance / spy_variance if spy_variance > 0 else 0
    
    # Alpha
    alpha = portfolio_annual_return - beta * spy_annual_return
    
    # Print results
    print(f"{'Metric':<20} {'Portfolio':<15} {'S&P 500':<15}")
    print("-"*50)
    print(f"{'Total Return:':<20} {portfolio_total_return:>13.2%} {spy_total_return:>13.2%}")
    print(f"{'Annual Return:':<20} {portfolio_annual_return:>13.2%} {spy_annual_return:>13.2%}")
    print(f"{'Volatility:':<20} {portfolio_volatility:>13.2%} {spy_volatility:>13.2%}")
    print(f"{'Sharpe Ratio:':<20} {portfolio_sharpe:>13.2f} {spy_sharpe:>13.2f}")
    print(f"{'Beta:':<20} {beta:>13.2f} {'1.00':>13}")
    print(f"{'Alpha:':<20} {alpha:>13.2%} {'0.00%':>13}")
    
    # Outperformance
    outperformance = portfolio_annual_return - spy_annual_return
    print(f"\n🎯 Portfolio vs S&P 500:")
    print(f"   Outperformance: {outperformance:.2%} annually")
    if outperformance > 0:
        print("   ✅ Your portfolio OUTPERFORMED the S&P 500!")
    else:
        print("   ❌ Your portfolio UNDERPERFORMED the S&P 500")
    
    # Step 4: Create visualizations
    print(f"\n📈 Creating charts...")
    
    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Portfolio vs S&P 500 Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    ax1.plot(portfolio_cumulative.index, (portfolio_cumulative - 1) * 100, 
             label='Portfolio', linewidth=2, color='blue')
    ax1.plot(spy_cumulative.index, (spy_cumulative - 1) * 100, 
             label='S&P 500', linewidth=2, color='red', alpha=0.7)
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily Returns Distribution
    ax2 = axes[0, 1]
    ax2.hist(portfolio_returns * 100, bins=30, alpha=0.7, label='Portfolio', 
             color='blue', density=True)
    ax2.hist(spy_returns * 100, bins=30, alpha=0.7, label='S&P 500', 
             color='red', density=True)
    ax2.set_title('Daily Returns Distribution')
    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly Returns Bar Chart
    ax3 = axes[1, 0]
    monthly_portfolio = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_spy = spy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    months = range(len(monthly_portfolio))
    width = 0.35
    ax3.bar([m - width/2 for m in months], monthly_portfolio * 100, width, 
            label='Portfolio', color='blue', alpha=0.7)
    ax3.bar([m + width/2 for m in months], monthly_spy * 100, width, 
            label='S&P 500', color='red', alpha=0.7)
    ax3.set_title('Monthly Returns')
    ax3.set_ylabel('Monthly Return (%)')
    ax3.set_xlabel('Month')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Scatter Plot (Portfolio vs S&P 500)
    ax4 = axes[1, 1]
    ax4.scatter(spy_returns * 100, portfolio_returns * 100, alpha=0.6, color='green')
    
    # Add regression line
    z = np.polyfit(spy_returns, portfolio_returns, 1)
    p = np.poly1d(z)
    ax4.plot(spy_returns * 100, p(spy_returns) * 100, "r--", alpha=0.8, linewidth=2)
    
    ax4.set_xlabel('S&P 500 Daily Return (%)')
    ax4.set_ylabel('Portfolio Daily Return (%)')
    ax4.set_title('Portfolio vs S&P 500 Correlation')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.05, 0.95, f'Beta: {beta:.2f}', transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Analysis complete!")
    return portfolio_returns, spy_returns

# Run the analysis
if __name__ == "__main__":
    try:
        portfolio_returns, spy_returns = create_portfolio_analyzer()
        
        print("""
        🔧 HOW TO MODIFY THIS CODE:
        
        1. Change the 'portfolio' dictionary with your stocks and weights:
           portfolio = {
               'AAPL': 0.4,   # 40% Apple
               'MSFT': 0.3,   # 30% Microsoft
               'GOOGL': 0.3   # 30% Google
           }
        
        2. Adjust the date range:
           start_date = '2022-01-01'
           end_date = '2024-12-31'
        
        3. Make sure weights sum to 1.0 (100%)
        
        🎯 This simple version works with Python 3.6+ and basic packages!
        """)
        
    except Exception as e:
        print(f"❌ Error running analysis: {str(e)}")
        print("Please check your internet connection and package installations.")