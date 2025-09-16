import pandas as pd
import yfinance as yf
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
qs.extend_pandas()

class PortfolioAnalyzer:
    def __init__(self, portfolio_stocks, portfolio_weights=None, start_date=None, end_date=None):
        """Initialize the portfolio analyzer using QuantStats."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.start_date = start_date
        self.end_date = end_date
        if isinstance(portfolio_stocks, dict):
            self.stocks = list(portfolio_stocks.keys())
            self.weights = list(portfolio_stocks.values())
        else:
            self.stocks = portfolio_stocks
            if portfolio_weights:
                self.weights = portfolio_weights
            else:
                self.weights = [1/len(self.stocks)] * len(self.stocks)
        self.weights = np.array(self.weights) / sum(self.weights)
        print(f"Portfolio: {dict(zip(self.stocks, self.weights))}")
        print(f"Analysis period: {start_date} to {end_date}")

    def fetch_data(self):
        """Fetch stock data and S&P 500 data."""
        print("Fetching data...")
        portfolio_data = {}
        for stock in self.stocks:
            try:
                ticker = yf.Ticker(stock)
                data = ticker.history(start=self.start_date, end=self.end_date)
                portfolio_data[stock] = data['Close']
                print(f"✓ {stock} data fetched")
            except Exception as e:
                print(f"✗ Error fetching {stock}: {e}")
        try:
            sp500 = yf.Ticker("SPY")
            sp500_data = sp500.history(start=self.start_date, end=self.end_date)
            self.benchmark_data = sp500_data['Close']
            print("✓ S&P 500 (SPY) data fetched")
        except Exception as e:
            print(f"✗ Error fetching S&P 500: {e}")
        self.stock_data = pd.DataFrame(portfolio_data).dropna()
        self.stock_data.index = self.stock_data.index.tz_localize(None)
        self.benchmark_data.index = self.benchmark_data.index.tz_localize(None)
        return self.stock_data, self.benchmark_data

    def calculate_portfolio_returns(self):
        """Calculate portfolio returns using weights."""
        stock_returns = self.stock_data.pct_change().dropna()
        self.portfolio_returns = (stock_returns * self.weights).sum(axis=1)
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()
        common_dates = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        self.portfolio_returns = self.portfolio_returns.loc[common_dates]
        self.benchmark_returns = self.benchmark_returns.loc[common_dates]
        return self.portfolio_returns, self.benchmark_returns

    def generate_quantstats_report(self, save_html=False):
        """Generate comprehensive QuantStats report."""
        print("\n" + "="*60)
        print("GENERATING QUANTSTATS ANALYSIS")
        print("="*60)
        if save_html:
            qs.reports.html(self.portfolio_returns, 
                            benchmark=self.benchmark_returns,
                            output='portfolio_report.html',
                            title='Portfolio vs S&P 500 Analysis')
            print("📄 Full HTML report saved as 'portfolio_report.html'")
        print("\n📊 PORTFOLIO PERFORMANCE METRICS:")
        print("-" * 40)
        print("\n📊 FULL QUANTSTATS METRICS TABLE:")
        qs.reports.metrics(self.portfolio_returns, 
                          benchmark=self.benchmark_returns,
                          display=True)

    def plot_daily_cumulative_returns(self):
        """Plot daily cumulative returns of the portfolio and S&P 500 benchmark."""
        if not hasattr(self, 'portfolio_returns') or not hasattr(self, 'benchmark_returns'):
            print("Portfolio or benchmark returns not calculated yet. Calculating now...")
            self.calculate_portfolio_returns()
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod() - 1
        cumulative_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_portfolio.index, cumulative_portfolio * 100, label='Portfolio', color='blue')
        plt.plot(cumulative_benchmark.index, cumulative_benchmark * 100, label='S&P 500', color='red', alpha=0.7)
        plt.title('Daily Cumulative Returns: Portfolio vs S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def create_comparison_plots(self):
        """Create comparison visualizations."""
        plt.style.use('default')
        qs.extend_pandas()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio vs S&P 500 Analysis (QuantStats)', fontsize=16, fontweight='bold')
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod() - 1
        cumulative_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        ax1.plot(cumulative_portfolio.index, cumulative_portfolio * 100, 
                label='Portfolio', linewidth=2, color='blue')
        ax1.plot(cumulative_benchmark.index, cumulative_benchmark * 100, 
                label='S&P 500', linewidth=2, color='red', alpha=0.7)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 2. Rolling Sharpe Ratio
        ax2 = axes[0, 1]
        rolling_sharpe_port = qs.stats.rolling_sharpe(self.portfolio_returns)
        rolling_sharpe_bench = qs.stats.rolling_sharpe(self.benchmark_returns)
        ax2.plot(rolling_sharpe_port.index, rolling_sharpe_port, 
                label='Portfolio', color='blue')
        ax2.plot(rolling_sharpe_bench.index, rolling_sharpe_bench, 
                label='S&P 500', color='red', alpha=0.7)
        ax2.set_title('Rolling 6-Month Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # 3. Drawdown Comparison
        ax3 = axes[1, 0]
        dd_portfolio = qs.stats.to_drawdown_series(self.portfolio_returns)
        dd_benchmark = qs.stats.to_drawdown_series(self.benchmark_returns)
        ax3.fill_between(dd_portfolio.index, dd_portfolio * 100, 0, 
                        alpha=0.7, color='blue', label='Portfolio')
        ax3.fill_between(dd_benchmark.index, dd_benchmark * 100, 0, 
                        alpha=0.5, color='red', label='S&P 500')
        ax3.set_title('Drawdown Comparison')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # 4. Monthly Returns Heatmap (Portfolio only)
        ax4 = axes[1, 1]
        monthly_returns = qs.stats.monthly_returns(self.portfolio_returns)
        monthly_rets = self.portfolio_returns.resample('M').apply(qs.stats.comp)
        monthly_rets.plot(kind='bar', ax=ax4, color='blue', alpha=0.7)
        ax4.set_title('Portfolio Monthly Returns')
        ax4.set_ylabel('Monthly Return')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.tight_layout()

    def quick_analysis(self):
        """Quick analysis using QuantStats built-in functions."""
        print("\n🚀 QUICK QUANTSTATS ANALYSIS")
        print("=" * 50)
        qs.reports.metrics(self.portfolio_returns, 
                          benchmark=self.benchmark_returns, 
                          display=True)

    def run_analysis(self, generate_html=False):
        """Run the complete analysis."""
        try:
            self.fetch_data()
            if self.stock_data.empty:
                print("No data available for analysis")
                return
            self.calculate_portfolio_returns()
            self.quick_analysis()
            self.generate_quantstats_report(save_html=generate_html)
            self.create_comparison_plots()
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None

# Example ISIN to ticker mapping (add your own as needed)
isin_to_ticker = {
    "US0378331005": "AAPL",  # Apple
    "US5949181045": "MSFT",  # Microsoft
    "US02079K3059": "GOOGL",  # Alphabet (Google)
    "US67066G1040": "NVDA",  # NVIDIA
    "US88160R1014": "TSLA",  # Tesla
    # Add more ISINs and tickers as needed
}

# Example usage with ISINs
if __name__ == "__main__":
    tech_portfolio_isin = {
        "US0378331005": 0.25,
        "US5949181045": 0.25,
        "US02079K3059": 0.20,
        "US67066G1040": 0.15,
        "US88160R1014": 0.15
    }
    # Convert ISINs to tickers for PortfolioAnalyzer
    tech_portfolio = {isin_to_ticker[isin]: weight for isin, weight in tech_portfolio_isin.items()}
    analyzer = PortfolioAnalyzer(
        portfolio_stocks=tech_portfolio,
        start_date='2024-09-14',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    analyzer.run_analysis(generate_html=False)
    analyzer.plot_daily_cumulative_returns()
    #analyzer.create_comparison_plots()
    import matplotlib.pyplot as plt
    plt.show(block=True)
    input("Press Enter to close the plots...")