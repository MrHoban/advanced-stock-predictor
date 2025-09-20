import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Stock Market Predictor", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedStockPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
    @st.cache_data
    def fetch_stock_data(_self, symbol, period="1y"):
        """Fetch stock data and company information"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            
            # Get additional market data
            recommendations = stock.recommendations
            calendar = stock.calendar
            
            return data, info, recommendations, calendar
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None, None, None, None
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_%R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced features for machine learning"""
        df = self.calculate_technical_indicators(df)
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2d'] = df['Close'].pct_change(periods=2)
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Close_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        
        # Volatility features
        df['Volatility_5d'] = df['Close'].rolling(5).std()
        df['Volatility_20d'] = df['Close'].rolling(20).std()
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Trend features
        df['Trend_5d'] = np.where(df['Close'] > df['Close'].shift(5), 1, 0)
        df['Trend_20d'] = np.where(df['Close'] > df['Close'].shift(20), 1, 0)
        
        # Gap features
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Percent'] = df['Gap'] / df['Close'].shift(1) * 100
        
        # Target variables for different prediction horizons
        df['Target_1d'] = df['Close'].shift(-1)
        df['Target_5d'] = df['Close'].shift(-5)
        df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        return df
    
    def prepare_model_data(self, df, target_days=1):
        """Prepare data for machine learning models"""
        df_features = self.create_advanced_features(df.copy())
        
        # Select features for modeling
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'RSI', 'MACD', 'MACD_Signal', '%K', '%D',
            'Williams_%R', 'CCI', 'MFI', 'ATR',
            'BB_Position', 'BB_Width',
            'Price_Change', 'Price_Change_2d', 'Price_Change_5d',
            'Volume_Change', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Close_SMA20_Ratio', 'Volatility_5d', 'Volatility_20d',
            'Volume_Ratio', 'Trend_5d', 'Trend_20d',
            'Gap_Percent'
        ]
        
        # Remove features that don't exist or have too many NaN values
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        X = df_features[available_features].fillna(method='ffill').fillna(method='bfill')
        
        if target_days == 1:
            y = df_features['Target_1d']
        elif target_days == 5:
            y = df_features['Target_5d']
        else:
            y = df_features['Target_1d']
        
        # Remove rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y, available_features
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        # Split data (use last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                self.trained_models[name] = model
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
        
        # Select best model based on R² score
        if model_scores:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
            self.best_model = model_scores[best_model_name]['model']
            self.best_model_name = best_model_name
        
        return model_scores, X_test, y_test
    
    def train_single_model(self, X, y, model_name):
        """Train a single specified model"""
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_scores = {
            model_name: {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
        }
        
        self.best_model = model
        self.best_model_name = model_name
        self.trained_models[model_name] = model
        
        return model_scores, X_test, y_test
    
    def predict_future_prices(self, X, features, days=5):
        """Predict future prices for multiple days"""
        if self.best_model is None:
            return None
        
        predictions = []
        last_features = X.iloc[-1:].copy()
        
        for _ in range(days):
            pred = self.best_model.predict(last_features)[0]
            predictions.append(pred)
            
            # Update features for next prediction (simplified approach)
            # In practice, you'd want more sophisticated feature updating
            last_features = last_features.copy()
        
        return predictions

def create_candlestick_chart(data, title="Stock Price"):
    """Create an interactive candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=500,
        showlegend=False
    )
    
    return fig

def create_technical_analysis_chart(data):
    """Create comprehensive technical analysis chart"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Price & Moving Averages', 'RSI', 'Volume'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price and Moving Averages
        fig.add_trace(go.Candlestick(
            x=data.index, 
            open=data['Open'], 
            high=data['High'],
            low=data['Low'], 
            close=data['Close'], 
            name="Price"
        ), row=1, col=1)
        
        # Simple moving averages
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=sma_20,
            mode='lines', 
            name='SMA 20', 
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=sma_50,
            mode='lines', 
            name='SMA 50', 
            line=dict(color='red', width=1)
        ), row=1, col=1)
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=rsi,
            mode='lines', 
            name='RSI', 
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=data.index, 
            y=data['Volume'],
            name='Volume', 
            marker_color='lightblue'
        ), row=3, col=1)
        
        fig.update_layout(
            height=700,
            title="Technical Analysis",
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        return create_candlestick_chart(data, "Technical Analysis (Fallback)")

def display_stock_info(info):
    """Display comprehensive stock information"""
    if not info:
        return
    
    st.subheader("📊 Company Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Basic Info**")
        st.write(f"**Company:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Country:** {info.get('country', 'N/A')}")
    
    with col2:
        st.write("**Financial Metrics**")
        market_cap = info.get('marketCap')
        if market_cap:
            st.write(f"**Market Cap:** ${market_cap/1e9:.2f}B")
        
        pe_ratio = info.get('trailingPE')
        if pe_ratio:
            st.write(f"**P/E Ratio:** {pe_ratio:.2f}")
        
        dividend_yield = info.get('dividendYield')
        if dividend_yield:
            st.write(f"**Dividend Yield:** {dividend_yield*100:.2f}%")
        
        beta = info.get('beta')
        if beta:
            st.write(f"**Beta:** {beta:.2f}")
    
    with col3:
        st.write("**Price Metrics**")
        st.write(f"**52W High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.write(f"**52W Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}")
        st.write(f"**50D MA:** ${info.get('fiftyDayAverage', 'N/A')}")
        st.write(f"**200D MA:** ${info.get('twoHundredDayAverage', 'N/A')}")

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Advanced Stock Market Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize predictor
    predictor = AdvancedStockPredictor()
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'info' not in st.session_state:
        st.session_state.info = None
    if 'symbol' not in st.session_state:
        st.session_state.symbol = None
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    st.sidebar.markdown("---")
    
    # Input parameters
    symbol = st.sidebar.text_input("📊 Stock Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, GOOGL, TSLA)").upper()
    
    period_options = {
        "6 months": "6mo",
        "1 year": "1y", 
        "2 years": "2y",
        "5 years": "5y",
        "10 years": "10y"
    }
    period_display = st.sidebar.selectbox("📅 Time Period", list(period_options.keys()), index=1)
    period = period_options[period_display]
    
    prediction_days = st.sidebar.slider("🔮 Prediction Days", 1, 30, 5)
    
    model_type = st.sidebar.selectbox(
        "🤖 Model Selection",
        ["Auto (Best Performance)", "Random Forest", "Linear Regression", "Support Vector Regression"]
    )
    
    st.sidebar.markdown("---")
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        predictor = AdvancedStockPredictor()
        
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            data, info, recommendations, calendar = predictor.fetch_stock_data(symbol, period)
        
        # Store in session state
        st.session_state.data = data
        st.session_state.info = info
        st.session_state.symbol = symbol
    
    # Display data if available (outside button block)
    if st.session_state.data is not None and len(st.session_state.data) > 50:
        data = st.session_state.data
        info = st.session_state.info
        symbol = st.session_state.symbol
        
        st.success(f"✅ Successfully loaded {len(data)} days of data for {symbol}")
        
        # Display current metrics
        st.subheader("📊 Current Market Data")
        
        current_price = data['Close'][-1]
        prev_price = data['Close'][-2]
        daily_change = current_price - prev_price
        daily_change_pct = (daily_change / prev_price) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("💰 Current Price", f"${current_price:.2f}", f"${daily_change:.2f} ({daily_change_pct:.2f}%)")
        
        with col2:
            volume = data['Volume'][-1]
            avg_volume = data['Volume'].rolling(20).mean()[-1]
            volume_change = ((volume - avg_volume) / avg_volume) * 100
            st.metric("📊 Volume", f"{volume:,.0f}", f"{volume_change:.1f}% vs 20D avg")
        
        with col3:
            high_52w = data['High'].rolling(252).max()[-1]
            low_52w = data['Low'].rolling(252).min()[-1]
            st.metric("📈 52W High", f"${high_52w:.2f}")
        
        with col4:
            st.metric("📉 52W Low", f"${low_52w:.2f}")
        
        with col5:
            if info and 'marketCap' in info:
                market_cap = info['marketCap'] / 1e9
                st.metric("🏢 Market Cap", f"${market_cap:.1f}B")
        
        # Company information
        if info:
            display_stock_info(info)
        
        st.markdown("---")
        
        # Charts section - NOW OUTSIDE BUTTON BLOCK
        st.subheader("📈 Technical Analysis Charts")
        
        # Chart selection
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Candlestick Chart", "Technical Analysis"]
        )
        
        try:
            if chart_type == "Candlestick Chart":
                fig_candle = create_candlestick_chart(data.tail(100), f"{symbol} - Candlestick Chart")
                st.plotly_chart(fig_candle, width='stretch')
            
            elif chart_type == "Technical Analysis":
                fig_tech = create_technical_analysis_chart(data.tail(100))
                st.plotly_chart(fig_tech, width='stretch')
        
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("Falling back to candlestick chart...")
            fig_fallback = create_candlestick_chart(data.tail(100), f"{symbol} - Candlestick Chart")
            st.plotly_chart(fig_fallback, width='stretch')
        
        st.markdown("---")
        
        # Prepare data and train models
        with st.spinner("Training machine learning models..."):
            X, y, features = predictor.prepare_model_data(data)
            
            if len(X) > 100:  # Ensure sufficient data
                if model_type == "Auto (Best Performance)":
                    model_scores, X_test, y_test = predictor.train_models(X, y)
                else:
                    # Train only the selected model
                    model_scores, X_test, y_test = predictor.train_single_model(X, y, model_type)
                
                # Model performance
                st.subheader("🎯 Model Performance")
                
                if model_scores:
                    performance_df = pd.DataFrame({
                        'Model': list(model_scores.keys()),
                        'R² Score': [scores['r2'] for scores in model_scores.values()],
                        'MSE': [scores['mse'] for scores in model_scores.values()],
                        'MAE': [scores['mae'] for scores in model_scores.values()]
                    }).round(4)
                    
                    st.dataframe(performance_df, width='stretch')
                    
                    # Best model info
                    best_score = max(model_scores.values(), key=lambda x: x['r2'])
                    st.info(f"🏆 Best Model: **{predictor.best_model_name}** (R² Score: {best_score['r2']:.4f})")
                    
                    # Future price predictions
                    st.markdown("---")
                    st.subheader("🔮 Future Price Predictions")
                    
                    future_predictions = predictor.predict_future_prices(X, features, prediction_days)
                    if future_predictions:
                        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.write(f"**{prediction_days}-Day Price Forecast**")
                            for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
                                st.write(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            # Create prediction chart
                            fig_pred = go.Figure()
                            
                            # Historical prices (last 30 days)
                            historical_data = data.tail(30)
                            fig_pred.add_trace(go.Scatter(
                                x=historical_data.index,
                                y=historical_data['Close'],
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Predicted prices
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_predictions,
                                mode='lines+markers',
                                name='Predicted Price',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            fig_pred.update_layout(
                                title=f"{symbol} - Price Prediction ({prediction_days} Days)",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Model Performance Chart (moved to separate section)
                    st.markdown("---")
                    st.subheader("🎯 Model Performance Chart")
                    
                    try:
                        # Actual vs Predicted chart
                        fig_perf = go.Figure()
                        
                        test_dates = X_test.index
                        
                        fig_perf.add_trace(go.Scatter(
                            x=test_dates,
                            y=y_test.values,
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_perf.add_trace(go.Scatter(
                            x=test_dates,
                            y=best_score['predictions'],
                            mode='lines',
                            name='Predicted Price',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig_perf.update_layout(
                            title=f"Model Performance - Actual vs Predicted ({predictor.best_model_name})",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_perf, width='stretch')
                    
                    except Exception as e:
                        st.error(f"Error creating performance chart: {str(e)}")
                    
                    # Feature importance (for tree-based models)
                    if hasattr(predictor.best_model, 'feature_importances_'):
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': predictor.best_model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 15 Most Important Features"
                        )
                        fig_importance.update_layout(height=500)
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Risk Analysis
                    st.markdown("---")
                    st.subheader("⚠️ Risk Analysis")
                    
                    # Calculate risk metrics
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                    max_drawdown = ((data['Close'] / data['Close'].cummax()) - 1).min()
                    
                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                    
                    with risk_col1:
                        st.metric("📊 Volatility (Annual)", f"{volatility*100:.2f}%")
                    
                    with risk_col2:
                        st.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    with risk_col3:
                        st.metric("📉 Max Drawdown", f"{max_drawdown*100:.2f}%")
                    
                    with risk_col4:
                        var_95 = np.percentile(returns, 5)
                        st.metric("⚠️ VaR (95%)", f"{var_95*100:.2f}%")
                    
                    # Risk interpretation
                    if volatility > 0.3:
                        st.warning("⚠️ High volatility stock - Higher risk and potential reward")
                    elif volatility < 0.15:
                        st.info("ℹ️ Low volatility stock - More stable but potentially lower returns")
                    else:
                        st.success("✅ Moderate volatility - Balanced risk profile")
                
                else:
                    st.error("❌ Failed to train models. Please try with a different stock or time period.")
            
            else:
                st.error("❌ Insufficient data for analysis. Please try a longer time period.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions. 
        Always consult with a financial advisor before making investment choices.</p>
        <p>Built with ❤️ using Streamlit, scikit-learn, and yfinance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
