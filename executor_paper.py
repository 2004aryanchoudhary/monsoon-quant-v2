import sqlite3
import yfinance as yf
import time
from db_handler import save_paper_trade

# ==========================================
# PHASE 3: RISK MANAGEMENT & KILL SWITCHES
# ==========================================
VIRTUAL_CAPITAL = 100000.00       # Starting paper account with Rs. 1 Lakh
MAX_RISK_PER_TRADE = 0.05         # Never risk more than 5% of capital on one trade
CONFIDENCE_THRESHOLD = 0.60       # The model's strict baseline

def check_kill_switches(confidence):
    """Safety checks before attempting execution."""
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"🛑 KILL SWITCH TRIPPED: Signal confidence ({confidence*100:.2f}%) is below {CONFIDENCE_THRESHOLD*100}% threshold.")
        return False
    return True

# ==========================================
# PHASE 1: VIRTUAL BROKER API (MOCKING)
# ==========================================
def fetch_live_price(ticker):
    """Simulates a broker API returning the current live Ask price."""
    print(f"📡 Requesting live market depth for {ticker} from Exchange...")
    time.sleep(1.5) # Simulate network latency
    try:
        # Using yfinance to grab the actual real-world closing price
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty:
            raise ValueError("No price data returned from Exchange.")
        live_price = float(data['Close'].iloc[-1])
        return live_price
    except Exception as e:
        print(f"⚠️ BROKER API ERROR: Could not fetch price. {e}")
        return None

# ==========================================
# PHASE 2: THE EXECUTION ENGINE
# ==========================================
def get_latest_signal():
    """Reads the Streamlit database to find what ML model wants to do today."""
    conn = sqlite3.connect("monsoon_production.db")
    cursor = conn.cursor()
    cursor.execute("SELECT date, ticker, confidence FROM trade_signals ORDER BY date DESC LIMIT 1")
    signal = cursor.fetchone()
    conn.close()
    return signal

def run_execution_cycle():
    print("\n" + "="*50)
    print("🤖 INITIALIZING PAPER TRADING EXECUTION SEQUENCE")
    print("="*50)
    
    # 1. Fetch Signal
    signal = get_latest_signal()
    if not signal:
        print("❌ No active signals found in the database. Aborting.")
        return
        
    signal_date, ticker, confidence = signal
    print(f"Target Asset: {ticker} | ML Confidence: {confidence*100:.2f}% | Generated: {signal_date}")
    
    # 2. Risk Check
    if not check_kill_switches(confidence):
        print("🛡️ Action: HOLD CASH. Execution sequence terminated.")
        return
        
    # 3. Get Live Price
    live_price = fetch_live_price(ticker)
    if live_price is None:
        print("❌ Execution aborted due to missing market data.")
        return
        
    # 4. Position Sizing Math
    max_capital_allocation = VIRTUAL_CAPITAL * MAX_RISK_PER_TRADE
    quantity = int(max_capital_allocation // live_price)
    total_value = quantity * live_price
    
    if quantity <= 0:
        print("❌ Insufficient capital to buy even 1 share. Aborting.")
        return
        
    # 5. Virtual Execution
    print(f"\n⚡ ROUTING MARKET ORDER:")
    print(f" -> Action: BUY")
    print(f" -> Ticker: {ticker}")
    print(f" -> Quantity: {quantity} shares")
    print(f" -> Fill Price: ₹{live_price:.2f}")
    print(f" -> Total Capital Deployed: ₹{total_value:.2f}")
    
    # 6. Log to Ledger
    save_paper_trade(ticker, "BUY", quantity, live_price, total_value)
    print("\n✅ VIRTUAL EXECUTION SUCCESSFUL. Ledger updated.")
    print("="*50)

if __name__ == "__main__":
    run_execution_cycle()