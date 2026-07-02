import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "monsoon_production.db"

def init_db():
    """Creates the database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table 1: The ML Signal Generation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_signals (
            date TEXT PRIMARY KEY,
            ticker TEXT,
            confidence REAL,
            forward_return REAL
        )
    ''')
    
    # Table 2: The Paper Trading Execution Ledger
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_ledger (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action TEXT,
            quantity INTEGER,
            fill_price REAL,
            total_value REAL,
            status TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_signal(date, ticker, confidence, forward_return=0.0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO trade_signals (date, ticker, confidence, forward_return)
        VALUES (?, ?, ?, ?)
    ''', (date, ticker, confidence, forward_return))
    conn.commit()
    conn.close()

def get_trade_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trade_signals ORDER BY date DESC", conn)
    conn.close()
    return df

def save_paper_trade(ticker, action, quantity, fill_price, total_value, status="FILLED"):
    """Logs a successfully executed virtual trade."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO paper_ledger (timestamp, ticker, action, quantity, fill_price, total_value, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, ticker, action, quantity, fill_price, total_value, status))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Production & Paper Trading Database Initialized.")

def get_paper_ledger():
    """Fetches the executed paper trades for the Streamlit dashboard."""
    conn = sqlite3.connect(DB_PATH)
    # Check if table exists to avoid errors on first run
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_ledger'")
    if cursor.fetchone() is None:
        conn.close()
        return pd.DataFrame()
        
    df = pd.read_sql_query("SELECT * FROM paper_ledger ORDER BY timestamp DESC", conn)
    conn.close()
    return df