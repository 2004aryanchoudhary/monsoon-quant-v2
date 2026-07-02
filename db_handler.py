import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "monsoon_production.db"

def init_db():
    """Creates the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_signals (
            date TEXT PRIMARY KEY,
            ticker TEXT,
            confidence REAL,
            forward_return REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_signal(date, ticker, confidence, forward_return=0.0):
    """Saves a new daily signal into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Insert or replace ensures we don't get duplicate errors for the same day
    cursor.execute('''
        INSERT OR REPLACE INTO trade_signals (date, ticker, confidence, forward_return)
        VALUES (?, ?, ?, ?)
    ''', (date, ticker, confidence, forward_return))
    
    conn.commit()
    conn.close()

def get_trade_history():
    """Fetches the trade history for the Streamlit dashboard."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trade_signals ORDER BY date DESC", conn)
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
    print("Production SQLite Database Initialized.")