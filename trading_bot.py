import schedule
import time
import subprocess
from datetime import datetime

def execute_daily_pipeline():
    print(f"\n[{datetime.now()}] ⏰ 3:15 PM Triggered. Initiating pipeline...")

    # 1. Run the ML Pipeline to generate today's signal
    print("-> Running Alpha Generation...")
    subprocess.run(["uv", "run", "main.py", "--skip-download"])

    # 2. Run the Paper Broker to execute the trade
    print("-> Running Order Execution...")
    subprocess.run(["uv", "run", "executor_paper.py"])

    print(f"[{datetime.now()}] ✅ Daily sequence complete. Going back to sleep.")

# Schedule the job for 3:15 PM (15:15 in 24-hour time) everyday
schedule.every().day.at("15:15").do(execute_daily_pipeline)

print("🤖 Trading Daemon initialized. Waiting for 3:15 PM...")

# The infinite loop that keeps the script alive
while True:
    schedule.run_pending()
    time.sleep(60) # Check the clock every 60 seconds