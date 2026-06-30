import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def fetch_national_weather_basket(locations: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical weather data for multiple agricultural hubs
    and aggregates them into a single National Crop Stress Index.
    """
    print(f"Fetching National Weather Basket from {start_date} to {end_date}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    all_city_data = []
    
    for city, coords in locations.items():
        print(f" -> Pulling data for {city} (Lat: {coords['lat']}, Lon: {coords['lon']})...")
        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["precipitation_sum", "temperature_2m_max"],
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        daily_data = data['daily']
        
        # Create a dataframe for this specific city
        df = pd.DataFrame({
            'Date': pd.to_datetime(daily_data['time']),
            'precip_mm': daily_data['precipitation_sum'],
            'max_temp_c': daily_data['temperature_2m_max']
        })
        
        all_city_data.append(df)
        # 1-second pause to be polite to the free API and avoid rate limits
        time.sleep(1) 
        
    # Combine all city dataframes vertically
    master_df = pd.concat(all_city_data)
    
    # Aggregate (Average) the weather across all cities for each day
    # This creates our smoothed National Basket signal
    national_index = master_df.groupby('Date').mean()
    
    print(f"\nSuccessfully aggregated {len(locations)} cities into a national index.")
    return national_index

if __name__ == "__main__":
    # Our critical agricultural basket representing broad market exposure
    AG_BASKET = {
        "Sikar_Rajasthan": {"lat": 27.6094, "lon": 75.1398},
        "Jalna_Maharashtra": {"lat": 19.8297, "lon": 75.8800},
        "Guntur_Andhra_Pradesh": {"lat": 16.3067, "lon": 80.4365}
    }
    
    START = "2018-01-01"
    # Offset by 7 days to ensure we only request finalized historical data
    END = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        basket_data = fetch_national_weather_basket(AG_BASKET, START, END)
        
        # Overwrite the old single-city file with our new national index
        output_filename = "data/raw/raw_weather_data.csv"
        basket_data.to_csv(output_filename)
        
        print(f"Saved National Weather Basket to {output_filename}")
        print("\nFirst 5 rows of aggregated national data:")
        print(basket_data.head())
        
    except Exception as e:
        print(f"Execution failed: {e}")