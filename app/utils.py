import pandas as pd
import numpy as np

def create_input_dataframe(building_type, square_feet, date, horizon, base_temp):
    """
    Generates a dataframe for the selected horizon (Daily, Monthly, Annual).
    """
    start_time = pd.to_datetime(date)
    
    # 1. Determine Time Range based on Horizon
    if horizon == 'Day':
        periods = 24
    elif horizon == 'Month':
        periods = 24 * 30
    elif horizon == 'Year':
        periods = 24 * 365
    else:
        periods = 24
        
    timestamps = pd.date_range(start=start_time, periods=periods, freq='h')
    
    # 2. Initialize DataFrame
    df = pd.DataFrame({'timestamp': timestamps})
    
    # 3. Add Known User Inputs
    building_map = {
        'Education': 0, 'Entertainment': 1, 'Lodging/Residential': 2, 
        'Office': 3, 'Other': 4
    }
    df['primary_use'] = building_map.get(building_type, 0)
    df['square_feet'] = square_feet
    
    # 4. Simulate Weather (Daily + Seasonal)
    # Daily fluctuation (Day/Night)
    day_variation = 5 * np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    
    # Seasonal fluctuation (Summer/Winter) - Only matters for Monthly/Annual
    # Assumes peak temp in July (Month 7)
    seasonal_variation = 10 * np.cos(2 * np.pi * (df['timestamp'].dt.dayofyear - 196) / 365)
    
    df['air_temperature'] = base_temp + day_variation + seasonal_variation
    
    # 5. Feature Engineering (Cyclical Time)
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
    
    # 6. Improved Lag Generation
    # Instead of setting lag = current, we calculate rolling stats on the simulated weather
    # This prevents the model from seeing "flat" weather history
    df['temp_mean_lag_3h'] = df['air_temperature'].rolling(window=3, min_periods=1).mean()
    df['temp_std_lag_3h'] = df['air_temperature'].rolling(window=3, min_periods=1).std().fillna(0)
    df['temp_mean_lag_24h'] = df['air_temperature'].rolling(window=24, min_periods=1).mean()
    df['temp_std_lag_24h'] = df['air_temperature'].rolling(window=24, min_periods=1).std().fillna(0)

    # 7. Metadata Defaults (Dummy Fillers)
    df['building_id'] = 100 
    df['site_id'] = 0 
    df['meter'] = 0 
    df['year_built'] = 1960
    df['floor_count'] = 2
    
    # Weather Defaults
    df['cloud_coverage'] = 0
    df['dew_temperature'] = df['air_temperature'] - 2 
    df['precip_depth_1_hr'] = 0
    df['sea_level_pressure'] = 1013
    df['wind_direction'] = 0
    df['wind_speed'] = 0
    
    # 8. Final Column Ordering
    expected_cols = [
        'building_id', 'meter', 'site_id', 'primary_use', 'square_feet', 
        'year_built', 'floor_count', 'air_temperature', 'cloud_coverage', 
        'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 
        'wind_direction', 'wind_speed', 
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'temp_mean_lag_3h', 'temp_std_lag_3h',
        'temp_mean_lag_24h', 'temp_std_lag_24h'
    ]
    
    final_df = pd.DataFrame()
    for col in expected_cols:
        if col in df.columns:
            final_df[col] = df[col]
        else:
            final_df[col] = 0
            
    return final_df, timestamps