# API Integration Guide for Utilities Dashboard

This guide outlines how to call the OpenEI, EIA, and NREL APIs to fetch data for the utilities dashboard. Each section corresponds to a specific dashboard element and provides implementation details for retrieving the required data.

## Table of Contents

- [API Keys and Authentication](#api-keys-and-authentication)
- [Zonal Price Data](#zonal-price-data)
- [Load and Demand Data](#load-and-demand-data)
- [Generation Mix](#generation-mix)
- [Transmission Data](#transmission-data)
- [Reserve Margins](#reserve-margins)
- [Error Handling and Rate Limiting](#error-handling-and-rate-limiting)
- [Data Update Frequency](#data-update-frequency)

## API Keys and Authentication

### EIA API
```python
EIA_API_KEY = "your-eia-api-key"
EIA_BASE_URL = "https://api.eia.gov/v2"
```

### OpenEI API
```python
OPENEI_API_KEY = "your-openei-api-key"
OPENEI_BASE_URL = "https://api.openei.org/utility_rates"
```

### NREL API
```python
NREL_API_KEY = "your-nrel-api-key"
NREL_BASE_URL = "https://developer.nrel.gov/api/utility_rates/v3"
```

## Zonal Price Data

### Locational Marginal Prices (LMPs) for each zone

#### EIA API

The EIA API provides real-time and historical LMP data for various ISOs and balancing authorities.

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_lmp_data(zone_id, start_date, end_date):
    """
    Fetches LMP data for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier (e.g., 'US-FLA-FPC')
        start_date (str or datetime): Start date/time (if string, format 'YYYY-MM-DD')
        end_date (str or datetime): End date/time (if string, format 'YYYY-MM-DD')
        
    Returns:
        DataFrame with LMP data
    """
    # Convert datetime objects to string format if needed
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        'US-SE-SOCO': 'SOCO',
        # Add mappings for all zones
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    if not ba_code:
        raise ValueError(f"No balancing authority mapping for zone {zone_id}")
    
    url = f"{EIA_BASE_URL}/electricity/rto/lmp/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": "value",
        "facets[balancingauthority][]": ba_code,
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch LMP data: {response.status_code}, {response.text}")
```

### Day-ahead prices for comparison

```python
def get_day_ahead_prices(zone_id, date):
    """
    Fetches day-ahead LMP prices for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        date (str or datetime): Target date (if string, format 'YYYY-MM-DD')
        
    Returns:
        DataFrame with day-ahead price data
    """
    # Convert datetime object to string format if needed
    if isinstance(date, datetime):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = date
    
    # Similar implementation as get_lmp_data but with appropriate parameters
    url = f"{EIA_BASE_URL}/electricity/rto/damlmp/data"
    
    # Convert zone_id to applicable BA code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    # Calculate end date (next day)
    if isinstance(date, datetime):
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        end_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": "value",
        "facets[balancingauthority][]": ba_code,
        "start": date_str,
        "end": end_date,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch day-ahead prices: {response.status_code}, {response.text}")
```

### Historical price trends

```python
def get_historical_price_trends(zone_id, start_date, end_date, frequency='monthly'):
    """
    Fetches historical price trends for analysis
    
    Parameters:
        zone_id (str): Zone identifier
        start_date (str or datetime): Start date/time (if string, format 'YYYY-MM-DD')
        end_date (str or datetime): End date/time (if string, format 'YYYY-MM-DD')
        frequency (str): Data frequency ('hourly', 'daily', 'monthly')
        
    Returns:
        DataFrame with historical price data
    """
    # Convert datetime objects to string format if needed
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Map zone_id to EIA region/BA code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    url = f"{EIA_BASE_URL}/electricity/retail-sales/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": frequency,
        "data": ["price", "revenue", "sales"],
        "facets[stateid][]": ba_code[:2],  # State code from BA
        "start": start_date,
        "end": end_date,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch historical price trends: {response.status_code}, {response.text}")
```

### OpenEI Utility Rate Database

For more detailed utility rate information:

```python
def get_openei_utility_rates(state, utility_name=None):
    """
    Fetches utility rate data from OpenEI
    
    Parameters:
        state (str): State abbreviation (e.g., 'FL', 'GA')
        utility_name (str, optional): Name of utility
        
    Returns:
        List of utility rate data
    """
    url = f"{OPENEI_BASE_URL}/v3"
    params = {
        "api_key": OPENEI_API_KEY,
        "format": "json",
        "version": 3,
        "getdata": "true",
        "address": state,
        "limit": 100
    }
    
    if utility_name:
        params["eia"] = utility_name
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get('items', [])
    else:
        raise Exception(f"Failed to fetch OpenEI utility rates: {response.status_code}, {response.text}")
```

### NREL Utility Rates API (Fallback)

```python
def get_nrel_utility_rates(lat, lon):
    """
    Fetches utility rate data from NREL API based on geographic coordinates
    
    Parameters:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        Dictionary with utility rate data
    """
    url = f"{NREL_BASE_URL}/rates"
    params = {
        "api_key": NREL_API_KEY,
        "lat": lat,
        "lon": lon,
        "format": "json"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch NREL utility rates: {response.status_code}, {response.text}")
```

## Load and Demand Data

### Current system load for each zone

```python
def get_current_load(zone_id, lookback_hours=24):
    """
    Fetches current system load for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        lookback_hours (int): Number of hours to look back from current time
        
    Returns:
        DataFrame with current load data
    """
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)
    
    url = f"{EIA_BASE_URL}/electricity/rto/region-data/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": ["D"],  # D is demand/load
        "facets[respondent][]": ba_code,
        "start": start_time.strftime('%Y-%m-%d'),
        "end": end_time.strftime('%Y-%m-%d'),
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch current load: {response.status_code}, {response.text}")
```

### Forecasted load (day-ahead and week-ahead)

```python
def get_forecasted_load(zone_id, start_time=None, forecast_type='day-ahead'):
    """
    Fetches forecasted load for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        start_time (datetime, optional): Start time for forecast (defaults to current time)
        forecast_type (str): 'day-ahead' or 'week-ahead'
        
    Returns:
        DataFrame with forecasted load data
    """
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    # Set time range based on forecast type
    if start_time is None:
        start_time = datetime.now()
        
    if forecast_type == 'day-ahead':
        end_time = start_time + timedelta(days=1)
    else:  # week-ahead
        end_time = start_time + timedelta(days=7)
    
    url = f"{EIA_BASE_URL}/electricity/rto/region-data-hourly-forecast/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": ["DF"],  # DF is demand forecast
        "facets[respondent][]": ba_code,
        "start": start_time.strftime('%Y-%m-%d'),
        "end": end_time.strftime('%Y-%m-%d'),
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch forecasted load: {response.status_code}, {response.text}")
```

### Historical load for comparison

```python
def get_historical_load(zone_id, start_date, end_date, frequency='hourly'):
    """
    Fetches historical load data for comparison
    
    Parameters:
        zone_id (str): Zone identifier
        start_date (str or datetime): Start date/time (if string, format 'YYYY-MM-DD')
        end_date (str or datetime): End date/time (if string, format 'YYYY-MM-DD')
        frequency (str): Data frequency ('hourly', 'daily', 'monthly')
        
    Returns:
        DataFrame with historical load data
    """
    # Convert datetime objects to string format if needed
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    url = f"{EIA_BASE_URL}/electricity/rto/region-data/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": frequency,
        "data": ["D"],
        "facets[respondent][]": ba_code,
        "start": start_date,
        "end": end_date,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch historical load: {response.status_code}, {response.text}")
```

### Peak load records

```python
def get_peak_load_records(zone_id, start_date=None, end_date=None, year=None):
    """
    Fetches peak load records for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        start_date (datetime, optional): Start date for analysis
        end_date (datetime, optional): End date for analysis
        year (int, optional): Year to filter records (alternative to start/end dates)
        
    Returns:
        DataFrame with peak load records
    """
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    # Set date range for the query
    if year:
        start_date_str = f"{year}-01-01"
        end_date_str = f"{year}-12-31"
    elif start_date and end_date:
        # Use provided date range
        start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
    else:
        # Default to last 5 years
        end_date_str = datetime.now().strftime('%Y-%m-%d')
        start_date_str = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Fetch hourly data for the period
    historical_data = get_historical_load(zone_id, start_date_str, end_date_str)
    
    # Process to find peak values
    if not historical_data.empty:
        # Group by month and find maximum load
        if 'value' in historical_data.columns and 'period' in historical_data.columns:
            historical_data['period'] = pd.to_datetime(historical_data['period'])
            historical_data['month'] = historical_data['period'].dt.to_period('M')
            monthly_peaks = historical_data.groupby('month')['value'].max().reset_index()
            return monthly_peaks
    
    return pd.DataFrame()  # Return empty DataFrame if data processing fails
```

## Generation Mix

### Current generation by fuel type for each zone

```python
def get_generation_mix(zone_id, start_time=None, end_time=None):
    """
    Fetches current generation mix by fuel type for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        start_time (datetime, optional): Start time (defaults to 24 hours ago)
        end_time (datetime, optional): End time (defaults to current time)
        
    Returns:
        DataFrame with generation mix data
    """
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    # Set default time range if not provided
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(hours=24)
    
    # Convert to string format
    start_date_str = start_time.strftime('%Y-%m-%d') if isinstance(start_time, datetime) else start_time
    end_date_str = end_time.strftime('%Y-%m-%d') if isinstance(end_time, datetime) else end_time
    
    url = f"{EIA_BASE_URL}/electricity/rto/fuel-type-data/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": ["NG"],  # NG is net generation
        "facets[respondent][]": ba_code,
        "start": start_date_str,
        "end": end_date_str,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch generation mix: {response.status_code}, {response.text}")
```

### Renewable vs conventional generation percentages

```python
def get_renewable_vs_conventional(zone_id):
    """
    Calculates renewable vs conventional generation percentages
    
    Parameters:
        zone_id (str): Zone identifier
        
    Returns:
        Dictionary with renewable and conventional percentages
    """
    # Get generation mix data
    gen_mix = get_generation_mix(zone_id)
    
    if gen_mix.empty:
        return {"renewable": 0, "conventional": 0}
    
    # Define renewable fuel types
    renewable_fuels = ['SUN', 'WND', 'WAT', 'GEO', 'BIO']  # Solar, Wind, Hydro, Geothermal, Biomass
    
    # Calculate totals if necessary columns exist
    if 'fueltype' in gen_mix.columns and 'value' in gen_mix.columns:
        total_generation = gen_mix['value'].sum()
        renewable_generation = gen_mix[gen_mix['fueltype'].isin(renewable_fuels)]['value'].sum()
        
        if total_generation > 0:
            renewable_pct = (renewable_generation / total_generation) * 100
            conventional_pct = 100 - renewable_pct
            return {
                "renewable": round(renewable_pct, 2),
                "conventional": round(conventional_pct, 2)
            }
    
    return {"renewable": 0, "conventional": 0}
```

### Generation capacities

```python
def get_generation_capacities(zone_id, start_date=None, end_date=None):
    """
    Fetches generation capacities for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        start_date (datetime, optional): Start date (defaults to 30 days ago)
        end_date (datetime, optional): End date (defaults to current time)
        
    Returns:
        DataFrame with generation capacities by fuel type
    """
    # Map zone_id to state abbreviation
    zone_to_state_map = {
        'US-FLA-FPC': 'FL',
        'US-FLA-FPL': 'FL',
        'US-SE-SOCO': ['GA', 'AL', 'MS'],
        # Add other mappings
    }
    
    # Set default time range if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    # Convert to string format
    start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
    end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
    
    states = zone_to_state_map.get(zone_id)
    if isinstance(states, str):
        states = [states]
    
    # We'll need to aggregate data for all states in the balancing authority
    all_capacities = []
    
    for state in states:
        url = f"{EIA_BASE_URL}/electricity/operating-generator-capacity/data"
        params = {
            "api_key": EIA_API_KEY,
            "frequency": "monthly",
            "data": ["capacity"],
            "facets[stateid][]": state,
            "start": start_date_str,
            "end": end_date_str,
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get('response', {}).get('data', []))
            all_capacities.append(df)
    
    # Combine all state data
    if all_capacities:
        combined_capacity = pd.concat(all_capacities)
        # Group by fuel type and sum capacities
        if 'plantprimaryFuel' in combined_capacity.columns and 'capacity' in combined_capacity.columns:
            capacity_by_fuel = combined_capacity.groupby('plantprimaryFuel')['capacity'].sum().reset_index()
            return capacity_by_fuel
    
    return pd.DataFrame()  # Return empty DataFrame if data retrieval fails
```

## Transmission Data

### Interface flows between zones

```python
def get_interface_flows(from_zone_id, to_zone_id, start_time=None, end_time=None):
    """
    Fetches interface flows between two zones
    
    Parameters:
        from_zone_id (str): Source zone identifier
        to_zone_id (str): Destination zone identifier
        start_time (datetime, optional): Start time (defaults to 24 hours ago)
        end_time (datetime, optional): End time (defaults to current time)
        
    Returns:
        DataFrame with interface flow data
    """
    # Map zone_ids to EIA balancing authority codes
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    from_ba = zone_to_ba_map.get(from_zone_id)
    to_ba = zone_to_ba_map.get(to_zone_id)
    
    # Set default time range if not provided
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(hours=24)
    
    # Convert to string format
    start_date_str = start_time.strftime('%Y-%m-%d') if isinstance(start_time, datetime) else start_time
    end_date_str = end_time.strftime('%Y-%m-%d') if isinstance(end_time, datetime) else end_time
    
    url = f"{EIA_BASE_URL}/electricity/rto/interchange-data/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": ["TI"],  # TI is Tie Line
        "facets[respondent][]": from_ba,
        "facets[fromba][]": from_ba,
        "facets[toba][]": to_ba,
        "start": start_date_str,
        "end": end_date_str,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch interface flows: {response.status_code}, {response.text}")
```

### Transmission constraints/congestion

```python
def get_transmission_constraints(zone_id, start_time=None, end_time=None):
    """
    Fetches transmission constraints and congestion data
    
    Parameters:
        zone_id (str): Zone identifier
        start_time (datetime, optional): Start time (defaults to 3 days ago)
        end_time (datetime, optional): End time (defaults to current time)
        
    Returns:
        DataFrame with transmission constraint data
    """
    # Set default time range if not provided
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(days=3)
    
    # Convert to string format
    start_date_str = start_time.strftime('%Y-%m-%d') if isinstance(start_time, datetime) else start_time
    end_date_str = end_time.strftime('%Y-%m-%d') if isinstance(end_time, datetime) else end_time
    
    # This information may need to be derived from LMP data
    # Higher LMPs in certain areas can indicate transmission constraints
    
    # Get LMP data for the zone
    lmp_data = get_lmp_data(zone_id, start_date_str, end_date_str)
    
    # Filter for significant price differences that indicate congestion
    if 'value' in lmp_data.columns and 'period' in lmp_data.columns:
        lmp_data['period'] = pd.to_datetime(lmp_data['period'])
        
        # Calculate hourly average price
        hourly_avg = lmp_data.groupby(lmp_data['period'].dt.floor('H'))['value'].mean()
        
        # Calculate price deviation to identify potential congestion
        lmp_data['hour'] = lmp_data['period'].dt.floor('H')
        lmp_data = lmp_data.merge(hourly_avg.rename('avg_price'), left_on='hour', right_index=True)
        lmp_data['price_deviation'] = lmp_data['value'] - lmp_data['avg_price']
        
        # Filter for significant deviations
        congestion_indicators = lmp_data[abs(lmp_data['price_deviation']) > 10]  # $10 threshold
        
        return congestion_indicators[['period', 'value', 'price_deviation']]
    
    return pd.DataFrame()
```

### Available transmission capacity

```python
def get_available_transmission_capacity(from_zone_id, to_zone_id):
    """
    Estimates available transmission capacity between zones
    
    Parameters:
        from_zone_id (str): Source zone identifier
        to_zone_id (str): Destination zone identifier
        
    Returns:
        Dictionary with available capacity data
    """
    # This data is complex and may require multiple API calls
    # Here's a simplified approach using interface flow data
    
    # Get current flow data
    flow_data = get_interface_flows(from_zone_id, to_zone_id)
    
    # We'd need to compare this with the known transfer capability
    # This is often available through regional transmission organizations
    
    # For demonstration, let's assume a fixed capacity between zones
    # In a real implementation, this would come from another data source
    
    # Map zone pairs to estimated total capacity (MW)
    capacity_map = {
        ('US-FLA-FPC', 'US-FLA-FPL'): 3000,
        ('US-FLA-FPL', 'US-FLA-FPC'): 3000,
        # Add other zone pairs
    }
    
    zone_pair = (from_zone_id, to_zone_id)
    total_capacity = capacity_map.get(zone_pair, 0)
    
    # Calculate current flow (average of recent values)
    current_flow = 0
    if not flow_data.empty and 'value' in flow_data.columns:
        current_flow = flow_data['value'].mean()
    
    # Calculate available capacity
    available_capacity = max(0, total_capacity - abs(current_flow))
    
    return {
        "total_capacity_mw": total_capacity,
        "current_flow_mw": current_flow,
        "available_capacity_mw": available_capacity,
        "utilization_percentage": (abs(current_flow) / total_capacity * 100) if total_capacity > 0 else 0
    }
```

## Reserve Margins

### Operating reserves (spinning and non-spinning)

```python
def get_operating_reserves(zone_id, start_time=None, end_time=None):
    """
    Fetches operating reserves (spinning and non-spinning) for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        start_time (datetime, optional): Start time (defaults to 24 hours ago)
        end_time (datetime, optional): End time (defaults to current time)
        
    Returns:
        DataFrame with operating reserve data
    """
    # Map zone_id to EIA balancing authority code
    zone_to_ba_map = {
        'US-FLA-FPC': 'FPC',
        'US-FLA-FPL': 'FPL',
        # Add other mappings
    }
    
    ba_code = zone_to_ba_map.get(zone_id)
    
    # Set default time range if not provided
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(hours=24)
    
    # Convert to string format
    start_date_str = start_time.strftime('%Y-%m-%d') if isinstance(start_time, datetime) else start_time
    end_date_str = end_time.strftime('%Y-%m-%d') if isinstance(end_time, datetime) else end_time
    
    url = f"{EIA_BASE_URL}/electricity/rto/region-data/data"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data": ["SR", "NR"],  # SR is spinning reserves, NR is non-spinning reserves
        "facets[respondent][]": ba_code,
        "start": start_date_str,
        "end": end_date_str,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('response', {}).get('data', []))
        return df
    else:
        raise Exception(f"Failed to fetch operating reserves: {response.status_code}, {response.text}")
```

### Available capacity by zone

```python
def get_available_capacity(zone_id):
    """
    Calculates available capacity for a specific zone
    
    Parameters:
        zone_id (str): Zone identifier
        
    Returns:
        Dictionary with available capacity data
    """
    # We need both generation capacity and current load
    capacity_data = get_generation_capacities(zone_id)
    load_data = get_current_load(zone_id)
    
    # Calculate total capacity
    total_capacity = 0
    if not capacity_data.empty and 'capacity' in capacity_data.columns:
        total_capacity = capacity_data['capacity'].sum()
    
    # Get current load
    current_load = 0
    if not load_data.empty and 'value' in load_data.columns:
        current_load = load_data['value'].iloc[-1] if len(load_data) > 0 else 0
    
    # Calculate available capacity and reserve margin
    available_capacity = max(0, total_capacity - current_load)
    reserve_margin_pct = (available_capacity / current_load * 100) if current_load > 0 else 0
    
    return {
        "total_capacity_mw": total_capacity,
        "current_load_mw": current_load,
        "available_capacity_mw": available_capacity,
        "reserve_margin_percentage": reserve_margin_pct
    }
```

## Error Handling and Rate Limiting

To handle API errors and rate limits properly:

```python
class APIRateLimitError(Exception):
    """Exception raised when API rate limit is exceeded"""
    pass

class APIError(Exception):
    """Exception raised for general API errors"""
    pass

def handle_api_response(response, endpoint_name):
    """
    Handles API response and common errors
    
    Parameters:
        response: Requests response object
        endpoint_name (str): Name of the API endpoint for error messages
        
    Returns:
        JSON response if successful
        
    Raises:
        APIRateLimitError: If rate limit is exceeded
        APIError: For other API errors
    """
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        raise APIRateLimitError(f"Rate limit exceeded for {endpoint_name}")
    else:
        raise APIError(f"Error accessing {endpoint_name}: {response.status_code} - {response.text}")

# Example usage:
def api_call_with_retry(url, params, endpoint_name, max_retries=3, backoff_factor=2):
    """
    Makes API call with retry logic for rate limiting
    
    Parameters:
        url (str): API URL
        params (dict): API parameters
        endpoint_name (str): Name of endpoint for error reporting
        max_retries (int): Maximum number of retries
        backoff_factor (int): Multiplier for exponential backoff
        
    Returns:
        API response data
    """
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            return handle_api_response(response, endpoint_name)
        except APIRateLimitError:
            # Exponential backoff
            wait_time = backoff_factor ** retries
            time.sleep(wait_time)
            retries += 1
        except APIError as e:
            # Log error and raise exception
            print(f"API Error: {e}")
            raise
    
    # If we exhaust retries
    raise APIRateLimitError(f"Rate limit exceeded for {endpoint_name} after {max_retries} retries")
```

## Data Update Frequency

Different APIs have different update frequencies. Here's a suggested schedule for fetching data:

1. **Real-time data** (LMPs, current load, generation mix)
   - Update frequency: Every 5-15 minutes
   - APIs: EIA real-time endpoints

2. **Hourly data** (Day-ahead prices, forecasted load)
   - Update frequency: Hourly
   - APIs: EIA hourly endpoints

3. **Daily data** (Historical trends, generation capacities)
   - Update frequency: Daily
   - APIs: EIA, OpenEI, NREL

Sample scheduling code:

```python
def schedule_data_updates():
    """
    Schedule different API calls at appropriate intervals
    """
    # Define zones to fetch data for
    zones = [
        'US-FLA-FPC', 'US-FLA-FPL', 'US-SE-SOCO',
        # Add other zones
    ]
    
    current_time = datetime.now()
    one_hour_ago = current_time - timedelta(hours=1)
    
    # Real-time data (every 15 minutes)
    for zone in zones:
        try:
            # Fetch and store real-time data with explicit datetime objects
            lmp_data = get_lmp_data(
                zone, 
                one_hour_ago,
                current_time
            )
            current_load = get_current_load(zone)
            generation_mix = get_generation_mix(zone, one_hour_ago, current_time)
            
            # Store in database
            # store_data(lmp_data, current_load, generation_mix)
            
        except Exception as e:
            print(f"Error updating real-time data for {zone}: {e}")
    
    # Hourly data (once per hour)
    current_hour = current_time.hour
    if current_hour % 1 == 0:  # Every hour
        for zone in zones:
            try:
                # Fetch and store hourly data with explicit datetime objects
                day_ahead_prices = get_day_ahead_prices(
                    zone,
                    current_time
                )
                forecasted_load = get_forecasted_load(zone, current_time)
                
                # Store in database
                # store_hourly_data(day_ahead_prices, forecasted_load)
                
            except Exception as e:
                print(f"Error updating hourly data for {zone}: {e}")
    
    # Daily data (once per day)
    if current_time.hour == 1 and current_time.minute < 15:  # 1:00-1:15 AM
        thirty_days_ago = current_time - timedelta(days=30)
        
        for zone in zones:
            try:
                # Fetch and store daily data with explicit datetime objects
                historical_trends = get_historical_price_trends(
                    zone,
                    thirty_days_ago,
                    current_time,
                    'daily'
                )
                generation_capacities = get_generation_capacities(zone, thirty_days_ago, current_time)
                
                # Store in database
                # store_daily_data(historical_trends, generation_capacities)
                
            except Exception as e:
                print(f"Error updating daily data for {zone}: {e}")
```