# RegionWeather Database

This is a PostgreSQL database with TimescaleDB extension for storing weather data from the RegionWeather class.

## Database Schema

The database consists of three main tables:

### Regions Table

Stores metadata about each geographical region:

| Column      | Type      | Description                    |
|-------------|-----------|--------------------------------|
| region_name | TEXT      | Primary key, unique identifier |
| lat         | FLOAT     | Latitude                       |
| lon         | FLOAT     | Longitude                      |
| alt         | FLOAT     | Altitude                       |
| start_date  | TIMESTAMP | Start date of data collection  |
| end_date    | TIMESTAMP | End date of data collection    |

### Hourly Data Table

Stores hourly weather data with a foreign key to the regions table:

| Column      | Type      | Description                     |
|-------------|-----------|---------------------------------|
| time        | TIMESTAMP | Timestamp for the data point    |
| region_name | TEXT      | Foreign key to regions table    |
| temp        | FLOAT     | Temperature                     |
| dwpt        | FLOAT     | Dew point                       |
| rhum        | FLOAT     | Relative humidity               |
| prcp        | FLOAT     | Precipitation                   |
| snow        | FLOAT     | Snow                            |
| wdir        | FLOAT     | Wind direction                  |
| wspd        | FLOAT     | Wind speed                      |
| wpgt        | FLOAT     | Wind peak gust                  |
| pres        | FLOAT     | Atmospheric pressure            |
| tsun        | FLOAT     | Sunshine duration               |
| coco        | INT       | Weather condition code          |

The combination of `region_name` and `time` forms a compound primary key.

### Daily Data Table

Stores daily weather data with a foreign key to the regions table:

| Column      | Type      | Description                     |
|-------------|-----------|---------------------------------|
| time        | TIMESTAMP | Timestamp for the data point    |
| region_name | TEXT      | Foreign key to regions table    |
| temp        | FLOAT     | Temperature                     |
| tmin        | FLOAT     | Minimum temperature             |
| tmax        | FLOAT     | Maximum temperature             |
| tavg        | FLOAT     | Average temperature             |
| dwpt        | FLOAT     | Dew point                       |
| rhum        | FLOAT     | Relative humidity               |
| prcp        | FLOAT     | Precipitation                   |
| snow        | FLOAT     | Snow                            |
| wdir        | FLOAT     | Wind direction                  |
| wspd        | FLOAT     | Wind speed                      |
| wpgt        | FLOAT     | Wind peak gust                  |
| pres        | FLOAT     | Atmospheric pressure            |
| tsun        | FLOAT     | Sunshine duration               |
| coco        | INT       | Weather condition code          |

The combination of `region_name` and `time` forms a compound primary key.

## TimescaleDB Hypertables

Both the hourly and daily data tables are configured as TimescaleDB hypertables, which provides:

- Automatic time partitioning for time-series data
- Better query performance for time-based queries
- Efficient data retention policies
- Advanced time-series analytical functions

## Setup Instructions

### Prerequisites

1. PostgreSQL 12 or higher
2. TimescaleDB extension

### Installing PostgreSQL and TimescaleDB

#### Ubuntu/Debian

```bash
# Add TimescaleDB repository
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt update

# Install PostgreSQL and TimescaleDB
sudo apt install postgresql-12 postgresql-12-timescaledb

# Initialize TimescaleDB
sudo timescaledb-tune --quiet --yes

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### macOS (using Homebrew)

```bash
# Install PostgreSQL and TimescaleDB
brew install postgresql timescaledb

# Initialize TimescaleDB
timescaledb-tune --quiet --yes

# Restart PostgreSQL
brew services restart postgresql
```

### Database Creation

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database
CREATE DATABASE weather_db;

# Connect to the new database
\c weather_db

# Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

### Python Requirements

Install required Python packages:

```bash
pip install psycopg2-binary pandas numpy meteostat
```

## Usage

See the example script in `backend-prod/examples/db_example.py` for complete usage instructions. Here's a basic workflow:

```python
from db.db_manager import DatabaseManager

# Initialize database connection
db_manager = DatabaseManager(
    host='localhost',
    port=5432,
    dbname='weather_db',
    user='postgres',
    password='postgres'
)

# Create schema if it doesn't exist
db_manager.initialize_database()

# Insert region data
db_manager.insert_region(region_weather)

# Insert hourly weather data
db_manager.insert_hourly_data(region_name, hourly_df)

# Insert daily weather data
db_manager.insert_daily_data(region_name, daily_df)

# Query region data
regions = db_manager.get_regions()

# Query weather data
hourly_data = db_manager.get_hourly_data(region_name, start_date, end_date)
daily_data = db_manager.get_daily_data(region_name, start_date, end_date)
```

## Performance Considerations

- For large datasets, consider implementing data retention policies using TimescaleDB's data retention features
- Create appropriate indexes for commonly queried columns
- Use TimescaleDB's continuous aggregates for pre-aggregated views of data for faster queries 