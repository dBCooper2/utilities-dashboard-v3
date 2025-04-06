# Weather Data Service with TimescaleDB

A containerized application for collecting and storing weather data from the RegionWeather class using PostgreSQL with TimescaleDB extension.

## Dependencies

- Docker
- Docker Compose

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/utilities-dashboard-v3.git
   cd utilities-dashboard-v3
   ```

2. Run the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```

This will:
- Build the backend-prod container
- Pull and start the TimescaleDB container
- Run the test script to verify functionality

## Project Structure

- `backend-prod/`: Contains the Python application code
  - `classes/regionweather.py`: Weather data retrieval class
  - `db/db_manager.py`: Database management utilities
  - `tests/test_florida_region.py`: Test script for Florida region data
  - `Dockerfile`: Container definition for the backend
  - `requirements.txt`: Python dependencies

- `docker-compose.yml`: Orchestration file for all services

## Database Schema

The application creates the following database structure:

### Regions Table
Stores metadata about each geographical region:
- `region_name` (TEXT, PRIMARY KEY)
- `lat` (FLOAT)
- `lon` (FLOAT)
- `alt` (FLOAT)
- `start_date` (TIMESTAMP)
- `end_date` (TIMESTAMP)

### Hourly Data Table (TimescaleDB Hypertable)
Stores hourly weather data:
- `time` (TIMESTAMP)
- `region_name` (TEXT, FOREIGN KEY)
- Weather metrics (temp, dwpt, rhum, etc.)

### Daily Data Table (TimescaleDB Hypertable)
Stores daily aggregated weather data:
- `time` (TIMESTAMP)
- `region_name` (TEXT, FOREIGN KEY)
- Weather metrics (temp, tmin, tmax, etc.)

## Accessing the Database

You can connect to the TimescaleDB instance using:

```bash
docker exec -it utilities-dashboard-v3_timescaledb_1 psql -U postgres -d weather_db
```

## Running Tests Manually

To run the test script manually:

```bash
docker exec -it utilities-dashboard-v3_backend-prod_1 python tests/test_florida_region.py
```

## Environment Variables

The application uses the following environment variables (with defaults):

- `POSTGRES_HOST`: Database hostname (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_DB`: Database name (default: weather_db)
- `POSTGRES_USER`: Database username (default: postgres)
- `POSTGRES_PASSWORD`: Database password (default: postgres)