import pandas as pd
import sqlite3
import os
from datetime import datetime

def export_db_to_csv():
    # Connect to the SQLite database
    db_path = 'weather_forecasts.db'
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found!")
        return
        
    conn = sqlite3.connect(db_path)
    
    # Create exports directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = f'db_exports_{timestamp}'
    os.makedirs(export_dir, exist_ok=True)
    
    # Tables to export
    tables = ['regions', 'historical_weather', 'forecasts', 'model_metrics']
    
    for table in tables:
        try:
            # Read the table into a pandas DataFrame
            print(f"Exporting {table}...")
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            
            # Export to CSV
            csv_path = os.path.join(export_dir, f"{table}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Successfully exported {len(df)} rows from {table} to {csv_path}")
            
        except Exception as e:
            print(f"Error exporting {table}: {e}")
    
    conn.close()
    print(f"\nExport complete! Files are in the '{export_dir}' directory")

if __name__ == '__main__':
    export_db_to_csv() 