import pandas as pd
import re
from datetime import datetime
from sqlalchemy import create_engine

# Function to extract data from a log file


def extract_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Function to transform the extracted data


def transform_data(data):
    log_entries = []

    # Regex to parse log lines
    regex = r'\[(?P<time>[^\]]+)\] (?P<host>[^ ]+) (?P<plugin>[^ ]+) (?P<plugin_instance>[^ ]+) (?P<type>[^ ]+) (?P<type_instance>[^ ]+) (?P<value>[^ ]+)'

    for line in data:
        match = re.match(regex, line)
        if match:
            entry = match.groupdict()
            entry['time'] = datetime.strptime(
                entry['time'], '%Y-%m-%d %H:%M:%S')
            entry['value'] = float(entry['value'])
            log_entries.append(entry)

    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(log_entries)

    # Handle missing values, if any
    df.fillna(0, inplace=True)

    return df

# Function to load transformed data into MySQL


def load_data(df, db_uri):
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        df.to_sql('performance_logs', con=conn,
                  if_exists='append', index=False)

# Main ETL process function


def etl_process(file_path, db_uri):
    # Step 1: Extract
    raw_data = extract_data(file_path)

    # Step 2: Transform
    transformed_data = transform_data(raw_data)

    # Step 3: Load
    load_data(transformed_data, db_uri)

    print("ETL process completed successfully.")


# Example usage
if __name__ == "__main__":
    file_path = 'path_to_your_log_file.log'
    db_uri = 'mysql+mysqlconnector://user:password@localhost:3306/log_data'

    etl_process(file_path, db_uri)
