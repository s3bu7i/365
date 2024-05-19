import re
from datetime import datetime
import pandas as pd


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
