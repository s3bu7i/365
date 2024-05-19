import re
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine

log_data = """
<30>May 19 12:00:00 router01 %SYS-5-CONFIG_I: Configured from console by admin on console
<30>May 19 12:00:10 switch01 fpc0 Interface ge-0/0/1.0, Changed state to up
<30>May 19 12:00:20 bts01 [RRC] RRC Connection Reestablishment Request: UE ID 1234
<30>May 19 12:01:00 nagios01 [SERVICE ALERT] CoreRouter;Ping;CRITICAL;HARD;1;PING CRITICAL - Packet loss = 100%
<30>May 19 12:01:30 appserver01 192.168.1.14 - - [19/May/2024:12:01:30 +0000] "DELETE /api/v1/remove HTTP/1.1" 204 0
"""

# Define regex pattern to parse log entries
pattern = re.compile(r'<\d+>(\w{3} \d+ \d{2}:\d{2}:\d{2}) (\S+) (.+)')


def parse_log_data(log_data):
    log_entries = []
    for line in log_data.strip().split('\n'):
        match = pattern.match(line)
        if match:
            log_time_str, host, message = match.groups()
            log_time = datetime.strptime(
                f"2024 {log_time_str}", "%Y %b %d %H:%M:%S")
            log_entries.append({
                'log_time': log_time,
                'host': host,
                'message': message
            })
    return log_entries


parsed_data = parse_log_data(log_data)

# Convert to DataFrame
df = pd.DataFrame(parsed_data)


def load_data_to_mysql(df, db_uri):
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        df.to_sql('log_data', con=conn, if_exists='append', index=False)


# Database connection URI
db_uri = 'mysql+mysqlconnector://user:password@localhost:3306/log_data'

# Load data into MySQL
load_data_to_mysql(df, db_uri)

print("Data loaded into MySQL successfully.")
