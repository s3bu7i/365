import mysql.connector
import re
from datetime import datetime

# MySQL Connection
cnx = mysql.connector.connect(user='your_user', password='your_password',
                              host='127.0.0.1', database='log_data')
cursor = cnx.cursor()

# Parse log line
def parse_log_line(line):
    regex = r'\[(?P<time>[^\]]+)\] (?P<host>[^ ]+) (?P<plugin>[^ ]+) (?P<plugin_instance>[^ ]+) (?P<type>[^ ]+) (?P<type_instance>[^ ]+) (?P<value>[^ ]+)'
    match = re.match(regex, line)
    if match:
        data = match.groupdict()
        data['time'] = datetime.strptime(data['time'], '%Y-%m-%d %H:%M:%S')
        data['value'] = float(data['value'])
        return data
    return None

# Insert data into MySQL
def insert_data(data):
    add_log = ("INSERT INTO performance_logs "
               "(log_time, host, plugin, plugin_instance, type, type_instance, value) "
               "VALUES (%(time)s, %(host)s, %(plugin)s, %(plugin_instance)s, %(type)s, %(type_instance)s, %(value)s)")
    cursor.execute(add_log, data)
    cnx.commit()

# Read and process log file
with open('performance.log', 'r') as file:
    for line in file:
        data = parse_log_line(line)
        if data:
            insert_data(data)

# Close connection
cursor.close()
cnx.close()
