from kafka import KafkaProducer
import json
import time
from datetime import datetime
import mysql.connector


def insert_log_to_db(log_entry):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='yourpassword',
        database='log_data'
    )
    cursor = conn.cursor()

    query = ("INSERT INTO performance_logs "
             "(log_time, host, plugin, plugin_instance, type, type_instance, value) "
             "VALUES (%s, %s, %s, %s, %s, %s, %s)")
    data = (log_entry['time'], log_entry['host'], log_entry['plugin'], log_entry['plugin_instance'],
            log_entry['type'], log_entry['type_instance'], log_entry['value'])

    cursor.execute(query, data)
    conn.commit()
    cursor.close()
    conn.close()


def send_log():
    log_entry = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "host": "localhost",
        "plugin": "cpu",
        "plugin_instance": "0",
        "type": "user",
        "type_instance": "",
        "value": 0.23
    }
    insert_log_to_db(log_entry)
    print(f"Inserted: {log_entry}")


while True:
    send_log()
    time.sleep(5)
