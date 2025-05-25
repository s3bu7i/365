import sqlite3
import datetime

# Connect to the SQLite database
conn = sqlite3.connect('my_database.db')

# Create a table to store the text and timestamp
conn.execute('''CREATE TABLE IF NOT EXISTS my_table
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL);''')

# Prompt the user to input some text
text = input("Enter some text: ")

# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Insert the text and timestamp into the database
conn.execute(
    "INSERT INTO my_table (text, timestamp) VALUES (?, ?)", (text, timestamp))
conn.commit()

# Print a message to confirm that the data was saved
print("Text saved to database with timestamp:", timestamp)

# Close the database connection
conn.close()
