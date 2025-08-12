import datetime

# Get the current date and time
now = datetime.datetime.now()

# Print the current date and time to the console
print("Current date and time:")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Save the current date and time to a text file
with open('current_time.txt', 'w') as f:
    f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
