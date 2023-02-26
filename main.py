import os

# Define the base folder name
base_folder = "2023.03."

# Define the number of folders to create
num_folders = 31

# Loop through the number of folders to create
for i in range(num_folders):
    # Create the folder name by appending the folder number to the base folder name
    folder_name = f'{base_folder}{i+1}'
    # Create the folder if it does not already exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder {folder_name} created successfully.')
    else:
        print(f'Folder {folder_name} already exists.')
