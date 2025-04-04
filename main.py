import os

base_folder = "2025.4."

num_folders = 30

for i in range(num_folders):

    folder_name = f'{base_folder}{i+1}'
    # Create the folder if it does not already exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder {folder_name} created successfully.')
    else:
        print(f'Folder {folder_name} already exists.')
