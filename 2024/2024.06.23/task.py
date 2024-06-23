import datetime
from cryptography.fernet import Fernet

# Generate a key for encryption and decryption
# You must use this key to decrypt the encrypted messages
key = Fernet.generate_key()
cipher_suite = Fernet(key)


def encrypt_text(text):
    """Encrypts the provided text using Fernet encryption."""
    return cipher_suite.encrypt(text.encode()).decode()


def decrypt_text(encrypted_text):
    """Decrypts the provided text using Fernet decryption."""
    return cipher_suite.decrypt(encrypted_text.encode()).decode()


def add_task(tasks):
    """Adds a new task to the tasks list."""
    task_description = input("Enter the task description: ")
    deadline_str = input("Enter the deadline (YYYY-MM-DD): ")

    try:
        deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    encrypted_description = encrypt_text(task_description)
    tasks.append({"description": encrypted_description, "deadline": deadline})
    print("Task added successfully!")


def view_tasks(tasks):
    """Displays all tasks."""
    if not tasks:
        print("No tasks available.")
        return

    for i, task in enumerate(tasks, start=1):
        decrypted_description = decrypt_text(task["description"])
        print(f"Task {i}:")
        print(f"  Description: {decrypted_description}")
        print(f"  Deadline: {task['deadline'].strftime('%Y-%m-%d')}")


def main():
    tasks = []
    while True:
        print("\nTask Management System")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            add_task(tasks)
        elif choice == '2':
            view_tasks(tasks)
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
