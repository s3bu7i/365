import pyautogui as pt
import time
from pynput import keyboard

# Stop the script if the escape key is pressed
stop_script = False


def on_press(key):
    """Stop the script if the escape key is pressed."""
    global stop_script
    try:
        if key == keyboard.Key.esc:
            print("\nScript stopped by user.")
            stop_script = True
            return False
    except Exception as e:
        print(f"Error: {e}")

# Main function to send messages


def main():
    global stop_script

    try:
        limit = int(input("Enter the number of messages to send: "))
        message = input("Enter your message: ")
        delay = float(input("Enter delay between messages (in seconds): "))
        add_numbering = input(
            "Add numbering to messages? (y/n): ").lower() == 'y'
        add_timestamp = input(
            "Add timestamps to messages? (y/n): ").lower() == 'y'
        print("You have 5 seconds to switch to the target window...")
        time.sleep(5)

        i = 0
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        while i < limit and not stop_script:
            formatted_message = message
            if add_numbering:
                formatted_message = f"{i + 1}. {formatted_message}"
            if add_timestamp:
                formatted_message = f"{
                    formatted_message} [{time.strftime('%H:%M:%S')}]"
            pt.write(formatted_message)
            pt.press("enter")
            i += 1
            print(f"Sent: {formatted_message}")
            time.sleep(delay)
        # Stop the listener
        listener.stop()
        if stop_script:
            print("Script stopped manually.")
        else:
            print("All messages sent successfully!")
    except ValueError:
        print("Invalid input. Please enter numbers where required.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Run the main function
if __name__ == "__main__":
    main()
