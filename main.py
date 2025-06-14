import sys
from PyQt6.QtWidgets import QApplication

# Assuming the main application logic, including the system tray icon,
# hotkey listener, and state machine, is encapsulated in the FontSnipApp class
# located in the src/fontsnip/app.py file.
try:
    from src.fontsnip.app import FontSnipApp
except ImportError as e:
    print(f"Error: Could not import the main application class 'FontSnipApp'.")
    print(f"Please ensure the project structure is correct (e.g., src/fontsnip/app.py exists).")
    print(f"Details: {e}")
    sys.exit(1)

def main():
    """
    The main entry point for the FontSnip application.

    This function initializes the QApplication, creates the main application
    controller instance (FontSnipApp), and starts the application's event loop.
    """
    # Create the PyQt application instance.
    # sys.argv allows passing command-line arguments to the application if needed.
    app = QApplication(sys.argv)

    # To prevent the application from closing when the last window is closed
    # (e.g., after a capture), we set this attribute. The application will
    # only exit when explicitly told to, for instance, via the system tray icon.
    app.setQuitOnLastWindowClosed(False)

    # The main application logic is encapsulated in the FontSnipApp class.
    # This class will handle the system tray, hotkey listening, and all
    # state transitions as described in the project overview.
    # We pass the app instance to it in case it needs to interact with it directly.
    font_snip_app = FontSnipApp()

    # Start the Qt event loop. The application will now run, waiting for events
    # like the global hotkey press. The return value of exec() is the exit code,
    # which we pass to sys.exit() to ensure a clean shutdown.
    sys.exit(app.exec())


if __name__ == '__main__':
    # This standard Python construct ensures that the main() function is called
    # only when the script is executed directly.
    main()