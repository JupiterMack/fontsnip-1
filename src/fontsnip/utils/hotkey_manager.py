# -*- coding: utf-8 -*-
"""
src/fontsnip/utils/hotkey_manager.py

A utility class to manage the global hotkey using the 'pynput' library.

This module provides the HotkeyManager class, which runs in a separate thread
to listen for a specific key combination. When the hotkey is detected, it
triggers a callback function, allowing the main application to remain
responsive.
"""

import logging
from typing import Callable, Optional
from pynput import keyboard


class HotkeyManager:
    """
    Manages a global hotkey listener in a separate thread.

    Attributes:
        hotkey_str (str): The string representation of the hotkey (e.g., '<ctrl>+<alt>+s').
        callback (Callable[[], None]): The function to call when the hotkey is pressed.
        listener (Optional[keyboard.GlobalHotKeys]): The pynput listener instance.
    """

    def __init__(self, hotkey_str: str, callback: Callable[[], None]):
        """
        Initializes the HotkeyManager.

        Args:
            hotkey_str (str): The hotkey combination to listen for.
            callback (Callable[[], None]): The function to execute upon hotkey activation.
        """
        self.hotkey_str = hotkey_str
        self.callback = callback
        self.listener: Optional[keyboard.GlobalHotKeys] = None
        logging.info(f"Initializing hotkey manager for combination: {self.hotkey_str}")

    def _on_activate(self):
        """
        Internal callback method triggered by the pynput listener.
        """
        logging.debug(f"Hotkey '{self.hotkey_str}' activated.")
        if self.callback:
            try:
                self.callback()
            except Exception as e:
                logging.error(f"Error executing hotkey callback: {e}", exc_info=True)

    def start(self):
        """
        Starts the hotkey listener thread.

        Initializes and starts the pynput.keyboard.GlobalHotKeys listener.
        If a listener is already running, it will be stopped and a new one started.
        Handles potential errors if the hotkey string is invalid.
        """
        if self.listener and self.listener.is_alive():
            logging.warning("Hotkey listener is already running. Stopping it before starting a new one.")
            self.stop()

        try:
            # The listener runs in its own daemon thread.
            self.listener = keyboard.GlobalHotKeys(
                {self.hotkey_str: self._on_activate}
            )
            self.listener.start()
            logging.info(f"Global hotkey listener started for '{self.hotkey_str}'.")
        except Exception as e:
            # pynput can raise various errors, including ValueError for bad strings.
            logging.error(f"Failed to start hotkey listener for '{self.hotkey_str}': {e}", exc_info=True)
            self.listener = None

    def stop(self):
        """
        Stops the hotkey listener thread if it is running.
        """
        if self.listener and self.listener.is_alive():
            logging.info("Stopping global hotkey listener.")
            self.listener.stop()
            # The .join() might be useful to ensure the thread has fully terminated,
            # but pynput's stop() is generally sufficient and non-blocking.
            self.listener = None
        else:
            logging.debug("Attempted to stop listener, but it was not running.")


if __name__ == '__main__':
    # Example usage and simple test for the HotkeyManager
    import time

    # Configure basic logging for demonstration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def my_callback_function():
        """A simple function to be called by the hotkey."""
        print("Hotkey was pressed! Callback executed.")

    # Define the hotkey combination
    # Note: On some systems, you might need specific keys like <cmd> for Mac
    # or <ctrl>+<alt> for others.
    HOTKEY = '<ctrl>+<alt>+s'
    print(f"--- HotkeyManager Test ---")
    print(f"Press {HOTKEY} to trigger the callback.")
    print("Press Ctrl+C in this terminal to exit the test.")

    # Create and start the manager
    manager = HotkeyManager(hotkey_str=HOTKEY, callback=my_callback_function)
    manager.start()

    try:
        # Keep the main thread alive to allow the listener thread to run.
        # In a real application, the main GUI loop would serve this purpose.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down.")
    finally:
        # Cleanly stop the listener
        manager.stop()
        print("Hotkey listener stopped. Exiting.")