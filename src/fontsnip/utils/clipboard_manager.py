# -*- coding: utf-8 -*-
"""
src/fontsnip/utils/clipboard_manager.py

A simple wrapper utility for interacting with the system clipboard.

This module centralizes clipboard operations, primarily using the 'pyperclip'
library. It provides a robust function to copy text, including error handling
for environments where a clipboard might not be available.
"""

import pyperclip
import logging

# Set up a logger for this module. The application's entry point should configure the root logger.
logger = logging.getLogger(__name__)


def copy_to_clipboard(text: str) -> bool:
    """
    Copies the given text to the system clipboard.

    This function acts as a wrapper around pyperclip.copy() to provide
    centralized error handling and logging for the application.

    Args:
        text (str): The string to be copied.

    Returns:
        bool: True if the text was copied successfully, False otherwise.
    """
    try:
        pyperclip.copy(text)
        logger.info(f"Successfully copied to clipboard: '{text}'")
        return True
    except pyperclip.PyperclipException as e:
        # This can happen on systems without a clipboard (e.g., some Linux servers)
        # or if the necessary copy/paste mechanism is not installed (e.g., xclip/xsel).
        logger.error(f"Failed to copy text to clipboard: {e}")
        logger.warning(
            "Clipboard functionality may not be available on this system. "
            "If on Linux, please ensure 'xclip' or 'xsel' is installed."
        )
        return False


if __name__ == '__main__':
    # This block allows for direct testing of the clipboard functionality.
    # To run, execute `python -m src.fontsnip.utils.clipboard_manager` from the project root.

    # Basic logging configuration for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_string = "Arial Bold"
    print("--- Testing Clipboard Manager ---")
    print(f"Attempting to copy the string: '{test_string}'")

    success = copy_to_clipboard(test_string)

    if success:
        print("Copy operation reported success.")
        try:
            pasted_content = pyperclip.paste()
            print(f"Verification: Pasted content is '{pasted_content}'")
            if pasted_content == test_string:
                print("✅ Test PASSED: Pasted content matches the original string.")
            else:
                print("❌ Test FAILED: Pasted content does not match.")
        except pyperclip.PyperclipException as e:
            print(f"❌ Test FAILED: Could not paste for verification. Error: {e}")
    else:
        print("❌ Test FAILED: Copy operation reported failure.")
        print("This may be expected on a system without a GUI or clipboard utility.")

    print("--- Test Complete ---")