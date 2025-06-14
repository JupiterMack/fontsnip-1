"""
FontSnip Application Package.

This package contains the core logic for the FontSnip utility, including
the user interface, screen capture, image processing, and font matching
modules.

By importing the main application class here, we provide a simplified
entry point for the application runner.
"""

__version__ = "0.1.0"

# Import the main application class to make it accessible at the package level.
# This allows for a cleaner import in the main entry point, e.g.,
# from src.fontsnip import FontSnipApp
# Note: This assumes a file named 'app.py' exists within this package,
# which contains the 'FontSnipApp' class.
try:
    from .app import FontSnipApp
except ImportError:
    # This might happen during initial setup or if the file structure changes.
    # We'll let it pass silently here, and the error will be caught
    # at the actual point of use (e.g., in main.py).
    pass


# Define the public API of the package for 'from fontsnip import *'
__all__ = ["FontSnipApp"]