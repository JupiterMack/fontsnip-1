# -*- coding: utf-8 -*-
"""
src/fontsnip/config.py

Module for handling application configuration.

This module defines default settings for FontSnip, such as the global hotkey
combination and image processing parameters. It provides functionality to load
user-defined settings from a configuration file (config.ini), creating one
with default values on the first run.
"""

import configparser
import platform
from pathlib import Path

# --- Constants ---
APP_NAME = "FontSnip"
DEFAULT_CONFIG_FILENAME = "config.ini"
DEFAULT_DB_FILENAME = "font_features.pkl"
DEFAULT_HOTKEY = "<ctrl>+<alt>+s"


def get_app_dir() -> Path:
    """
    Gets the application's data directory in a cross-platform way.

    This directory is used to store the configuration file and the
    pre-computed font database.

    - Windows: %APPDATA%/FontSnip
    - macOS: ~/Library/Application Support/FontSnip
    - Linux: ~/.config/FontSnip

    Returns:
        Path: A Path object to the application's data directory.
    """
    if platform.system() == "Windows":
        # C:\Users\<Username>\AppData\Roaming\FontSnip
        app_dir = Path.home() / "AppData" / "Roaming" / APP_NAME
    elif platform.system() == "Darwin":  # macOS
        # /Users/<Username>/Library/Application Support/FontSnip
        app_dir = Path.home() / "Library" / "Application Support" / APP_NAME
    else:  # Linux and other Unix-like
        # /home/<Username>/.config/FontSnip
        app_dir = Path.home() / ".config" / APP_NAME

    # Create the directory if it doesn't exist
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


class Config:
    """
    Manages application configuration by loading defaults and overriding
    them with settings from a user-specific config file.
    """

    def __init__(self):
        """Initializes the configuration manager."""
        self.parser = configparser.ConfigParser()
        self.app_dir = get_app_dir()
        self.config_file_path = self.app_dir / DEFAULT_CONFIG_FILENAME

        self._load_defaults()
        self._load_from_file()

    def _load_defaults(self):
        """Sets the default configuration values in the parser object."""
        self.parser["General"] = {
            "hotkey": DEFAULT_HOTKEY
        }
        self.parser["Processing"] = {
            "upscale_factor": "2.0",
            "ocr_confidence_threshold": "60",
            "denoise": "False"
        }
        self.parser["Database"] = {
            "db_filename": DEFAULT_DB_FILENAME
        }
        self.parser["Display"] = {
            "results_count": "3"
        }

    def _load_from_file(self):
        """
        Loads settings from the config.ini file, overriding defaults.
        If the file doesn't exist, it will be created with default values.
        """
        if not self.config_file_path.exists():
            self._save_defaults()
        else:
            # Read the existing file, which may override some or all defaults
            self.parser.read(self.config_file_path)

    def _save_defaults(self):
        """Saves the current (default) configuration to the config file."""
        try:
            with open(self.config_file_path, 'w') as configfile:
                # Add a header comment to the file
                configfile.write(f"# {APP_NAME} Configuration File\n")
                configfile.write("# You can edit these values. Restart the app for changes to take effect.\n\n")
                self.parser.write(configfile)
        except IOError as e:
            # This is a non-critical error, so we just print it
            print(f"Error: Could not write to config file at {self.config_file_path}: {e}")

    # --- Properties to access settings easily and with correct types ---

    @property
    def hotkey(self) -> str:
        """The global hotkey combination for triggering capture mode."""
        return self.parser.get("General", "hotkey", fallback=DEFAULT_HOTKEY)

    @property
    def upscale_factor(self) -> float:
        """The factor by which to upscale the captured image for better OCR."""
        return self.parser.getfloat("Processing", "upscale_factor", fallback=2.0)

    @property
    def ocr_confidence_threshold(self) -> int:
        """The minimum confidence level (0-100) for a character to be considered."""
        return self.parser.getint("Processing", "ocr_confidence_threshold", fallback=60)

    @property
    def denoise(self) -> bool:
        """Whether to apply a denoising filter before OCR."""
        return self.parser.getboolean("Processing", "denoise", fallback=False)

    @property
    def db_path(self) -> Path:
        """The full path to the font features database file."""
        filename = self.parser.get("Database", "db_filename", fallback=DEFAULT_DB_FILENAME)
        return self.app_dir / filename

    @property
    def results_count(self) -> int:
        """The number of top font matches to display."""
        return self.parser.getint("Display", "results_count", fallback=3)


# --- Singleton Instance ---
# Create a single, globally accessible instance of the Config class.
# Other modules can import this instance directly.
# e.g., from src.fontsnip.config import config
config = Config()


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print(f"--- {APP_NAME} Configuration ---")
    print(f"Application Data Directory: {config.app_dir}")
    print(f"Config file path: {config.config_file_path}")
    print(f"Font DB path: {config.db_path}")

    print("\n--- Loaded Settings ---")
    print(f"Hotkey: {config.hotkey} (type: {type(config.hotkey).__name__})")
    print(f"Upscale Factor: {config.upscale_factor} (type: {type(config.upscale_factor).__name__})")
    print(f"OCR Confidence: {config.ocr_confidence_threshold} (type: {type(config.ocr_confidence_threshold).__name__})")
    print(f"Denoise: {config.denoise} (type: {type(config.denoise).__name__})")
    print(f"Results to show: {config.results_count} (type: {type(config.results_count).__name__})")

    # Test that the config file is created if it doesn't exist
    if not config.config_file_path.exists():
        print("\nConfig file did not exist. It should have been created with default values.")
    else:
        print("\nConfig file loaded successfully.")
        print("You can edit it and re-run this script to see your changes.")