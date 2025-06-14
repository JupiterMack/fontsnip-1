# -*- coding: utf-8 -*-
"""
The Utilities Package for FontSnip.

This package provides a collection of helper functions and utility modules that
support various parts of the FontSnip application. It aims to centralize
common, reusable logic, such as platform-specific path resolution, file I/O,
and other miscellaneous tasks.

By centralizing these utilities, we keep the core application logic in other
modules cleaner and more focused on their primary responsibilities.

Modules:
- paths: Contains functions for locating system font directories, application
         data folders, and other important file paths.

Example Usage:
    from fontsnip.utils import get_system_font_dirs
    font_dirs = get_system_font_dirs()
"""

# This __init__.py file is intentionally left simple. As utility functions are
# developed (e.g., in a `paths.py` or `logging.py` module within this
# directory), they can be imported here to create a convenient, flat API
# for the rest of the application.
#
# For example, if a `paths.py` module with a `get_app_data_path` function
# is created, you would add:
#
# from .paths import get_app_data_path
#
# __all__ = [
#     "get_app_data_path",
# ]
#
# For now, its main purpose is to mark 'utils' as a Python package.