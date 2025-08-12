#!/usr/bin/env python
"""
WSGI entry point for deployment
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "languge_companion.settings")

# Import the Django WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
