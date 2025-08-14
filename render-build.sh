#!/usr/bin/env bash
# Exit if any command fails
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Collect static files for Django
python manage.py collectstatic --noinput
