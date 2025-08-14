#!/usr/bin/env bash
# Exit if any command fails
set -o errexit

# Update package lists
apt-get update

# Install Java Runtime Environment without asking for confirmation
apt-get install -y default-jre

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
