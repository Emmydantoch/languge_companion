#!/usr/bin/env bash
# Exit if any command fails
set -o errexit

echo "Starting build process..."

# Install system dependencies (Java) using apt-get with proper permissions
echo "Installing Java..."
apt-get update
apt-get install -y default-jre default-jdk

# Verify Java installation
echo "Verifying Java installation..."
java -version

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run Django migrations
echo "Running Django migrations..."
python manage.py migrate --noinput

echo "Build completed successfully!"
