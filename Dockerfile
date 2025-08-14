FROM python:3.11-slim

# Install system dependencies including Java
RUN apt-get update && \
    apt-get install -y \
    default-jre \
    default-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Verify Java installation
RUN java -version

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV DJANGO_SETTINGS_MODULE=languge_companion.settings

# Create staticfiles directory
RUN mkdir -p staticfiles

# Collect static files (skip if no static files exist)
RUN python manage.py collectstatic --noinput || echo "No static files to collect"

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "languge_companion.wsgi:application"]