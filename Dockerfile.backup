# Use more secure and updated base image
FROM python:3.11-slim-bookworm

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies including Java with security updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre \
    default-jdk \
    && apt-get upgrade -y \
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

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app
ENV DJANGO_SETTINGS_MODULE=languge_companion.settings

# Switch to non-root user
USER appuser

# Create staticfiles directory
RUN mkdir -p staticfiles

# Collect static files (skip if no static files exist)
RUN python manage.py collectstatic --noinput || echo "No static files to collect"

# Run database migrations
RUN python manage.py migrate --noinput || echo "No migrations to run"

# Expose port
EXPOSE 8000

# Start command with non-root user
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--user", "appuser", "--group", "appuser", "languge_companion.wsgi:application"]