# Use Python 3.13.2 as the base image
FROM python:3.13.2-slim

# Install Java (OpenJDK 17) for language-tool-python
RUN apt-get update && apt-get install -y openjdk-17-jre && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.1.3

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies with Poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Copy the rest of the application
COPY . .

# Collect static files (optional, for Django)
RUN python manage.py collectstatic --noinput

# Expose the port
EXPOSE 10000

# Set environment variable for port
ENV PORT=10000

# Start the application with Gunicorn using Django's WSGI
CMD ["poetry", "run", "gunicorn", "languge_companion.wsgi:application", "--bind", "0.0.0.0:$PORT"]