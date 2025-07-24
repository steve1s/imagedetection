# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt if it exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Expose port 5000 for Flask or similar apps
EXPOSE 5000

# Default command to run the app (assumes app.py is the entry point)
CMD ["python", "app.py"]
