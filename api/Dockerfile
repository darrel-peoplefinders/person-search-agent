# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variables (if needed later)
# ENV ENVIRONMENT=production

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the FastAPI app with uvicorn
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1
