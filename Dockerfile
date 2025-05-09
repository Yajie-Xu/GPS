# Use a compatible Python version
FROM python:3.8-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Change this to match your actual entry script
CMD ["python", "Main.py"]