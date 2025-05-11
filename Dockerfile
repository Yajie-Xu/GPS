# Use a compatible Python version
FROM python:slim

# Set working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt    # NEW
# RUN pip install --no-cache-dir -r requirements.txt    # NEW
# Change this to match your actual entry script
CMD ["python", "Main.py"]