# -------------------------------
# Base Image
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# Environment Settings
# -------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------
# Working Directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Install Dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy App Files
# -------------------------------
COPY . .

# -------------------------------
# Expose Port
# -------------------------------
EXPOSE 3000

# -------------------------------
# Run FastAPI App
# -------------------------------
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
