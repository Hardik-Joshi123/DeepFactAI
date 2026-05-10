FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy backend requirements
COPY backend/pyproject.toml backend/pyproject.toml
WORKDIR /app/backend
RUN pip install -e .

# Copy frontend
WORKDIR /app
COPY frontend frontend
WORKDIR /app/frontend
RUN npm install

# Build frontend
RUN npm run build

# Copy backend code
COPY backend backend

WORKDIR /app

EXPOSE 3000 8000

CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 & cd frontend && npm start"]
