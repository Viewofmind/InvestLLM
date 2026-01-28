# Multi-stage build for InvestLLM Trading Platform

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source
COPY frontend/ .

# Build production bundle
RUN npm run build

# ============================================
# Stage 2: Python Backend + Nginx
# ============================================
FROM python:3.11-slim

# Install nginx and supervisor
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements
COPY backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist /var/www/html

# Configure Nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/sites-enabled/investllm

# Configure Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /var/www/html

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start supervisor (manages nginx + uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
