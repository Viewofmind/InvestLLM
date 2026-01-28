#!/bin/bash
# Deploy InvestLLM to Google Cloud Run
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Docker installed
# 3. GCP project created with billing enabled
#
# Usage:
#   ./deploy.sh [PROJECT_ID] [REGION]
#
# Example:
#   ./deploy.sh my-project-123 asia-south1

set -e

# Configuration
PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-asia-south1}"
SERVICE_NAME="investllm"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "======================================"
echo "InvestLLM Google Cloud Deployment"
echo "======================================"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Verify gcloud is authenticated
if ! gcloud auth print-identity-token > /dev/null 2>&1; then
    echo "Error: Not authenticated with gcloud"
    echo "Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project "${PROJECT_ID}"

# Enable required APIs
echo "[1/6] Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com

# Create secrets (if not exists)
echo "[2/6] Setting up secrets..."
create_secret_if_missing() {
    SECRET_NAME=$1
    if ! gcloud secrets describe "${SECRET_NAME}" > /dev/null 2>&1; then
        echo "Creating secret: ${SECRET_NAME}"
        echo -n "placeholder" | gcloud secrets create "${SECRET_NAME}" --data-file=-
        echo "Warning: Update ${SECRET_NAME} with actual value using:"
        echo "  gcloud secrets versions add ${SECRET_NAME} --data-file=<your-key-file>"
    fi
}

create_secret_if_missing "zerodha-api-key"
create_secret_if_missing "zerodha-api-secret"
create_secret_if_missing "firecrawl-api-key"
create_secret_if_missing "gemini-api-key"

# Build Docker image
echo "[3/6] Building Docker image..."
cd "$(dirname "$0")/.."
docker build -t "${IMAGE_NAME}:latest" -f Dockerfile .

# Push to Container Registry
echo "[4/6] Pushing to Container Registry..."
docker push "${IMAGE_NAME}:latest"

# Deploy to Cloud Run
echo "[5/6] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --timeout 300s \
    --set-env-vars "TRADING_MODE=paper,INITIAL_CAPITAL=1000000" \
    --set-secrets "ZERODHA_API_KEY=zerodha-api-key:latest,ZERODHA_API_SECRET=zerodha-api-secret:latest,FIRECRAWL_API_KEY=firecrawl-api-key:latest,GEMINI_API_KEY=gemini-api-key:latest"

# Get service URL
echo "[6/6] Deployment complete!"
echo ""
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)')
echo "======================================"
echo "Service deployed successfully!"
echo "======================================"
echo "URL: ${SERVICE_URL}"
echo ""
echo "Next steps:"
echo "1. Update secrets with actual API keys:"
echo "   gcloud secrets versions add zerodha-api-key --data-file=<key-file>"
echo ""
echo "2. Set up custom domain (optional):"
echo "   gcloud run domain-mappings create --service ${SERVICE_NAME} --domain your-domain.com"
echo ""
