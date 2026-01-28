#!/bin/bash
# Set up secrets for InvestLLM in Google Cloud Secret Manager
#
# Usage:
#   ./setup-secrets.sh
#
# This script will prompt for API keys and store them securely

set -e

echo "======================================"
echo "InvestLLM Secret Setup"
echo "======================================"
echo ""

# Function to set secret
set_secret() {
    SECRET_NAME=$1
    PROMPT=$2

    echo -n "${PROMPT}: "
    read -s SECRET_VALUE
    echo ""

    if [ -z "${SECRET_VALUE}" ]; then
        echo "Skipping ${SECRET_NAME} (empty value)"
        return
    fi

    # Check if secret exists
    if gcloud secrets describe "${SECRET_NAME}" > /dev/null 2>&1; then
        # Add new version
        echo -n "${SECRET_VALUE}" | gcloud secrets versions add "${SECRET_NAME}" --data-file=-
    else
        # Create new secret
        echo -n "${SECRET_VALUE}" | gcloud secrets create "${SECRET_NAME}" --data-file=-
    fi

    echo "Secret ${SECRET_NAME} updated successfully"
}

# Zerodha API
echo "=== Zerodha Kite API ==="
echo "Get your API key from: https://developers.kite.trade/"
set_secret "zerodha-api-key" "Zerodha API Key"
set_secret "zerodha-api-secret" "Zerodha API Secret"

echo ""

# Firecrawl API
echo "=== Firecrawl API ==="
echo "Get your API key from: https://firecrawl.dev/"
set_secret "firecrawl-api-key" "Firecrawl API Key"

echo ""

# Gemini API
echo "=== Google Gemini API ==="
echo "Get your API key from: https://aistudio.google.com/"
set_secret "gemini-api-key" "Gemini API Key"

echo ""
echo "======================================"
echo "Secrets configured successfully!"
echo "======================================"
echo ""
echo "To deploy the application, run:"
echo "  ./deploy.sh"
echo ""
