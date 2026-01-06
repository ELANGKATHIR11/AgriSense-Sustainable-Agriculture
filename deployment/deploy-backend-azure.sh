#!/bin/bash
# AgriSense Backend Deployment to Azure Container Apps

# Configuration
RESOURCE_GROUP="agrisense-rg"
REGISTRY_NAME="agrisenseregistry"
CONTAINER_APP_ENV="agrisense-env"
CONTAINER_APP_NAME="agrisense-api"
IMAGE_NAME="agrisense-api:latest"
LOCATION="eastus"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying AgriSense Backend to Azure Container Apps${NC}"

# Step 1: Create Resource Group
echo -e "${YELLOW}1. Creating resource group...${NC}"
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Step 2: Create Container Registry
echo -e "${YELLOW}2. Creating Azure Container Registry...${NC}"
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic

# Step 3: Build and push Docker image
echo -e "${YELLOW}3. Building and pushing Docker image...${NC}"
az acr build \
  --registry $REGISTRY_NAME \
  --image $IMAGE_NAME \
  --file Dockerfile .

# Step 4: Create Container Apps Environment
echo -e "${YELLOW}4. Creating Container Apps Environment...${NC}"
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Step 5: Create Container App with API
echo -e "${YELLOW}5. Creating Container App...${NC}"
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $REGISTRY_NAME.azurecr.io/$IMAGE_NAME \
  --target-port 8004 \
  --ingress external \
  --registry-server $REGISTRY_NAME.azurecr.io \
  --registry-username $(az acr credential show --name $REGISTRY_NAME --query username -o tsv) \
  --registry-password $(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" -o tsv) \
  --env-vars \
    CORS_ORIGINS="https://agrisense-fe79c.web.app" \
    DATABASE_URL="$(echo $COSMOS_CONNECTION_STRING)" \
    LOG_LEVEL="INFO"

# Step 6: Get the public URL
echo -e "${YELLOW}6. Getting Container App URL...${NC}"
APP_URL=$(az containerapp show \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_APP_NAME \
  --query "properties.configuration.ingress.fqdn" -o tsv)

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}Backend URL: https://$APP_URL${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update your frontend .env file with the new backend URL:"
echo "   VITE_BACKEND_API_URL=https://$APP_URL/api/v1"
echo "2. Rebuild and redeploy the frontend:"
echo "   cd src/frontend && npm run build && firebase deploy --only hosting"
