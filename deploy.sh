#!/bin/bash

# Smart Summarizer - Quick Deploy Script
# This script helps you deploy to various platforms

echo "=================================="
echo "Smart Summarizer - Deployment"
echo "=================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Display menu
echo "Select deployment platform:"
echo "1) Railway (Recommended for ML apps)"
echo "2) Render"
echo "3) Heroku"
echo "4) Docker (Local)"
echo "5) Test locally"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Deploying to Railway..."
        if ! command_exists railway; then
            echo "Railway CLI not found. Installing..."
            npm install -g @railway/cli
        fi
        railway login
        railway init
        railway up
        echo ""
        echo "✓ Deployed to Railway!"
        railway open
        ;;
    2)
        echo ""
        echo "Deploying to Render..."
        echo "Please follow these steps:"
        echo "1. Push your code to GitHub"
        echo "2. Go to https://render.com"
        echo "3. Click 'New' → 'Web Service'"
        echo "4. Connect your GitHub repository"
        echo "5. Render will auto-deploy using render.yaml"
        echo ""
        read -p "Press Enter to open Render dashboard..."
        open "https://render.com" 2>/dev/null || xdg-open "https://render.com" 2>/dev/null
        ;;
    3)
        echo ""
        echo "Deploying to Heroku..."
        if ! command_exists heroku; then
            echo "Heroku CLI not found. Please install from: https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        heroku login
        heroku create smart-summarizer-$(date +%s)
        git push heroku main
        heroku open
        echo ""
        echo "✓ Deployed to Heroku!"
        ;;
    4)
        echo ""
        echo "Building Docker image..."
        docker build -t smart-summarizer .
        echo ""
        echo "Starting container..."
        docker run -d -p 5001:5001 --name smart-summarizer smart-summarizer
        echo ""
        echo "✓ Docker container started!"
        echo "Access at: http://localhost:5001"
        echo ""
        echo "Useful commands:"
        echo "  docker logs -f smart-summarizer  # View logs"
        echo "  docker stop smart-summarizer     # Stop container"
        echo "  docker start smart-summarizer    # Start container"
        ;;
    5)
        echo ""
        echo "Starting local development server..."
        ./run_webapp.sh
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Deployment complete!"
