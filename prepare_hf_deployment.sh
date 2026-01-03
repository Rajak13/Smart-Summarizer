#!/bin/bash

echo "🚀 Preparing Smart Summarizer for Hugging Face Spaces deployment..."

# Create deployment directory
echo "📁 Creating deployment directory..."
mkdir -p hf_deployment
cd hf_deployment

# Copy necessary files and folders
echo "📋 Copying project files..."
cp -r ../models .
cp -r ../utils .
cp -r ../webapp .
cp -r ../data .
cp ../requirements.txt .
cp ../Dockerfile .
cp ../README.md .
cp ../.gitignore .

# Create uploads directory
mkdir -p uploads

echo "✅ Files prepared for Hugging Face Spaces deployment!"
echo ""
echo "📂 Deployment files are in: ./hf_deployment/"
echo ""
echo "🔗 Next steps:"
echo "1. Create a new Space on Hugging Face (SDK: Docker)"
echo "2. Upload all files from ./hf_deployment/ to your Space"
echo "3. Wait for the build to complete"
echo "4. Your app will be live!"
echo ""
echo "📖 See HUGGINGFACE_DEPLOYMENT.md for detailed instructions"