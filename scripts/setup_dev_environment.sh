#!/bin/bash

set -e

echo "🚀 Setting up Hough Server development environment..."
echo ""

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first!"
    exit 1
fi

echo "📦 Installing dependencies..."
poetry install

echo ""
echo "⚙️ Installing pre-commit hooks..."
poetry run pre-commit install-hooks

echo ""
echo "🔧  Setting up git hooks..."
poetry run pre-commit install -t pre-commit -t pre-push -t post-checkout -t post-merge -t post-rewrite -t commit-msg

echo ""
echo "✅ Development environment setup complete!"
echo ""
