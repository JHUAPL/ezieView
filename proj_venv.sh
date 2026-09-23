#!/usr/bin/env bash

# Install uv if not already on system
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Update uv itself to the latest stable version
echo "Updating uv..."
uv self update

# Create venv if not present
echo "Checking venv status..."
[ -d .venv ] || uv venv -p 3.12
# shellcheck disable=SC1091
. ./.venv/bin/activate

# Add zensical and ruff to the development dependency group if not already present
echo "Checking tool installation..."
has_zensical=$(grep zensical pyproject.toml | cut -d'"' -f 2)
if [ -z "$has_zensical" ]; then
    echo "Installing zensical..."
    uv add --dev zensical
else
    echo "Has $has_zensical"
fi

has_ruff=$(grep "ruff" pyproject.toml | grep -v tool | cut -d'"' -f 2)
if [ -z "$has_ruff" ]; then
    echo "Installing ruff..."
    uv add --dev ruff
else
    echo "Has $has_ruff"
fi

has_mkdocstrings=$(grep "mkdocstrings" pyproject.toml | grep -v tool | cut -d'"' -f 2)
if [ -z "$has_mkdocstrings" ]; then
    echo "Installing mkdocstrings..."
    uv add --dev mkdocstrings
    uv add --dev "mkdocstrings[python]"
    uv add --dev mkdocs-material
    uv add --dev mkdocs-to-pdf
else
    echo "Has $has_mkdocstrings"
fi

# [Re-]Install ezieview
# Optional env prefix for DMZ systems: UV_CACHE_DIR=/project/ezie/.cache/uv
uv pip install \
    --system-certs \
    --index "https://pypi.org/simple" \
    -e  .

# Initialize zensical if this has not already been done
# [ -f zensical.toml ] || zensical new .
