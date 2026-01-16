#!/bin/bash
# Automated setup script for PCP Analysis Project
# This script sets up the Python environment using pyenv

set -e  # Exit on any error

echo "========================================="
echo "PCP Analysis Project - Setup Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if pyenv is installed
echo "Checking for pyenv..."
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}ERROR: pyenv is not installed!${NC}"
    echo ""
    echo "Please install pyenv first:"
    echo "  curl https://pyenv.run | bash"
    echo ""
    echo "Then add to your shell config (~/.bashrc or ~/.zshrc):"
    echo '  export PYENV_ROOT="$HOME/.pyenv"'
    echo '  export PATH="$PYENV_ROOT/bin:$PATH"'
    echo '  eval "$(pyenv init -)"'
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ pyenv found${NC}"
echo ""

# Check Python version
REQUIRED_PYTHON="3.12.0"
echo "Checking for Python ${REQUIRED_PYTHON}..."

if ! pyenv versions | grep -q "${REQUIRED_PYTHON}"; then
    echo -e "${YELLOW}Python ${REQUIRED_PYTHON} not found. Installing...${NC}"
    pyenv install ${REQUIRED_PYTHON}
    echo -e "${GREEN}✓ Python ${REQUIRED_PYTHON} installed${NC}"
else
    echo -e "${GREEN}✓ Python ${REQUIRED_PYTHON} already installed${NC}"
fi
echo ""

# Create virtual environment
ENV_NAME="pcp-analysis-env"
echo "Setting up virtual environment: ${ENV_NAME}..."

if pyenv virtualenvs | grep -q "${ENV_NAME}"; then
    echo -e "${YELLOW}Virtual environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing environment..."
        pyenv virtualenv-delete -f ${ENV_NAME}
        echo "Creating new virtual environment..."
        pyenv virtualenv ${REQUIRED_PYTHON} ${ENV_NAME}
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    echo "Creating virtual environment..."
    pyenv virtualenv ${REQUIRED_PYTHON} ${ENV_NAME}
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Set local Python version
echo "Setting local Python version to ${ENV_NAME}..."
pyenv local ${ENV_NAME}
echo -e "${GREEN}✓ Local Python version set${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install requirements
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}WARNING: requirements.txt not found!${NC}"
fi
echo ""

# Verification
echo "========================================="
echo "Setup Complete! Verifying installation..."
echo "========================================="
echo ""

echo "Python version:"
python --version
echo ""

echo "Python location:"
which python
echo ""

echo "Active pyenv environment:"
pyenv version
echo ""

echo "Installed packages:"
pip list | grep -E "(numpy|scipy|matplotlib|gudhi|scikit-learn|ripser|seaborn|tqdm)" || echo "Core packages not found!"
echo ""

echo "========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "========================================="
echo ""
echo "You can now run the analysis scripts:"
echo "  cd scripts/1_data_generation"
echo "  python generate_ensembles.py"
echo ""
echo "For more information, see README.md and SETUP.md"
echo ""

