# Setup Guide for PCP Analysis Project

## Python Version Management with pyenv

This project uses **Python 3.12.0** and is managed with pyenv for version consistency across different environments.

---

## Prerequisites

### 1. Install pyenv (if not already installed)

```bash
# Check if pyenv is installed
which pyenv

# If not installed, install pyenv (Linux/macOS):
curl https://pyenv.run | bash

# Add to your shell configuration (~/.bashrc, ~/.zshrc, etc.):
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

---

## Quick Setup (Recommended)

### Option 1: Automatic Setup Script

```bash
# Clone the repository
git clone <your-repo-url>
cd PCP

# Run the setup script
bash setup.sh
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd PCP

# Install Python 3.12.0 (if not already installed)
pyenv install 3.12.0

# Create a virtual environment for this project
pyenv virtualenv 3.12.0 pcp-analysis-env

# Set it as the local Python for this directory
pyenv local pcp-analysis-env

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Verification

After setup, verify your environment:

```bash
# Check Python version (should show 3.12.0)
python --version

# Check that you're using pyenv
which python
# Should show: ~/.pyenv/shims/python

# Verify pyenv environment
pyenv version
# Should show: pcp-analysis-env (set by /path/to/PCP/.python-version)

# Check installed packages
pip list
```

---

## For New Collaborators

When you clone this repository:

1. **pyenv will automatically use Python 3.12.0** because of the `.python-version` file
2. **Create your own virtual environment**:
   ```bash
   pyenv virtualenv 3.12.0 pcp-analysis-env
   pyenv local pcp-analysis-env
   pip install -r requirements.txt
   ```

---

## Switching Between Projects

pyenv automatically switches Python versions when you `cd` into directories with `.python-version` files:

```bash
cd ~/PCP              # Automatically uses Python 3.12.0 (pcp-analysis-env)
cd ~/other-project    # Automatically uses that project's Python version
cd ~/PCP              # Back to Python 3.12.0
```

---

## Troubleshooting

### Python 3.12.0 not installed?

```bash
# Install it with pyenv
pyenv install 3.12.0
```

### Virtual environment not activating?

```bash
# Check available environments
pyenv virtualenvs

# Set local environment manually
pyenv local pcp-analysis-env
```

### Package installation fails?

```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing requirements again
pip install -r requirements.txt
```

### gudhi installation issues?

gudhi requires compilation. Ensure you have:
- gcc/g++ compiler
- Python development headers

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# Then retry
pip install gudhi==3.10.1
```

---

## Adding New Dependencies

When you install new packages:

```bash
# Install the package
pip install <package-name>

# Update requirements.txt
pip freeze > requirements.txt
```

Better approach - manual requirements.txt maintenance:

```bash
# Add only direct dependencies to requirements.txt
echo "new-package==1.2.3" >> requirements.txt
```

---

## Python Version Compatibility

This project requires **Python 3.12.0** because:
- numpy 1.26.3 works well with Python 3.12
- scipy 1.14.1 is compatible with Python 3.12
- gudhi 3.10.1 supports Python 3.12
- All other dependencies are compatible

**Minimum Python version**: 3.11.x  
**Recommended Python version**: 3.12.0 (tested and verified)  
**Maximum Python version**: 3.12.x (3.13+ may have compatibility issues)

---

## GitHub Repository Setup Checklist

When creating a new repository:

- [x] `.python-version` file (locks Python version)
- [x] `requirements.txt` (dependency list with versions)
- [x] `SETUP.md` (this file)
- [ ] `.gitignore` (exclude `__pycache__`, `*.pyc`, virtual envs, etc.)
- [ ] `README.md` (updated with setup instructions)
- [ ] Optional: `setup.sh` (automated setup script)

---

## Example: Setting Up on a New Machine

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/PCP.git
cd PCP

# 2. Check if Python 3.12.0 is available via pyenv
pyenv versions | grep 3.12.0

# 3. If not available, install it
pyenv install 3.12.0

# 4. Create a dedicated virtual environment
pyenv virtualenv 3.12.0 pcp-analysis-env

# 5. Set it as the local environment (updates .python-version)
pyenv local pcp-analysis-env

# 6. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 7. Verify setup
python --version  # Should show Python 3.12.0
pip list          # Should show all required packages

# 8. Run a test
cd scripts/1_data_generation
python generate_ensembles.py
```

---

## Summary

- **pyenv manages Python versions** (3.12.0 for this project)
- **Virtual environments isolate dependencies** (pcp-analysis-env)
- **`.python-version` file ensures consistency** across all machines
- **`requirements.txt` locks package versions** for reproducibility

This ensures that anyone cloning your repository can replicate your exact Python environment!

