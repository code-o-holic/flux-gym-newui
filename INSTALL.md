# FluxGym New UI - Installation Guide

## Clone with Submodules

**Important:** This repository uses `sd-scripts` as a git submodule. Always clone with submodules:

### Fresh Clone
```bash
git clone --recurse-submodules https://github.com/code-o-holic/flux-gym-newui.git
cd flux-gym-newui
```

### If You Already Cloned (without submodules)
```bash
git submodule update --init --recursive
```

## Branch Structure

- **`stable`** - Production-ready releases, tested features
- **`dev`** - Development branch with latest features (may be unstable)

### Clone Specific Branch
```bash
# Clone stable branch (recommended for production)
git clone --recurse-submodules -b stable https://github.com/code-o-holic/flux-gym-newui.git

# Clone dev branch (for latest features)
git clone --recurse-submodules -b dev https://github.com/code-o-holic/flux-gym-newui.git
```

## Setup Environment

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the New UI:**
   ```bash
   python ui_new/app.py
   ```

   The interface will be available at: http://localhost:7861

## Submodule Details

- **sd-scripts**: Pinned to commit `f5d44fd` from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- **Purpose**: Provides training scripts and utilities for FLUX model fine-tuning
- **Pinning**: Ensures reproducible builds across different environments

## Troubleshooting

### Submodule Issues
If you encounter submodule problems:
```bash
git submodule deinit --all
git submodule update --init --recursive
```

### Missing sd-scripts
If the `sd-scripts` folder is empty:
```bash
git submodule update --init --recursive
```

### Update Submodules
To update to the latest pinned version:
```bash
git submodule update --remote
git add sd-scripts
git commit -m "Update sd-scripts submodule"
```
