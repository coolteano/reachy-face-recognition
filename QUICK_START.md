# Quick Start Guide - Git Setup

## 🚀 Get Your Project on Git in 3 Steps

### Step 1: Create Remote Repository

**GitHub:**
1. Go to https://github.com/new
2. Repository name: `reachy-face-recognition`
3. Choose Public or Private
4. **Don't** initialize with README (we have one)
5. Click "Create repository"

**GitLab:**
1. Go to https://gitlab.com/projects/new
2. Project name: `reachy-face-recognition`
3. Choose visibility
4. **Don't** initialize with README
5. Click "Create project"

### Step 2: Connect and Push

```bash
cd reachy-face-recognition

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/reachy-face-recognition.git

# Push to remote
git branch -M main  # or 'master' if preferred
git push -u origin main
```

### Step 3: Share!

Share this URL with others:
```
https://github.com/YOUR_USERNAME/reachy-face-recognition
```

## 📥 Cloning on Another Machine

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/reachy-face-recognition.git

# Navigate to project
cd reachy-face-recognition

# Install dependencies
pip install -r requirements.txt

# Start using!
python face_recognition_system.py
```

## 🔄 Updating the Repository

When you make changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

## ✅ Current Status

- ✅ Git repository initialized
- ✅ Initial commit created
- ✅ All files staged and committed
- ⏳ **Next:** Create remote repository and push

See `SETUP_GIT.md` for detailed instructions.

