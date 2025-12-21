# Setting Up Git Repository for ReachyMini Face Recognition

This guide will help you create a Git repository and push it to GitHub/GitLab so others can clone it.

## Step 1: Initialize Git Repository (Already Done)

The repository has been initialized. If you need to do it again:

```bash
cd reachy-face-recognition
git init
```

## Step 2: Add Files to Git

```bash
# Add all files
git add .

# Check what will be committed
git status
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: ReachyMini face recognition system"
```

## Step 4: Create Remote Repository

### Option A: GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Repository name: `reachy-face-recognition` (or your preferred name)
4. Description: "Face recognition system for ReachyMini robot"
5. Choose Public or Private
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### Option B: GitLab

1. Go to [GitLab](https://gitlab.com) and sign in
2. Click "New project" → "Create blank project"
3. Project name: `reachy-face-recognition`
4. Visibility: Public or Private
5. **DO NOT** initialize with README
6. Click "Create project"

## Step 5: Connect Local Repository to Remote

After creating the remote repository, GitHub/GitLab will show you commands. Use these:

### For GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/reachy-face-recognition.git
```

### For GitLab:
```bash
git remote add origin https://gitlab.com/YOUR_USERNAME/reachy-face-recognition.git
```

Replace `YOUR_USERNAME` with your actual username.

## Step 6: Push to Remote

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

If you're using an older Git version or prefer master:
```bash
git branch -M master
git push -u origin master
```

## Step 7: Verify

Visit your repository URL to verify all files are uploaded:
- GitHub: `https://github.com/YOUR_USERNAME/reachy-face-recognition`
- GitLab: `https://gitlab.com/YOUR_USERNAME/reachy-face-recognition`

## Cloning the Project (For Others)

Others can now clone your project:

```bash
# GitHub
git clone https://github.com/YOUR_USERNAME/reachy-face-recognition.git

# GitLab
git clone https://gitlab.com/YOUR_USERNAME/reachy-face-recognition.git

# Then install dependencies
cd reachy-face-recognition
pip install -r requirements.txt
```

## Updating the Repository

When you make changes:

```bash
# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to remote
git push
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:

**Option 1: Use Personal Access Token (Recommended)**
1. GitHub: Settings → Developer settings → Personal access tokens → Generate new token
2. GitLab: Preferences → Access Tokens → Create personal access token
3. Use the token as password when pushing

**Option 2: Use SSH**
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to GitHub/GitLab (copy ~/.ssh/id_ed25519.pub)
# Then use SSH URL:
git remote set-url origin git@github.com:YOUR_USERNAME/reachy-face-recognition.git
```

### Large Files

If you have large files (like video samples), consider:
- Using Git LFS: `git lfs install && git lfs track "*.mp4"`
- Or add them to `.gitignore` and host elsewhere

## Next Steps

1. ✅ Initialize Git (done)
2. ✅ Add files (ready to run `git add .`)
3. ✅ Create commit (ready to run `git commit`)
4. ⏳ Create remote repository on GitHub/GitLab
5. ⏳ Push to remote
6. ⏳ Share the repository URL with others

