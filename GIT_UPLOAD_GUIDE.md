# Git Upload Guide - AI Video Commerce Recommender

This guide provides step-by-step instructions for uploading your project to GitHub.

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Git installed** on your computer
   ```bash
   git --version
   ```
   If not installed, download from: https://git-scm.com/

2. **GitHub account** created at: https://github.com/

3. **New GitHub repository** created:
   - Go to https://github.com/new
   - Repository name: `AI-Video-Commerce` (or your preferred name)
   - Description: "AI-powered video commerce recommendation system"
   - Choose: Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

## 🚀 Step-by-Step Upload Instructions

### Step 1: Navigate to Your Project Directory

Open your terminal and navigate to the project folder:

```bash
cd /Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code
```

### Step 2: Verify Files

Check that all important files are present:

```bash
ls -la
```

You should see:
- ✅ `.gitignore` - Git ignore rules
- ✅ `README.md` - Project documentation
- ✅ `LICENSE` - MIT License
- ✅ `env.example` - Environment template
- ✅ `Dockerfile` - Docker configuration
- ✅ `docker-compose.yml` - Docker Compose setup
- ✅ All Python files (.py)
- ✅ `Code.Frontend/` folder with frontend code
- ✅ `requirements.txt` - Python dependencies

### Step 3: Initialize Git Repository

```bash
# Initialize a new Git repository
git init
```

You should see: `Initialized empty Git repository`

### Step 4: Configure Git (First Time Only)

If this is your first time using Git, configure your identity:

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"
```

### Step 5: Add Remote Repository

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name:

```bash
# HTTPS (recommended for most users)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Or SSH (if you have SSH keys set up)
# git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/AI-Video-Commerce.git
```

Verify the remote was added:
```bash
git remote -v
```

### Step 6: Add Files to Staging Area

Add all files (respecting .gitignore):

```bash
git add .
```

Check what will be committed:

```bash
git status
```

You should see files in green (staged for commit). The .gitignore file will automatically exclude unnecessary files like:
- `venv/` folders
- `__pycache__/` directories
- `.env` files (secrets)
- Log files
- Temporary files
- PDF documentation files
- Incomplete downloads

### Step 7: Create Your First Commit

```bash
git commit -m "Initial commit: AI-Powered Video Commerce Recommender System

- Complete backend implementation with FastAPI
- Video processing with CLIP and OCR
- Vector search with FAISS
- Collaborative filtering recommendation engine
- Neural ranking model
- Redis-based feature store
- React frontend with modern UI
- Docker deployment support
- Comprehensive documentation"
```

### Step 8: Rename Branch to 'main' (GitHub Standard)

```bash
# Rename the default branch from 'master' to 'main'
git branch -M main
```

### Step 9: Push to GitHub

Push your code to GitHub:

```bash
git push -u origin main
```

You may be prompted to authenticate:
- **HTTPS**: Enter your GitHub username and password (or personal access token)
- **SSH**: Should work automatically if SSH keys are configured

### Step 10: Verify Upload

1. Go to your GitHub repository URL: `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`
2. You should see all your files
3. The README.md will be displayed on the main page
4. Verify the project looks correct

## 🔐 Authentication Options

### Option A: Personal Access Token (Recommended for HTTPS)

If GitHub prompts for a password:

1. Go to GitHub Settings: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: "AI Video Commerce Upload"
4. Select scopes: ✅ `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing

### Option B: SSH Keys (Advanced)

For more convenient authentication:

1. Generate SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```

2. Copy public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. Add to GitHub: https://github.com/settings/ssh/new
4. Use SSH remote URL: `git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git`

## 📝 Common Git Commands for Future Updates

### Making Changes

```bash
# Check status
git status

# Add specific files
git add filename.py

# Add all changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

### Viewing History

```bash
# View commit history
git log

# View recent commits (pretty format)
git log --oneline --graph --decorate

# View changes
git diff
```

### Branching (For Features)

```bash
# Create new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# List branches
git branch

# Merge branch
git merge feature/new-feature

# Delete branch
git branch -d feature/new-feature
```

## 🔄 Updating Your GitHub Repository

When you make changes:

```bash
# 1. Check what changed
git status

# 2. Add changes
git add .

# 3. Commit with message
git commit -m "Add new feature: describe what you changed"

# 4. Push to GitHub
git push
```

## ⚠️ Important Notes

### Files That WON'T Be Uploaded (Thanks to .gitignore)

- ❌ `venv/` - Virtual environment (too large)
- ❌ `__pycache__/` - Python cache
- ❌ `.env` - Environment secrets
- ❌ `logs/` - Log files
- ❌ `*.pyc` - Compiled Python files
- ❌ `node_modules/` - Node dependencies (too large)
- ❌ Model files (`.pt`, `.pkl`, `.h5`) - Too large
- ❌ PDF documentation files
- ❌ Incomplete downloads (`.crdownload`)

### Files That WILL Be Uploaded

- ✅ All Python source code (`.py`)
- ✅ Frontend code (`.js`, `.tsx`, `.css`, `.html`)
- ✅ Configuration files (`config.py`, `package.json`, etc.)
- ✅ Documentation (`README.md`, `LICENSE`)
- ✅ Docker files (`Dockerfile`, `docker-compose.yml`)
- ✅ Dependencies (`requirements.txt`, `package.json`)
- ✅ Scripts (`startup.sh`)
- ✅ Environment template (`env.example`)

## 🆘 Troubleshooting

### Problem: "fatal: not a git repository"

**Solution:** Make sure you're in the correct directory and have run `git init`

```bash
cd /Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code
git init
```

### Problem: "remote origin already exists"

**Solution:** Remove the old remote and add the new one

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Problem: "failed to push some refs"

**Solution:** Pull first, then push

```bash
git pull origin main --rebase
git push origin main
```

### Problem: "Permission denied (publickey)"

**Solution:** Use HTTPS instead of SSH, or set up SSH keys properly

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push origin main
```

### Problem: "Large files warning"

**Solution:** Large files are already excluded by .gitignore. If you see this warning:

```bash
# Remove large files from staging
git rm --cached path/to/large/file

# Or reset and re-add
git reset
git add .
git commit -m "Your message"
```

### Problem: "Username/password authentication deprecated"

**Solution:** Use a Personal Access Token instead of password (see Authentication Options above)

## 📱 After Uploading

### Make Your Repository More Attractive

1. **Add repository description** on GitHub
2. **Add topics/tags**: machine-learning, recommendation-system, fastapi, react, ai, computer-vision
3. **Pin important repositories** on your profile
4. **Enable GitHub Pages** (optional) for documentation
5. **Add a repository image** (optional)

### Share Your Project

Your repository URL will be:
```
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
```

Share it on:
- LinkedIn
- Twitter
- Your portfolio website
- Your resume/CV

## 🎓 Next Steps

1. ✅ **Star your repository** to show it's important
2. ✅ **Watch the repository** for notifications
3. ✅ **Create a GitHub Project** for task management
4. ✅ **Set up CI/CD** with GitHub Actions (optional)
5. ✅ **Add collaborators** if working with a team
6. ✅ **Create releases** when you reach milestones
7. ✅ **Enable Discussions** for community engagement

## 📚 Useful Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com/
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Markdown Guide**: https://www.markdownguide.org/

---

**Need Help?** Open an issue on your repository or refer to the GitHub documentation.

**Good luck with your project! 🚀**

