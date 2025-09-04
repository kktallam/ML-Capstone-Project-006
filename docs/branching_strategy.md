# Branching Strategy

This document describes how our team manages branches, commits, and pull requests in the shared GitHub repository.

---

## Branch Types

- **`main`**  
  - Protected branch.  
  - Always stable and deployable.  
  - Only updated through pull requests (PRs) that pass all checks.  

- **Feature branches** (`feature/<short-description>`)  
  - Used for developing new features or enhancements.  
  - Branched off `main`.  
  - Example: `feature/add-user-auth`.  

- **Fix branches** (`fix/<short-description>`)  
  - Used for bug fixes or small patches.  
  - Branched off `main`.  
  - Example: `fix/fix-login-error`.  

- **Release tags**  
  - We tag stable versions of `main` using semantic versioning (`v1.0.0`, `v1.1.0`, etc.).

---

## Workflow

1. **Create a branch** from `main`:  
   ```bash
   git checkout -b feature/<slug>

2. **Push Changes** from local repo to your target branch:
   git add .
   git commit -m "Your Commit Message"
   git push origin target-branch-name
