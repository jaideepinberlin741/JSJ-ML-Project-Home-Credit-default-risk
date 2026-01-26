# How to Make Someone an Admin on GitHub

## For Repository Admins

If you're already an admin of the repository, here's how to add someone else:

### Method 1: Add as Collaborator with Admin Access

1. **Go to your repository on GitHub**
   - Navigate to: `https://github.com/your-org-or-username/repo-name`

2. **Click on "Settings"** (top right, in the repository menu)

3. **Click "Collaborators and teams"** (left sidebar)
   - Or go directly to: `https://github.com/your-org-or-username/repo-name/settings/access`

4. **Click "Add people"** button

5. **Enter the GitHub username or email** of the person you want to add

6. **Select permission level:**
   - Choose **"Admin"** from the dropdown
   - This gives them full control (can deploy, manage settings, etc.)

7. **Click "Add [username] to this repository"**

8. **They'll receive an email invitation** - they need to accept it

### Method 2: For Organization Repositories

If the repo is under an organization:

1. Go to repository **Settings** → **Collaborators and teams**
2. Click **"Add teams"** or **"Invite a collaborator"**
3. Select **"Admin"** permission
4. Send invitation

---

## Permission Levels Explained

- **Read**: Can view and clone
- **Triage**: Can manage issues and PRs
- **Write**: Can push code, but can't change settings
- **Maintain**: Can manage some settings, but not delete repo
- **Admin**: Full control (can deploy, change settings, delete repo) ✅ **This is what you need for Streamlit Cloud**

---

## For the Person Being Added

Once you receive the invitation:

1. **Check your email** for the GitHub invitation
2. **Or go to**: `https://github.com/notifications`
3. **Click "Accept"** on the repository invitation
4. **You now have admin access!**

---

## Quick Check: Are You Already an Admin?

1. Go to your repository on GitHub
2. Look for the **"Settings"** tab (should be visible in the top menu)
3. Click on **"Settings"**
4. **If you can see all settings** (including options like "Collaborators and teams", "Branches", "Danger Zone"), you're an admin ✅
5. **If "Settings" is missing** or you get a 404/access denied, you're not an admin ❌

**What permission level do you have?**
- Can you see Settings? → Check what you can do:
  - Can you add collaborators? → You're Admin ✅
  - Can you see "Collaborators" but can't add people? → You're Write/Maintain
  - Can't see Settings at all? → You're Read/Triage

---

## Troubleshooting

**"I'm already on the repo but can't deploy"**
- You likely have "Write" or "Maintain" access, not "Admin"
- **Solution:** Ask a current admin to upgrade your permission to "Admin"
- They can do this in Settings → Collaborators → find your name → change permission to "Admin"

**"I don't see Settings"**
- You're not an admin - ask someone who is to add you or upgrade your permission

**"Settings exists but I can't add collaborators"**
- You have "Write" or "Maintain" access, not "Admin"
- Ask a current admin to upgrade your permission to "Admin"

**"I'm in an organization and can't add people"**
- Organization admins control member access
- Ask an organization owner/admin to add you

---

## Alternative: Fork the Repository

If you can't get admin access, you can:

1. **Fork the repository** to your own GitHub account
2. You'll be the owner of your fork
3. Deploy from your fork on Streamlit Cloud
4. You'll have full control

**To fork:**
- Go to the repository
- Click the **"Fork"** button (top right)
- Select your account
- Done! You now have your own copy with admin rights
