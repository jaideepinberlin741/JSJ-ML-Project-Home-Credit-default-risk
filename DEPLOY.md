# Deploy to Streamlit Cloud

## The CSV Problem (Simple Explanation)

The app needs `data/processed/final_train_features.csv` (219MB) to:
- Know what features the model expects (132 features)
- Calculate default values for features not collected from the user

**Problem:** This file is in `.gitignore`, so it won't be uploaded to GitHub/Streamlit Cloud.

**Solution:** Choose one:

### Option 1: Use Git LFS (Recommended)
```bash
git lfs install
git lfs track "data/processed/final_train_features.csv"
git add .gitattributes data/processed/final_train_features.csv
git commit -m "Add feature data with Git LFS"
git push
```

### Option 2: Make the app work without it
Modify `app.py` to get feature names from the model itself (LightGBM stores them) and hardcode default values. This requires code changes.

---

## Deploy Steps

1. **Handle the CSV** (see above)

2. **Push to GitHub**
   ```bash
   git push origin main  # or your branch name
   ```

3. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo and branch
   - Main file: `app.py`
   - Click "Deploy!"

   **⚠️ Permission Issue?** If you get an error about not being an admin:
   - **Option A:** Ask a repo admin to add you as a collaborator with admin rights
   - **Option B:** Ask a repo admin to deploy it for you (they can add you as a viewer later)
   - **Option C:** Fork the repo to your own GitHub account and deploy from there
   - **Option D:** Use alternative platforms (see below)

4. **Done!** The app will be live at `https://your-app-name.streamlit.app`

---

## Alternative Deployment Options

If you can't get Streamlit Cloud access, try these:

### Railway (Free tier available)
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. New Project → Deploy from GitHub repo
4. Add Python buildpack
5. Set start command: `streamlit run app.py --server.port $PORT`

### Render (Free tier available)
1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. New → Web Service
4. Connect your repo
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Local Testing
You can always run it locally:
```bash
streamlit run app.py
```
Then share via ngrok or similar tunneling service.

---

## Files Needed

- ✅ `app.py` - Streamlit web app
- ✅ `requirements.txt` - Python dependencies (make sure it has streamlit, numpy, joblib, lightgbm)
- ✅ `models/lightgbm_model.pkl` - Trained LightGBM model (should be in Git)
- ⚠️ `data/processed/final_train_features.csv` - See "The CSV Problem" above
