# 🚀 Deploy Embedding Service to Railway

## Quick Deploy

1. **Create new GitHub repo** (or push to existing)

```bash
cd /home/sinxisterrr/embedding-service

# Option A: Push to new GitHub repo
gh repo create embedding-service --public --source=. --remote=origin --push

# Option B: Add remote manually
git remote add origin https://github.com/YOUR_USERNAME/embedding-service.git
git branch -M main
git push -u origin main
```

2. **Deploy to Railway**

Go to [railway.app](https://railway.app) and:
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose `embedding-service`
- Railway will auto-detect Node.js and deploy!

**OR** use Railway CLI:

```bash
railway init
railway up
```

3. **Get your internal URL**

Railway will give you: `embedding-service.railway.internal`

4. **Add env var to your 3 bots**

In Railway dashboard for each bot, add:

```
EMBEDDING_SERVICE_URL=http://embedding-service.railway.internal:3000
```

---

## That's It!

Your embedding service is now running and all 3 bots can use it for semantic search! 🎉

**Cost:** ~$2-3/month for all 3 bots to share
