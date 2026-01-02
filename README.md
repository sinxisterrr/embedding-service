# 🚀 Shared Embedding Service

One embedding model, all your bots can use it!

## What This Does

- Runs `all-MiniLM-L6-v2` (384 dimensions)
- All 9+ Discord bots call this service
- Railway internal networking (free)
- Cost: ~$2-3/month total

## Deploy to Railway

```bash
cd embedding-service
railway init
railway up
```

Get your internal URL: `embedding-service.railway.internal`

## Usage from Your Bots

```typescript
const EMBEDDING_URL = process.env.EMBEDDING_SERVICE_URL || 'http://embedding-service.railway.internal:3000';

// Single text
const response = await fetch(`${EMBEDDING_URL}/embed`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'your text here' })
});
const { embedding } = await response.json();

// Batch
const response = await fetch(`${EMBEDDING_URL}/embed/batch`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ texts: ['text1', 'text2', 'text3'] })
});
const { embeddings } = await response.json();
```

## Endpoints

- `GET /health` - Check if model is loaded
- `POST /embed` - Single text embedding
- `POST /embed/batch` - Batch embeddings

## Environment Variables

None needed! Just deploy and go.

## Resource Usage

- RAM: ~512MB
- CPU: Low (only spikes during embeddings)
- Disk: ~100MB (model cache)
- Network: Internal Railway (free)

## Cost

- Railway Hobby: Included in your plan
- Or: ~$2-3/month if separate service

**Way cheaper than any API!**
