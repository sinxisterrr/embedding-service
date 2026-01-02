import express from 'express';
import { pipeline } from '@xenova/transformers';

const app = express();
app.use(express.json({ limit: '10mb' }));

let embedder = null;

// Load the model on startup
async function loadModel() {
  console.log('🔄 Loading embedding model (all-MiniLM-L6-v2)...');
  const start = Date.now();

  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);
  console.log(`✅ Model loaded in ${elapsed}s`);
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: 'all-MiniLM-L6-v2',
    ready: embedder !== null,
    dimensions: 384
  });
});

// Single text embedding
app.post('/embed', async (req, res) => {
  try {
    if (!embedder) {
      return res.status(503).json({ error: 'Model not loaded yet' });
    }

    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'Missing "text" field' });
    }

    const output = await embedder(text, { pooling: 'mean', normalize: true });
    const embedding = Array.from(output.data);

    res.json({
      embedding,
      dimensions: embedding.length
    });

  } catch (error) {
    console.error('❌ Embedding error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Batch embeddings
app.post('/embed/batch', async (req, res) => {
  try {
    if (!embedder) {
      return res.status(503).json({ error: 'Model not loaded yet' });
    }

    const { texts } = req.body;

    if (!Array.isArray(texts)) {
      return res.status(400).json({ error: 'Missing "texts" array' });
    }

    const embeddings = await Promise.all(
      texts.map(async (text) => {
        const output = await embedder(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
      })
    );

    res.json({
      embeddings,
      count: embeddings.length,
      dimensions: embeddings[0]?.length || 0
    });

  } catch (error) {
    console.error('❌ Batch embedding error:', error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;

// Load model then start server
loadModel().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Embedding service listening on port ${PORT}`);
    console.log(`📊 Model: all-MiniLM-L6-v2 (384 dimensions)`);
    console.log(`🔗 Ready to serve embeddings to all bots!`);
  });
}).catch(err => {
  console.error('❌ Failed to load model:', err);
  process.exit(1);
});
