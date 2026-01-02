import express from 'express';
import { pipeline } from '@xenova/transformers';

const app = express();
app.use(express.json({ limit: '10mb' }));

let embedder = null;

// Server-side cache for embeddings
const embeddingCache = new Map();
const MAX_CACHE_SIZE = 5000;

// Performance tracking
let requestCount = 0;
let cacheHits = 0;

// Cache helper
function getCacheKey(text) {
  return text.slice(0, 300); // Use first 300 chars as key
}

function cacheGet(key) {
  if (embeddingCache.has(key)) {
    cacheHits++;
    return embeddingCache.get(key);
  }
  return null;
}

function cacheSet(key, value) {
  if (embeddingCache.size >= MAX_CACHE_SIZE) {
    // Remove oldest entry (first key)
    const firstKey = embeddingCache.keys().next().value;
    if (firstKey) {
      embeddingCache.delete(firstKey);
    }
  }
  embeddingCache.set(key, value);
}

// Load the model on startup
async function loadModel() {
  console.log('🔄 Loading embedding model (all-MiniLM-L6-v2)...');
  const start = Date.now();

  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);
  console.log(`✅ Model loaded in ${elapsed}s`);
}

// Generate embedding with caching
async function generateEmbedding(text) {
  const cacheKey = getCacheKey(text);

  // Check cache first
  const cached = cacheGet(cacheKey);
  if (cached) {
    return cached;
  }

  // Generate new embedding
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  const embedding = Array.from(output.data);

  // Cache it
  cacheSet(cacheKey, embedding);

  return embedding;
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: 'all-MiniLM-L6-v2',
    ready: embedder !== null,
    dimensions: 384,
    stats: {
      requests: requestCount,
      cacheHits,
      cacheSize: embeddingCache.size,
      hitRate: requestCount > 0 ? (cacheHits / requestCount * 100).toFixed(1) + '%' : '0%'
    }
  });
});

// Single text embedding
app.post('/embed', async (req, res) => {
  const start = Date.now();

  try {
    if (!embedder) {
      return res.status(503).json({ error: 'Model not loaded yet' });
    }

    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'Missing "text" field' });
    }

    requestCount++;
    const embedding = await generateEmbedding(text);

    const elapsed = Date.now() - start;
    console.log(`✅ Embed: ${elapsed}ms (${embedding.length} dims, cache: ${embeddingCache.size})`);

    res.json({
      embedding,
      dimensions: embedding.length
    });

  } catch (error) {
    console.error('❌ Embedding error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Batch embeddings with parallel processing + caching
app.post('/embed/batch', async (req, res) => {
  const start = Date.now();

  try {
    if (!embedder) {
      return res.status(503).json({ error: 'Model not loaded yet' });
    }

    const { texts } = req.body;

    if (!Array.isArray(texts)) {
      return res.status(400).json({ error: 'Missing "texts" array' });
    }

    requestCount += texts.length;

    // Process all in parallel with caching
    const embeddings = await Promise.all(
      texts.map(text => generateEmbedding(text))
    );

    const elapsed = Date.now() - start;
    const avgTime = (elapsed / texts.length).toFixed(1);
    console.log(`✅ Batch: ${texts.length} embeddings in ${elapsed}ms (avg ${avgTime}ms each, cache: ${embeddingCache.size})`);

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

// Clear cache endpoint (optional - for debugging)
app.post('/cache/clear', (req, res) => {
  embeddingCache.clear();
  cacheHits = 0;
  requestCount = 0;
  console.log('🧹 Cache cleared');
  res.json({ status: 'cache cleared' });
});

const PORT = process.env.PORT || 3000;

// Load model then start server
loadModel().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Embedding service listening on port ${PORT}`);
    console.log(`📊 Model: all-MiniLM-L6-v2 (384 dimensions)`);
    console.log(`💾 Cache: ${MAX_CACHE_SIZE} entries max`);
    console.log(`🔗 Ready to serve embeddings to all bots!`);
  });
}).catch(err => {
  console.error('❌ Failed to load model:', err);
  process.exit(1);
});
