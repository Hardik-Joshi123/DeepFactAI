# Getting Started Guide

Complete guide to set up and run the Fake News Detection System.

## Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 18 or higher
- **npm/pnpm**: Latest version
- **Git**: For version control
- **GPU** (Optional): CUDA 11.8+ for faster training

Check your versions:
```bash
python --version
node --version
npm --version
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fake-news-detection
```

### 2. Backend Setup

#### Option A: Quick Setup with Make
```bash
make install
```

#### Option B: Manual Setup

**Install Python dependencies:**
```bash
cd backend
pip install -e .
# or with uv (faster)
uv pip install -e .
cd ..
```

**The pyproject.toml includes:**
- FastAPI and Uvicorn
- PyTorch and Transformers
- LIME for explainability
- scikit-learn, NumPy, Pandas
- And more...

### 3. Frontend Setup

```bash
cd frontend
pnpm install
# or npm install
cd ..
```

### 4. Environment Configuration

```bash
# Copy example env file
cp .env.example .env.local

# Edit .env.local with your settings (optional for development)
```

## Running the Application

### Option A: Full Stack with Vercel (Recommended)

```bash
make dev
# or directly
vercel dev
```

This starts:
- Backend API on `http://localhost:8000`
- Frontend on `http://localhost:3000`
- Both services reload on code changes

### Option B: Individual Services

**Terminal 1 - Backend:**
```bash
make backend
# or directly
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
make frontend
# or directly
cd frontend && npm run dev
```

### Option C: Docker Deployment

```bash
# Build image
make docker

# Run container
make docker-run

# Or manually
docker build -t fake-news-detector:latest .
docker run -p 3000:3000 -p 8000:8000 fake-news-detector:latest
```

## Training Models

### Dataset Setup

1. **Download a fake news dataset** (choose one):
   - [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   - [UCI Fake News](https://archive.ics.uci.edu/dataset/311)
   - [LIAR Dataset](https://www.cs.uvm.edu/~iDB/fake-news/)

2. **Place data files** in `data/` directory:
   ```
   data/
   ├── fake_news.csv
   ├── real_news.csv
   └── combined.csv
   ```

### Training LSTM Model

```bash
make train-lstm
# or
python scripts/train_lstm.py --epochs 10 --batch_size 32
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--save_path`: Where to save model (default: models/lstm_model.pth)

### Training BERT Model

```bash
make train-bert
# or
python scripts/train_bert.py --epochs 5 --batch_size 8
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 8, adjust for GPU memory)
- `--max_length`: Max sequence length (default: 512)

### Evaluate Models

```bash
make evaluate
# or
python scripts/evaluate_models.py
```

Shows metrics for both LSTM and BERT models.

## Using Jupyter Notebooks

### Launch Jupyter

```bash
jupyter notebook
```

### Available Notebooks

1. **01_data_exploration.ipynb**
   - Load and explore dataset
   - Analyze class distribution
   - Text statistics and preprocessing
   - Vocabulary analysis

2. **02_lstm_training.ipynb**
   - LSTM model architecture
   - Training pipeline
   - Performance metrics
   - Confusion matrix

3. **03_bert_training.ipynb**
   - BERT fine-tuning
   - HuggingFace Transformers
   - Advanced training techniques
   - Model evaluation

4. **04_explainability_demo.ipynb**
   - LIME explanations
   - Word importance visualization
   - Model interpretation
   - Case studies

## API Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your news article here",
    "model": "lstm"
  }'

# Compare models
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your news article here"
  }'
```

### Using Python

```python
import requests

# Prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'text': 'Breaking news about climate change discoveries...',
        'model': 'bert'
    }
)
print(response.json())

# Response structure:
# {
#   "prediction": "REAL",
#   "confidence": 0.95,
#   "model": "bert",
#   "word_importance": {
#     "breaking": 0.85,
#     "news": 0.72,
#     ...
#   }
# }
```

## Troubleshooting

### Issue: Backend server not starting

**Error**: `Address already in use`
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
# Then restart
make backend
```

### Issue: Frontend build errors

**Error**: `Module not found`
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

### Issue: CUDA out of memory during training

**Solution**: Reduce batch size
```bash
python scripts/train_bert.py --batch_size 4
```

### Issue: Models not found

**Error**: `FileNotFoundError: models/lstm_model.pth`
```bash
# Setup model configurations
python scripts/setup_models.py
```

### Issue: Import errors in backend

**Solution**: Reinstall backend in editable mode
```bash
cd backend
pip install -e . --force-reinstall
```

## Development Workflow

### Adding a New Component

1. **Create component** in `frontend/app/components/`:
```typescript
export function MyComponent() {
  return <div>My component</div>
}
```

2. **Import in page** or parent component:
```typescript
import { MyComponent } from '@/app/components/MyComponent'
```

3. **Styling**: Use Tailwind CSS classes

### Adding a New API Endpoint

1. **Add endpoint** in `backend/main.py`:
```python
@app.post("/new-endpoint")
async def new_endpoint(request: MySchema):
    # Implementation
    return {"result": "success"}
```

2. **Call from frontend**:
```typescript
const response = await fetch('/api/new-endpoint', {
  method: 'POST',
  body: JSON.stringify(data)
})
```

## Performance Optimization

### Frontend
- Use SWR for efficient data fetching
- Implement code splitting for large components
- Optimize images with Next.js Image component

### Backend
- Use GPU for model inference
- Implement caching with Redis (optional)
- Batch predictions for higher throughput

### Models
- Use quantization for smaller models
- Implement model distillation
- Use ONNX for faster inference

## Deployment

### Vercel Deployment

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Connect to Vercel**:
   - Go to vercel.com
   - Import GitHub repository
   - Select Framework Preset: **Services**
   - Deploy

3. **Set environment variables** in Vercel dashboard

### Docker Deployment

```bash
# Build
docker build -t myregistry/fake-news-detector:latest .

# Push
docker push myregistry/fake-news-detector:latest

# Deploy on any container platform
```

## Monitoring & Logging

### View Backend Logs

```bash
# With uvicorn
uvicorn main:app --log-level debug

# Check logs in terminal where server is running
```

### View Frontend Logs

```bash
# Check browser console (F12)
# Or terminal where `npm run dev` is running
```

## Next Steps

1. **Download dataset** and train models
2. **Test predictions** with sample news
3. **Explore notebooks** for deeper understanding
4. **Deploy** to production
5. **Monitor** and improve model performance

## Support

- Check README.md for API documentation
- Review PROJECT_SUMMARY.md for architecture overview
- Check notebooks for examples and tutorials
- Open issues on GitHub for bugs

## Performance Tips

- Use `make dev` for fastest development
- Train on GPU if available
- Use caching to reduce API calls
- Implement pagination for large datasets
- Use batch predictions for bulk processing

Happy coding! 🚀
