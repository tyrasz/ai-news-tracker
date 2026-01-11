# AI News Tracker

A personalized news aggregator that learns your reading preferences using semantic embeddings and machine learning.

## Features

- **Semantic search**: Find articles about specific topics using natural language queries
- **Personalized recommendations**: Learn from your likes/dislikes to surface relevant content
- **Multiple recommendation algorithms**: Choose from For You, Explore, Deep Dive, Trending, Balanced, or Contrarian modes
- **Article clustering**: Group similar articles from different sources about the same story
- **Multi-source aggregation**: Pull from 40+ RSS feeds across tech, world news, business, science, and more
- **Web interface**: Clean web UI for browsing and managing articles
- **CLI**: Full-featured command-line interface
- **Neural re-ranker**: Optional PyTorch-based model for enhanced personalization
- **MIND training**: Train on Microsoft's MIND news recommendation dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/tyrasz/ai-news-tracker.git
cd ai-news-tracker

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# Optional: Install PyTorch for neural re-ranker
pip install torch
```

## Quick Start

```bash
# Initialize with default tech news sources
news init

# Or add all sources
news init --all

# Fetch articles
news fetch

# View personalized recommendations
news read

# Search for a specific topic
news topic artificial intelligence

# Like/dislike articles to train preferences
news like 42
news dislike 17

# Start the web interface
news serve
```

## CLI Commands

### Basic Commands

| Command | Description |
|---------|-------------|
| `news init` | Initialize database with news sources |
| `news fetch` | Download new articles from all sources |
| `news read` | Show personalized recommendations |
| `news topic <query>` | Search articles about a topic |
| `news like <id>` | Mark article as liked |
| `news dislike <id>` | Mark article as disliked |
| `news open <id>` | View article details and open in browser |
| `news stats` | Show statistics about articles and preferences |
| `news serve` | Start the web interface |

### Source Management

```bash
# List available feed categories
news categories

# Initialize with specific categories
news init -c world -c politics

# Add a custom RSS feed
news add-source "My Blog" https://example.com/rss

# List configured sources
news sources
```

### Options

```bash
# Use a different database file
news --db custom.db read

# Enable verbose logging
news --verbose fetch

# Write logs to file
news --log-file app.log fetch
```

## Feed Categories

| Category | Sources |
|----------|---------|
| tech | Hacker News, TechCrunch, Ars Technica, The Verge, Wired, MIT Tech Review |
| world | Reuters, AP News, BBC World, Al Jazeera, NPR World, The Guardian |
| us | NPR News, Reuters US, AP US, PBS NewsHour, BBC US |
| politics | Politico, The Hill, Reuters Politics, FiveThirtyEight |
| business | Reuters Business, CNBC, Bloomberg, Financial Times, Economist |
| science | Nature, Science Daily, Phys.org, New Scientist, Scientific American |
| middle_east | Al Jazeera, BBC Middle East, Reuters Middle East, The Guardian |
| europe | BBC Europe, Reuters Europe, The Guardian Europe, Euronews |
| asia | BBC Asia, Reuters Asia, The Guardian Asia, South China Morning Post |

## Recommendation Algorithms

| Algorithm | Description |
|-----------|-------------|
| `for_you` | Personalized based on your preferences (default) |
| `explore` | Discover articles outside your usual interests |
| `deep_dive` | More from topics you've recently engaged with |
| `trending` | Most recent and fresh articles |
| `balanced` | Mix of personalized content with diverse perspectives |
| `contrarian` | Articles that challenge your typical preferences |

## Web Interface

Start the web server:

```bash
news serve --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommendations` | GET | Get personalized article recommendations |
| `/api/recommendations/grouped` | GET | Get recommendations with similar articles grouped |
| `/api/topic/{query}` | GET | Search for articles about a topic |
| `/api/algorithms` | GET | List available recommendation algorithms |
| `/api/feedback` | POST | Record like/dislike feedback |
| `/api/read/{article_id}` | POST | Mark article as read |
| `/api/article/{article_id}` | GET | Get full article details |
| `/api/stats` | GET | Get statistics |
| `/api/refresh` | POST | Fetch new articles from feeds |
| `/api/cache/stats` | GET | Get embedding cache statistics |
| `/api/cache/clear` | POST | Clear the embedding cache |

## Advanced Features

### Neural Re-ranker

The re-ranker uses a small neural network to learn from your feedback and improve recommendations.

```bash
# Check re-ranker status
news reranker status

# Train on your feedback (needs 10+ samples)
news reranker train
```

### MIND Dataset Training

Train on Microsoft's MIND news recommendation dataset for better initial recommendations.

```bash
# Download the MIND dataset
news mind download --size small

# Train the NRMS model
news mind train --epochs 5 --batch-size 64

# Check training status
news mind info
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=ai_news_tracker --cov-report=term-missing

# Lint code
ruff check .
```

## Architecture

```
ai_news_tracker/
├── cli.py          # Command-line interface
├── web.py          # FastAPI web server
├── models.py       # SQLAlchemy database models
├── sources.py      # RSS feed parsing
├── embeddings.py   # Sentence embedding engine with LRU cache
├── preferences.py  # User preference learning
├── recommender.py  # Recommendation engine
├── algorithms.py   # Recommendation algorithms
├── clustering.py   # Article similarity clustering
├── reranker.py     # Neural re-ranking model
├── mind_trainer.py # MIND dataset training
└── logging_config.py # Structured logging
```

## License

MIT
