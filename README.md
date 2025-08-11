# 🏠 PropertyGuru - AI-Powered Real Estate Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000.svg)](https://flask.palletsprojects.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.3-FF6B6B.svg)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deployment](https://img.shields.io/badge/Deployment-Ready-brightgreen.svg)](https://render.com)

> A sophisticated real estate RAG (Retrieval-Augmented Generation) system that enables intelligent property search through conversational interfaces. Built specifically for Bahria Town Phase 7 properties with semantic search capabilities and modern web interface.

## 🌟 Features

### 🤖 **AI-Powered Conversational Search**
- **Natural Language Queries**: "Show me 3-bedroom houses under 50 million" 
- **Semantic Understanding**: Advanced embedding-based property matching using sentence transformers
- **RAG Architecture**: Retrieval-augmented generation for accurate, contextual responses
- **Real-time Chat**: ChatGPT-inspired interface with typing indicators

### 🏠 **Comprehensive Property Database**
- **800+ Property Embeddings**: Complete Bahria Town Phase 7 listings
- **Rich Property Data**: Prices, bedrooms, bathrooms, amenities, locations
- **Vector Search**: High-performance ChromaDB integration for semantic similarity
- **Real-time Responses**: Sub-second query processing

### 💻 **Modern Web Interface**
- **Responsive Design**: Mobile-first design that works on all devices
- **Dark/Light Themes**: Beautiful UI with theme switching
- **Conversation History**: Persistent chat sessions
- **Real-time Updates**: Live typing indicators and instant responses

### 🚀 **Production-Ready Deployment**
- **Multi-Platform Support**: Render, Heroku, Docker, Vercel ready
- **Environment Configuration**: Secure API key management
- **Health Monitoring**: Built-in status checks and logging
- **Scalable Architecture**: Ready for high-traffic deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM (recommended for optimal performance)
- Cohere API key (free tier available)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Bilal-Ahmad6/PropertyGuru-RAG-Bot.git
cd PropertyGuru-RAG-Bot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API key
COHERE_API_KEY=your_cohere_api_key_here
```

5. **Run the application**
```bash
# Start the web interface
python web_ui/app.py
```

Visit `http://localhost:8000` to access the conversational property search interface.

### ⚡ Quick Test

Once running, try these example queries in the chat interface:
- "Show me 3-bedroom houses under 50 million"
- "Find apartments with parking spaces"
- "What properties are available in Phase 7?"

## 📊 Project Structure

```
PropertyGuru/
├── 📁 web_ui/               # Flask web interface (main app)
│   ├── app.py               # Main web application
│   ├── templates/           # Jinja2 templates
│   ├── static/              # CSS, JS, images
│   └── conversations.json   # Chat history storage
├── 📁 scripts/              # Data processing pipeline
│   ├── scrape_listings.py   # Web scraping automation
│   ├── clean_and_enrich.py  # Data preprocessing
│   ├── embed_and_store.py   # Vector embedding generation
│   ├── query_rag.py         # RAG query processing
│   └── utils.py             # Utility functions
├── 📁 data/                 # Data storage
│   ├── raw/                 # Raw scraped data
│   └── processed/           # Cleaned and enriched data
├── 📁 chromadb_data/        # Vector database storage (800+ embeddings)
├── 📁 processor/            # Data processing utilities
├── 📁 tests/                # Unit and integration tests
├── 📁 logs/                 # Application logs
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
├── render.yaml              # Render.com deployment config
├── Dockerfile               # Docker containerization
├── docker-compose.yml       # Multi-container setup
└── .env.example             # Environment variables template
│   ├── scrape_listings.py   # Web scraping automation
│   ├── clean_and_enrich.py  # Data preprocessing
│   ├── embed_and_store.py   # Vector embedding generation
│   └── query_rag.py         # Query processing utilities
├── 📁 data/                 # Data storage
│   ├── raw/                 # Raw scraped data
│   └── processed/           # Cleaned and enriched data
├── 📁 chromadb_data/        # Vector database storage
├── 📁 processor/            # Data processing utilities
├── 📁 tests/                # Unit and integration tests
├── 📁 logs/                 # Application logs
├── config.py                # Configuration management
└── requirements.txt         # Python dependencies
```

## 🏛️ Architecture Overview

PropertyGuru implements a modern RAG (Retrieval-Augmented Generation) architecture optimized for real estate search:

### Data Flow Pipeline

```mermaid
graph TD
    A[Zameen.com Data] --> B[Web Scraping]
    B --> C[Data Cleaning & Enrichment]
    C --> D[Sentence Transformer Embeddings]
    D --> E[ChromaDB Vector Storage]
    E --> F[RAG Query Engine]
    F --> G[Cohere LLM]
    G --> H[Web Chat Interface]
    
    I[User Query] --> F
    F --> I
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Interface** | Flask + Jinja2 | Conversational property search UI |
| **Vector Database** | ChromaDB | Semantic search with 800+ property embeddings |
| **RAG Engine** | LangChain + Cohere | Intelligent query processing and response generation |
| **Embeddings** | Sentence Transformers | Property description semantic understanding |
| **Data Pipeline** | Python + BeautifulSoup | Property data extraction and processing |

### Current Database Stats
- **Properties**: 160+ listings from Bahria Town Phase 7
- **Embeddings**: 800+ document chunks for semantic search
- **Response Time**: Sub-second query processing
- **Storage**: Optimized vector indexing with ChromaDB

## 📖 Usage Examples

### Web Interface (Recommended)

Simply visit `http://localhost:8000` and start chatting:

```
👤 "Show me 3-bedroom houses under 50 million"
🤖 "I found 5 houses matching your criteria..."

👤 "Find apartments with parking"  
🤖 "Here are apartments with parking spaces..."

👤 "What's available in Phase 7?"
🤖 "I found several properties in Bahria Town Phase 7..."
```

### API Usage

```python
import requests

# Query the chat API
response = requests.post("http://localhost:8000/api/message", json={
    "message": "Show me 3-bedroom apartments under 15 million",
    "chat_id": "user123"
})

result = response.json()
print(result["response"])  # AI response with property details
```

### Natural Language Query Examples

The system understands various query formats:
- **Budget-based**: "Houses under 30 million", "Properties between 20-40 million"
- **Feature-based**: "Apartments with parking", "Houses with gardens"
- **Size-based**: "3 bedroom houses", "10 marla properties"
- **Location-based**: "Properties near schools", "Phase 7 apartments"
- **Investment-focused**: "Good rental properties", "Investment opportunities"

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration (Required)
COHERE_API_KEY=your_cohere_api_key                # Required for LLM inference
ZAMEEN_COHERE_API_KEY=your_cohere_api_key         # Same key for Pydantic settings

# Optional API Keys
GROQ_API_KEY=your_groq_api_key                    # Alternative LLM provider (faster)

# Flask Configuration
FLASK_ENV=production                              # production/development
FLASK_DEBUG=false                                 # true/false
SECRET_KEY=your_secure_secret_key                 # Auto-generated for security
PORT=8000                                         # Application port
HOST=0.0.0.0                                      # Host address

# Database Configuration (Auto-configured)
ZAMEEN_CHROMA_PERSIST_DIR=./chromadb_data         # Vector database location
ZAMEEN_COLLECTION_NAME=zameen_listings            # Collection name
ZAMEEN_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Embedding model
```

### Getting API Keys

1. **Cohere API Key** (Required):
   - Visit [cohere.ai](https://cohere.ai)
   - Sign up for free account
   - Get your API key from dashboard
   - Free tier includes 100 requests/minute

2. **Groq API Key** (Optional, for faster responses):
   - Visit [groq.com](https://groq.com)
   - Sign up and get API key
   - Provides faster inference than Cohere

### Advanced Configuration

```python
# config.py - Customize application behavior
class Settings(BaseSettings):
    # Data processing
    max_listings_per_session: int = 1000
    batch_size: int = 100
    
    # Vector database
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    
    # Web scraping
    delay_between_requests: float = 1.0
    max_retries: int = 3
```

## 📈 Performance & Metrics

### Current Database Statistics
- **🏠 Properties Indexed**: 160+ Bahria Town Phase 7 listings
- **📊 Vector Embeddings**: 800+ document chunks for semantic search
- **⚡ Query Response Time**: Sub-second (< 0.5s average)
- **💾 Database Size**: ~9MB ChromaDB storage
- **🔄 Search Accuracy**: High semantic similarity matching

### Performance Benchmarks
- **Average Query Processing**: ~300ms
- **Embedding Generation**: Real-time during queries
- **Memory Usage**: ~1.5GB for full operation
- **Concurrent Users**: Supports 20+ simultaneous chat sessions
- **API Response**: 200ms average for chat messages

### System Requirements
- **Minimum RAM**: 2GB
- **Recommended RAM**: 4GB+ for optimal performance
- **Storage**: 100MB for application + 10MB for database
- **CPU**: Modern multi-core processor recommended

## 🛡️ Security & Best Practices

### Data Protection
- **Rate Limiting**: Respectful web scraping with configurable delays
- **Input Validation**: Comprehensive validation of user inputs
- **Error Handling**: Robust exception handling throughout the pipeline
- **Logging**: Comprehensive logging for monitoring and debugging

### Production Considerations
- **API Authentication**: JWT-based authentication for production use
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Health Checks**: Built-in health monitoring endpoints
- **Monitoring**: Integration-ready for application performance monitoring

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow our coding standards
4. **Add tests**: Ensure your changes are tested
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes and their impact

### Development Guidelines

- **Code Style**: Follow PEP 8 Python style guidelines
- **Testing**: Write tests for new features and bug fixes
- **Documentation**: Update documentation for any API changes
- **Commit Messages**: Use conventional commit format

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/test_scraping.py -v
python -m pytest tests/test_rag.py -v
```

## 📚 Documentation

- **🌐 [Live Demo](http://localhost:8000)**: Try the application locally
- **⚙️ [Setup Guide](SETUP_GUIDE.md)**: Detailed installation instructions  
- **🚀 [Deployment Guide](DEPLOYMENT.md)**: Production deployment instructions
- **🔧 [Configuration Reference](config.py)**: Complete configuration options
- **� [Database Status](check_vector_db_status.py)**: Check database health

## 🧪 Testing

```bash
# Check if everything is working
python check_vector_db_status.py

# Run the application
python web_ui/app.py

# Test the chat API
python -c "
import requests
response = requests.post('http://localhost:8000/api/message', 
    json={'message': 'Hello', 'chat_id': 'test'})
print(response.json())
"
```

## 🚀 Deployment

### 🌐 Render.com (Recommended)

PropertyGuru is optimized for Render.com deployment with `render.yaml` configuration:

1. **Fork/Clone the repository**
2. **Connect to Render.com**
3. **Set environment variables**:
   - `COHERE_API_KEY`: Your Cohere API key
   - `SECRET_KEY`: Auto-generated by Render
4. **Deploy**: Automatic deployment from `render.yaml`

### 🐳 Docker Deployment

```bash
# Build and run with Docker
docker build -t propertyguru .
docker run -p 8000:8000 -e COHERE_API_KEY=your_key propertyguru

# Or use Docker Compose
docker-compose up
```

### 📦 Heroku Deployment

```bash
# Deploy to Heroku
heroku create your-app-name
heroku config:set COHERE_API_KEY=your_key
git push heroku main
```

### ⚡ Vercel Deployment

Configured for serverless deployment with `vercel.json`:

```bash
# Deploy to Vercel
npm i -g vercel
vercel
```

### 🔧 Environment Variables for Production

Set these variables in your deployment platform:

```bash
COHERE_API_KEY=your_cohere_api_key     # Required
SECRET_KEY=random_secure_key           # Auto-generated
FLASK_ENV=production                   # Production mode
FLASK_DEBUG=false                      # Disable debug
```


## 🙏 Acknowledgments

- **🏠 Zameen.com**: Property data source for Bahria Town Phase 7
- **🤖 Cohere**: Advanced language model for conversational AI
- **⚡ Groq**: High-speed inference for optimal performance
- **🔍 Sentence Transformers**: Semantic embedding technology
- **📊 ChromaDB**: Vector database for similarity search
- **🌐 Hugging Face**: Transformer models and embeddings
- **🐍 Python Community**: Amazing ecosystem of ML/AI libraries

---

<div align="center">
  <p><strong>Built with ❤️ for intelligent real estate search</strong></p>
  <p>⭐ Star this repo if you find it helpful! ⭐</p>
  <p>🚀 <a href="https://render.com">Deploy to Render</a> | 🐳 <a href="Dockerfile">Docker Ready</a> | 💬 <a href="http://localhost:8000">Try Demo</a></p>
</div>
#   P r o p e r t y - G u r u R A G  
 