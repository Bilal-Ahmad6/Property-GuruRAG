# 🚀 PropertyGuru Deployment Guide

This guide covers various deployment options for the PropertyGuru application.

## 📋 Pre-Deployment Checklist

### Required Environment Variables
- `COHERE_API_KEY` or `ZAMEEN_COHERE_API_KEY` - Your Cohere API key
- `SECRET_KEY` - Flask secret key for session security
- `FLASK_ENV=production` - Set to production mode
- `FLASK_DEBUG=false` - Disable debug mode

### Required Data
1. **Vector Database**: Ensure `chromadb_data/` exists with embeddings
2. **Processed Data**: Ensure `data/processed/zameen_phase7_processed.json` exists
3. **API Keys**: Configure valid API keys in environment

## 🐳 Docker Deployment (Recommended)

### 1. Build and Run with Docker Compose
```bash
# Copy environment file
cp .env.example .env
# Edit .env with your actual values

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 2. Manual Docker Build
```bash
# Build image
docker build -t propertyguru .

# Run container
docker run -d \
  --name propertyguru \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chromadb_data:/app/chromadb_data \
  propertyguru
```

## ☁️ Cloud Platform Deployments

### Heroku
1. Install Heroku CLI
2. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```
3. Set environment variables:
   ```bash
   heroku config:set COHERE_API_KEY=your_key_here
   heroku config:set SECRET_KEY=your_secret_key
   heroku config:set FLASK_ENV=production
   ```
4. Deploy:
   ```bash
   git push heroku main
   ```

### Render.com
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the provided `render.yaml` configuration
4. Set environment variables in the Render dashboard

### Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Deploy: `vercel --prod`
3. Set environment variables in Vercel dashboard

### DigitalOcean App Platform
1. Create a new app in DigitalOcean
2. Connect your GitHub repository
3. Configure environment variables
4. Deploy using the platform's interface

## 🖥️ VPS/Server Deployment

### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.8+
- Nginx (for reverse proxy)
- Supervisor (for process management)

### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-venv nginx supervisor -y

# Create application user
sudo useradd -m -s /bin/bash propertyguru
sudo su - propertyguru
```

### 2. Application Setup
```bash
# Clone repository
git clone https://github.com/your-username/PropertyGuru.git
cd PropertyGuru

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your values
```

### 3. Nginx Configuration
Create `/etc/nginx/sites-available/propertyguru`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /home/propertyguru/PropertyGuru/web_ui/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/propertyguru /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Supervisor Configuration
Create `/etc/supervisor/conf.d/propertyguru.conf`:
```ini
[program:propertyguru]
command=/home/propertyguru/PropertyGuru/venv/bin/python web_ui/app.py
directory=/home/propertyguru/PropertyGuru
user=propertyguru
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/propertyguru.log
environment=PATH="/home/propertyguru/PropertyGuru/venv/bin"
```

Start the service:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start propertyguru
```

## 🔒 SSL/HTTPS Setup

### Using Certbot (Let's Encrypt)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already set up by certbot)
sudo crontab -l | grep certbot
```

## 📊 Monitoring and Maintenance

### Health Check Endpoint
The application includes a health check at `/health`:
```bash
curl http://your-domain.com/health
```

### Log Monitoring
- Application logs: `/app/logs/app.log` (Docker) or `/var/log/propertyguru.log` (VPS)
- Nginx logs: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`

### Backup Strategy
1. **Database**: Backup `chromadb_data/` directory
2. **Conversations**: Backup `web_ui/conversations.json`
3. **Processed Data**: Backup `data/processed/` directory

### Updates
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart application
sudo supervisorctl restart propertyguru  # VPS
# OR
docker-compose restart  # Docker
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify environment variables are set correctly
   - Check API key validity and quotas

2. **Vector Database Issues**
   - **"'dict' object has no attribute 'dimensionality'"**: ChromaDB schema compatibility issue
     - **Quick Fix**: Run `python chromadb_health.py repair`
     - **Manual Fix**: Delete `chromadb_data/` and run `python scripts/embed_and_store.py --input data/processed/zameen_phase7_chunks.jsonl`
   - Ensure `chromadb_data/` directory exists
   - Re-run embedding process if needed: `python scripts/embed_and_store.py`

3. **Memory Issues**
   - Monitor RAM usage, especially during embedding operations
   - Consider increasing server memory or using smaller models

4. **Port Conflicts**
   - Check if port 8000 is available: `sudo netstat -tlnp | grep :8000`
   - Change PORT environment variable if needed

5. **File Permissions**
   - Ensure application user has read/write access to data directories
   - Fix permissions: `sudo chown -R propertyguru:propertyguru /home/propertyguru/PropertyGuru`

### Performance Optimization

1. **Caching**: Implement Redis for conversation caching
2. **Database**: Use PostgreSQL for conversation storage in production
3. **CDN**: Use a CDN for static assets
4. **Load Balancing**: Use multiple application instances behind a load balancer

## 📞 Support

For deployment issues:
1. Check application logs
2. Verify environment configuration
3. Test health endpoint
4. Review this deployment guide
5. Check GitHub issues for similar problems
