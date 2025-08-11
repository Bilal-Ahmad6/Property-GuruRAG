import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify
import logging
from logging.handlers import RotatingFileHandler

# Load .env early
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
except Exception:
    pass

# Add project root to path so we can import scripts.query_rag
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.query_rag import rag_infer, load_processed_map  # type: ignore
from config import settings  # type: ignore

app = Flask(__name__)

# Configure secret key for session security
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Configure logging for production
if not app.debug:
    # Ensure logs directory exists
    logs_dir = PROJECT_ROOT / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Set up rotating file handler
    file_handler = RotatingFileHandler(
        logs_dir / 'app.log', 
        maxBytes=10240000, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('PropertyGuru application startup')

# In-memory + file-backed conversation store
CONVERSATIONS: Dict[str, List[Dict[str, Any]]] = {}
PERSIST_PATH = PROJECT_ROOT / 'web_ui' / 'conversations.json'


def load_conversations() -> None:
    if PERSIST_PATH.exists():
        try:
            data = json.loads(PERSIST_PATH.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                # Basic validation
                for k, v in data.items():
                    if isinstance(v, list):
                        CONVERSATIONS[k] = v
        except Exception:
            pass


def save_conversations() -> None:
    try:
        PERSIST_PATH.write_text(json.dumps(CONVERSATIONS, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


load_conversations()


def get_api_key() -> str:
    return os.getenv('COHERE_API_KEY') or os.getenv('ZAMEEN_COHERE_API_KEY') or settings.cohere_api_key


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Unhandled exception: {e}')
    return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Basic health checks
        api_key = get_api_key()
        chroma_exists = (PROJECT_ROOT / 'chromadb_data').exists()
        
        status = {
            'status': 'healthy',
            'api_key_configured': bool(api_key),
            'vector_db_exists': chroma_exists,
            'conversations_loaded': len(CONVERSATIONS),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(status), 200
    except Exception as e:
        # Ensure health check always returns something
        error_status = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_status), 500
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    chat_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    CONVERSATIONS[chat_id] = []
    save_conversations()
    return jsonify({"chat_id": chat_id})


@app.route('/api/list_chats')
def list_chats():
    # Return last 50 chats with first user message as title or default
    items = []
    for cid, msgs in sorted(CONVERSATIONS.items(), reverse=True):
        title = 'New Conversation'
        for m in msgs:
            if m.get('role') == 'user' and m.get('content'):
                raw = m['content'].strip().split('\n')[0]
                title = (raw[:40] + '...') if len(raw) > 40 else raw
                break
        items.append({'chat_id': cid, 'title': title})
    return jsonify({'chats': items[:50]})


@app.route('/api/get_chat/<chat_id>')
def get_chat(chat_id: str):
    return jsonify({'messages': CONVERSATIONS.get(chat_id, [])})


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    CONVERSATIONS.clear()
    save_conversations()
    return jsonify({'status': 'cleared'})


@app.route('/api/message', methods=['POST'])
def message():
    data = request.get_json(force=True)
    chat_id = data.get('chat_id')
    user_message = (data.get('message') or '').strip()
    store_history = bool(data.get('store_history', True))
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    if not chat_id:
        # auto create
        chat_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        CONVERSATIONS[chat_id] = []

    history = CONVERSATIONS.setdefault(chat_id, [])
    user_entry = {'role': 'user', 'content': user_message, 'ts': datetime.utcnow().isoformat()}
    history.append(user_entry)

    api_key = get_api_key()
    meta: Dict[str, Any] = {}
    if not api_key:
        assistant_text = 'Cohere API key missing. Please set COHERE_API_KEY in your .env.'
        mode = 'error'
    else:
        # Build conversation_history structure for rag_infer
        # Limit conversation history to last 5 exchanges to avoid context overflow
        recent_history = history[-10:] if len(history) > 10 else history
        
        convo_for_rag = []
        for m in recent_history:
            if m['role'] == 'user':
                convo_for_rag.append({'user': m['content'], 'assistant': ''})
            elif m['role'] == 'assistant' and convo_for_rag:
                convo_for_rag[-1]['assistant'] = m['content']
        try:
            # Use Cohere as the LLM engine
            result = rag_infer(
                query=user_message, 
                cohere_api_key=api_key,
                llm_engine="cohere",
                cohere_model="command-r-plus",
                conversation_history=convo_for_rag
            )
            assistant_text = result.get('answer', 'No response generated.')
            mode = result.get('mode')
            # Attach extra metadata for UI parity with CLI tool
            meta.update({k: result.get(k) for k in ['mode', 'requested_number', 'timings', 'retrieved'] if k in result})

            # If retrieval succeeded, enrich with property details similar to CLI output
            if mode == 'retrieval' and result.get('retrieved'):
                try:
                    processed_path = PROJECT_ROOT / 'data' / 'processed' / 'zameen_phase7_processed.json'
                    processed_map = load_processed_map(processed_path)
                    listing_details = []
                    retrieved_listings = result.get('retrieved', [])
                    
                    # Strictly respect user's requested number
                    requested_count = result.get('requested_number')
                    if requested_count:
                        retrieved_listings = retrieved_listings[:requested_count]
                    
                    for rec in retrieved_listings:
                        lid = str(rec.get('listing_id'))
                        pdata = processed_map.get(lid, {})
                        if not pdata:
                            continue
                        
                        # Extract amenities and clean them up
                        amenities = pdata.get('amenities', [])
                        clean_amenities = []
                        for amenity in amenities:
                            if isinstance(amenity, str) and len(amenity.strip()) > 0:
                                # Clean up amenity text
                                clean_amenity = amenity.strip()
                                if len(clean_amenity) > 3 and clean_amenity not in clean_amenities:
                                    clean_amenities.append(clean_amenity)
                        
                        listing_details.append({
                            'listing_id': lid,
                            'title': pdata.get('title'),
                            'price': pdata.get('price_raw') or pdata.get('price_numeric'),
                            'bedrooms': pdata.get('bedrooms'),
                            'bathrooms': pdata.get('bathrooms'),
                            'area_size': pdata.get('area_size'),
                            'area_unit': pdata.get('area_unit'),
                            'location': pdata.get('location'),
                            'url': pdata.get('url') or (pdata.get('raw', {}) or {}).get('url'),
                            'amenities': clean_amenities[:8],  # Limit to top 8 amenities to avoid UI clutter
                        })
                    if listing_details:
                        meta['listings'] = listing_details
                except Exception as enrich_err:
                    meta['listing_enrich_error'] = str(enrich_err)
        except Exception as e:
            # Special guidance for common errors
            err_str = str(e)
            if 'no such column: collections.topic' in err_str.lower():
                assistant_text = (
                    "Error accessing vector DB (schema mismatch). Delete the 'chromadb_data' folder and re-run embeddings (scripts/embed_and_store.py)."
                )
            elif 'request too large' in err_str.lower() or 'context length' in err_str.lower():
                assistant_text = (
                    "The request was too large for the model. Try asking a shorter, more specific question. "
                    "For example: 'Find 3 bedroom houses' instead of very long queries."
                )
            elif 'cohere api error' in err_str.lower():
                assistant_text = (
                    "There was an issue with the Cohere API. Please try again in a moment."
                )
            elif 'groq api error: 413' in err_str.lower():
                assistant_text = (
                    "Request too large for the current model. Please try a shorter, more focused question."
                )
            else:
                assistant_text = f'Error: {e}'
            mode = 'error'
            meta['error'] = err_str
    assistant_entry = {'role': 'assistant', 'content': assistant_text, 'ts': datetime.utcnow().isoformat(), 'meta': meta}
    history.append(assistant_entry)

    if store_history:
        save_conversations()
    else:
        # If user disabled history, remove this chat entirely when empty
        pass

    return jsonify({'chat_id': chat_id, 'assistant_message': assistant_entry, 'messages': history, 'title': history[0]['content'] if history and history[0]['role']=='user' else 'New Conversation'})


if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Auto-run with API key loaded from .env if available
    print("🚀 Starting PropertyGuru Web UI...")
    print(f"📍 Access the application at: http://{host}:{port}")
    print("⚠️  Press CTRL+C to stop the server")
    
    # Log startup info
    app.logger.info(f'Starting PropertyGuru on {host}:{port} (debug={debug})')
    
    app.run(host=host, port=port, debug=debug)

