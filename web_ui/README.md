# Real Estate Assistant Web UI

Flask-based chat interface inspired by ChatGPT design.

## Features
- Collapsible / mobile sidebar with conversation history
- Dark & light themes (default dark)
- Chat bubbles for user (right) and assistant (left)
- Sticky input bar with auto-growing textarea
- Typing indicator
- Settings modal (theme + font size)
- Scroll to bottom helper
- Simple in-memory conversation storage

## Run
Ensure dependencies installed (Flask already in requirements):

```
python web_ui/app.py
```

Open: http://localhost:8000

Set environment variable for Groq API key:
```
$Env:GROQ_API_KEY = "your_key_here"   # PowerShell
```

## Notes
- Conversations stored in-memory; restart clears them.
- For persistence, replace CONVERSATIONS with a database.
- Uses existing RAG + Groq logic via rag_infer().
