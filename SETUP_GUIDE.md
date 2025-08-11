# PropertyGuru Setup & Version Management Guide

This guide ensures that your PropertyGuru environment is set up correctly and free from version conflicts, particularly with ChromaDB.

## Quick Setup

```bash
# 1. Clone and navigate to the project
cd PropertyGuru

# 2. Run the automated setup script
python setup_environment.py

# 3. Add your API key to .env file
# Edit .env and add: COHERE_API_KEY=your_key_here

# 4. Build the vector database
python scripts/embed_and_store.py --input data/processed/zameen_phase7_chunks.jsonl

# 5. Start the application
python web_ui/app.py
```

## Version Management

### Critical Dependencies

The following packages have strict version requirements:

- **ChromaDB: 0.5.3** (MUST be this exact version)
- **sentence-transformers: ≥2.2.2** (compatible range)
- **Flask: 3.0.0**
- **python-docx: 1.1.2**

### ChromaDB Version Issues

**Problem**: ChromaDB version mismatches cause database corruption and API failures.

**Solution**: Always use ChromaDB 0.5.3 exactly.

```bash
# Force reinstall ChromaDB with correct version
pip uninstall chromadb -y
pip install chromadb==0.5.3 --no-cache-dir
```

### Database Compatibility

If you encounter ChromaDB errors after version changes:

1. **Check database status:**
   ```bash
   python check_vector_db_status.py
   ```

2. **If API fails but SQLite shows data:**
   ```bash
   # Delete old database
   rmdir /s chromadb_data  # Windows
   rm -rf chromadb_data    # Linux/Mac
   
   # Rebuild with current version
   python scripts/embed_and_store.py --input data/processed/zameen_phase7_chunks.jsonl
   ```

## Environment Verification

Use the verification script to check your setup:

```bash
python verify_dependencies.py
```

This will:
- ✅ Check all package versions
- ✅ Test ChromaDB functionality
- ✅ Verify database accessibility
- ⚠️ Report any version mismatches

## Troubleshooting Common Issues

### Issue 1: "object of type 'int' has no len()"
- **Cause**: ChromaDB version mismatch
- **Fix**: Delete `chromadb_data` folder and rebuild database

### Issue 2: "no such column: collections.topic"
- **Cause**: Database created with older ChromaDB version
- **Fix**: Rebuild database with current version

### Issue 3: Import errors
- **Cause**: Missing or incompatible package versions
- **Fix**: Run `python setup_environment.py`

### Issue 4: Web app crashes on startup
- **Cause**: Cohere API key not set
- **Fix**: Add `COHERE_API_KEY` to `.env` file

## Development Workflow

1. **Initial Setup**: Run `setup_environment.py`
2. **Regular Checks**: Use `verify_dependencies.py`
3. **Database Issues**: Use `check_vector_db_status.py`
4. **Clean Rebuild**: Delete `chromadb_data` and re-run embedding script

## Production Deployment

For production environments:

1. Use exact versions from `requirements.txt`
2. Always rebuild vector database in production environment
3. Verify setup with `verify_dependencies.py`
4. Monitor database health with `check_vector_db_status.py`

## File Overview

| Script | Purpose |
|--------|---------|
| `setup_environment.py` | Complete environment setup |
| `verify_dependencies.py` | Package version verification |
| `check_vector_db_status.py` | Database health monitoring |
| `requirements.txt` | Dependency specifications |

## Version History

- **v1.0**: Initial ChromaDB 0.4.24 setup
- **v1.1**: Upgraded to ChromaDB 0.5.3 for stability
- **v1.2**: Added comprehensive version management tools

---

**Remember**: When in doubt, delete the `chromadb_data` folder and rebuild. This ensures compatibility with your current ChromaDB version.
