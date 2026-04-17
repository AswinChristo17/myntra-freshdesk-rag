# Development Guide

## Local Development Setup

### Prerequisites
- Python 3.9+
- pip
- Virtual environment (recommended)
- VS Code or any Python IDE

### Initial Setup

1. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify .env is configured**
```bash
cat .env  # or type .env on Windows
```

Should show:
```env
FRESHDESK_DOMAIN=https://umaya1776.freshdesk.com
FRESHDESK_API_KEY=QHz48OlFv1v5Gbg_58ex
GROQ_API_KEY=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
```

### Running the Application

#### Development Mode (with auto-reload)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
python main.py
```

### Testing

#### Test Connection
```bash
python test_connection.py
```

#### Test Individual Services

**Test Freshdesk Service:**
```python
from services.freshdesk_service import freshdesk_service

# Get knowledge base
kb = freshdesk_service.get_knowledge_base()
print(f"Found {len(kb)} articles")

# Get specific ticket
ticket = freshdesk_service.get_ticket(123)
print(ticket)

# Add a note
note = freshdesk_service.add_private_note(123, "This is a test note")
```

**Test Groq Service:**
```python
from services.groq_service import groq_service

context = {
    "ticketId": 123,
    "subject": "Test",
    "description": "Test description",
    "knowledgeBase": [],
    "priority": 1,
    "tags": []
}

response = groq_service.generate_ticket_note(context)
print(response)
```

### API Endpoints During Development

Once the server is running at `http://localhost:8000`:

1. **Interactive Docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

2. **Test Health:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Manual Webhook Testing:**
   ```bash
   curl -X POST http://localhost:8000/webhook/ticket-created \
     -H "Content-Type: application/json" \
     -d @examples/webhook-payload.json
   ```

### Debug Logging

Enable verbose logging:
```bash
# Set DEBUG environment variable
set DEBUG=true  # Windows
export DEBUG=true  # macOS/Linux

python main.py
```

Modify logger in `utils/logger.py` if needed:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
```

## Code Structure

### main.py
- FastAPI application setup
- Route handlers
- Error handling
- Startup/shutdown events

### services/freshdesk_service.py
- `FreshdeskService` class for Freshdesk API
- Methods:
  - `get_knowledge_base()` - Fetch all KB articles
  - `get_ticket(ticket_id)` - Get ticket details
  - `add_private_note(ticket_id, content)` - Add private note
  - `search_knowledge_base(keyword)` - Search KB

### services/groq_service.py
- `GroqService` class for Groq API
- Methods:
  - `generate_ticket_note(context)` - Generate note using LLM
  - `filter_relevant_articles(description, kb)` - Filter relevant articles
  - `build_prompt(...)` - Build LLM prompt
  - `generate_ticket_summary(tickets)` - Generate summary

### utils/logger.py
- Logging utility functions
- Functions:
  - `log_info(message, data)`
  - `log_warn(message, data)`
  - `log_error(message, error)`
  - `log_debug(message, data)`

## Adding New Features

### Adding a New Endpoint

In `main.py`:
```python
from fastapi import HTTPException
from pydantic import BaseModel

class YourModel(BaseModel):
    field: str

@app.post("/api/your-endpoint")
async def your_endpoint(data: YourModel):
    """Your endpoint description"""
    try:
        log_info("Processing your endpoint")
        # Your logic here
        result = "success"
        return {"status": result}
    except Exception as e:
        log_error("Error in your endpoint", e)
        raise HTTPException(status_code=500, detail=str(e))
```

### Adding a New Service

Create `services/your_service.py`:
```python
from utils.logger import log_info, log_error

class YourService:
    def __init__(self):
        pass
    
    async def do_something(self):
        try:
            log_info("Doing something")
            # Your logic
            return result
        except Exception as e:
            log_error("Error in do_something", e)
            raise

# Create singleton instance
your_service = YourService()
```

Import in `main.py`:
```python
from services.your_service import your_service
```

## Virtual Environment Management

### Create
```bash
python -m venv venv
```

### Activate
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Deactivate
```bash
deactivate
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Freeze Current Dependencies
```bash
pip freeze > requirements.txt
```

## Common Development Tasks

### Update a Package
```bash
pip install --upgrade package-name
```

### Check Installed Packages
```bash
pip list
```

### Clear pip Cache
```bash
pip cache purge
```

### Run with Specific Python Version
```bash
python3.11 main.py
```

## Performance Optimization

### Caching KB Articles
```python
from functools import lru_cache
import time

class FreshdeskService:
    def __init__(self):
        self.kb_cache = None
        self.kb_cache_time = 0
    
    def get_knowledge_base(self):
        current_time = time.time()
        # Cache for 1 hour
        if self.kb_cache and (current_time - self.kb_cache_time) < 3600:
            return self.kb_cache
        
        # Fetch fresh data
        kb = self._fetch_kb()
        self.kb_cache = kb
        self.kb_cache_time = current_time
        return kb
```

### Async Processing
```python
import asyncio

async def process_multiple_tickets(tickets):
    tasks = [process_ticket(t) for t in tickets]
    results = await asyncio.gather(*tasks)
    return results
```

## Testing Best Practices

1. **Test with real data** - Use actual Freshdesk tickets
2. **Test error cases** - Try invalid inputs, missing fields
3. **Test API limits** - Check rate limits and quotas
4. **Monitor logs** - Always check logs for issues
5. **Verify notes** - Confirm private notes appear in Freshdesk

## Troubleshooting Tips

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Port already in use
```bash
# Use different port
uvicorn main:app --port 8001
```

### Issue: Slow responses
- Check network connection
- Monitor Groq API status
- Check Freshdesk API rate limits

### Issue: Webhook not working
- Verify ngrok is running
- Check webhook URL in Freshdesk
- Monitor server logs
- Check network connectivity

## Debugging Workflow

1. **Check logs** - Look for error messages
2. **Test connection** - Run `python test_connection.py`
3. **Verify credentials** - Check `.env` file
4. **Check APIs** - Verify API services are up
5. **Review code** - Check recent changes
6. **Add debug statements** - Use `log_debug()` in code

## Production Deployment Checklist

- [ ] All tests passing
- [ ] `.env` configured with production credentials
- [ ] HTTPS enabled (use real domain, not ngrok)
- [ ] Webhook URL updated in Freshdesk
- [ ] Logging configured for production
- [ ] Error monitoring setup
- [ ] Rate limiting enabled
- [ ] Database backups configured (if using DB)
- [ ] Monitoring and alerts setup

## Resources

- FastAPI Docs: https://fastapi.tiangolo.com/
- Python Docs: https://docs.python.org/3/
- Groq API: https://console.groq.com/docs
- Freshdesk API: https://developers.freshdesk.com/api/
- Pydantic: https://docs.pydantic.dev/
- Uvicorn: https://www.uvicorn.org/
