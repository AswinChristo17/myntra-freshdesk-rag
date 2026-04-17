# Freshdesk Groq Integration - FastAPI (Python)

Automatically generate and add intelligent private notes to Freshdesk tickets using Groq LLM and knowledge base articles.

## Features

- 🎯 Webhook listener for Freshdesk ticket creation events
- 🤖 Groq LLM-powered note generation (faster and cheaper than OpenAI)
- 📚 Knowledge base article retrieval and relevance filtering
- 🔒 Automatic private note addition to tickets
- 🌐 ngrok support for local testing
- 📝 Comprehensive logging
- ⚡ Built with FastAPI for high performance
- 🐍 Python-based for easy customization

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Freshdesk account with API access
- Groq API key
- ngrok (for local testing)

## Installation

1. **Navigate to the project directory**
```bash
cd d:\Kambaa\Myntra
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Configuration

Your `.env` file is already configured with your credentials:
- FQuick Start

### Step 1: Test Your Configuration
```bash
python test_connection.py
```

Expected output:
```
✓ Freshdesk connection successful. Found X KB articles
✓ Groq API connection successful
✓ All tests passed! Ready for production.
```

### Step 2: Start the Server
```bash
python main.py
```

Or with development auto-reload:
```bash
uvicorn m**Admin → Automations → Webhooks**
2. Click **Add webhook**
3. Fill in the details:
   - **Event:** Ticket Created
   - **URL:** `https://your-ngrok-url.ngrok.io/webhook/ticket-created` (replace with your ngrok URL)
   - **Method:** POST
4. Click **Save**

### Step 5: Test It Out

Create a test ticket in Freshdesk. Within seconds, you should see:
1. A private note automatically added to the ticket
2. Log messages in the server terminal showing the processing
```

2. **Start the application**
```bash
npm start
```

3. **In a new terminal, start ngrok**
```bash
ngrok http 3000
```

4. **Copy the ngrok URL** (e.g., `https://abc123.ngrok.io`)

5. **Update webhook URL in Freshdesk** with the ngrok URL

6. **Create a test ticket** in Freshdesk to verify the flow

## Project Structure

```
.
├── main.py                              # FastAPI application
├── services/
│   ├── freshdesk_service.py            # Freshdesk API integration
│   └── groq_service.py                 # Groq LLM integration
├── utils/
│   └── logger.py                       # Logging utility
├── .env                                 # Environment variables (pre-filled)
├── requirements.txt                     # Python dependencies
├── test_connection.py                   # Connection testing script
├── README.md                            # This file
└── DEVELOPMENT.md                       # Development guide
```

## API Endpoints

### Health Check
```
GET /health
```
Returns server status.

### Webhook - Ticket Created
```
POST /webhook/ticket-created
```
Listens for Freshdesk ticket creation events and processes them.

**Payload:**
```json
{
  "ticket": {
    1. Health Check
```
GET /health

Response:
{
  "status": "OK",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 2. API Info
```
GET /

Response:
{
  "name": "Freshdesk Groq Integration API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "webhook": "/webhook/ticket-created",
    "docs": "/docs"
  }
}
```

### 3. Webhook - Ticket Created
```
POST /webhook/ticket-created

Request Body:
{
  "ticket": {
    "id": 12345,
   Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/

# TTroubleshooting

### Issue: ModuleNotFoundError
**Solution:** Activate your virtual environment first:
```bash
# Windows
venv\Scripts\activate
Performance Tips

- Groq is faster and cheaper than OpenAI - great for high-volume support
- Requests are processed asynchronously
- KB articles are fetched on each ticket (can be cached for optimization)
- Average processing time: 2-5 seconds per ticket

## Security Considerations

- ✓ API keys stored in `.env` (never committed to git)
- ✓ HTTPS enforced in production (ngrok provides this locally)
- Consider adding request rate limiting for production
- Consider adding webhook signature verification
- Never expose `.env` file or API keys
**Solution:**
- Verify `GROQ_API_KEY` in `.env` is valid
- Check your Groq account has available quota

### Issue: Webhook not triggering
**Solution:**
- Verify ngrok is running: `ngrok http 8000`
- Verify webhook URL in Freshdesk matches ngrok URL
- Check ngrok URL hasn't expired (restart if needed)
- Check server logs for incoming requests
      "subject": "Test Issue",
      "description": "This is a test issue description",
      "priority": 1,
      "status": 2,
      "tags": ["test"]
    }
  }'
```

## Logging

All operations are logged with timestamps:
- **INFO**: General operations (successful actions)
- **WARNING**: Non-critical issues
- **ERROR**: Failures and exceptions

Example log output:
```
[INFO] 2024-01-15 10:30:00 - Server running on port 8000
[INFO] 2024-01-15 10:30:15 - Received ticket creation webhook
[INFO] 2024-01-15 10:30:15 - Processing ticket: 12345 - Test Issue
[INFO] 2024-01-15 10:30:16 - Retrieved 15 knowledge base articles
[INFO] 2024-01-15 10:30:17 - Generating Groq response for ticket 12345
[INFO] 2024-01-15 10:30:18 - Successfully generated note for ticket 12345
[INFO] 2024-01-15 10:30:19 - Added private note to ticket 12345
```
StatusBackground task queue for high volume
- [ ] KB caching with TTL
- [ ] Webhook signature verification
- [ ] Database to track processed tickets
- [ ] Admin dashboard
- [ ] Custom prompt templates
- [ ] Multi-language support
- [ ] Rate limiting

## Resources

- Freshdesk API: https://developers.freshdesk.com/api/
- Groq API: https://console.groq.com/docs
- FastAPI: https://fastapi.tiangolo.com/
- ngrok: https://ngrok.com/docs

## Support

For issues:
1. Check logs for error messages
2. Run `python test_connection.py` to diagnose
3. Verify all credentials in `.env`
4. Check API service status page more capable
- `gemma-7b-it` - Lightweight option

Change the model in `.env` by updating `GROQ_MODEL`:
```env
GROQ_MODEL=llama2-70b-4096
```
### 4. Interactive API Documentation
```
http://localhost:8000/docs          # Swagger UI
http://localhost:8000/redoc         # ReDoc
## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `FRESHDESK_DOMAIN` | Your Freshdesk domain URL | Yes |
| `FRESHDESK_API_KEY` | Freshdesk API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `OPENAI_MODEL` | OpenAI model to use | Yes |
| `PORT` | Server port (default: 3000) | No |
| `WEBHOOK_SECRET` | Optional webhook secret | No |
| `NGROK_URL` | ngrok URL for webhooks | No |

## Logging

The application includes comprehensive logging with different levels:
- **INFO**: General information and successful operations
- **WARN**: Warnings and non-critical issues
- **ERROR**: Errors and exceptions
- **DEBUG**: Debug information (only when `DEBUG` env var is set)

## Error Handling

- Invalid webhook payloads return 400 status
- API errors are caught and logged
- Server errors return 500 status with error details
- All errors are logged for debugging

## Troubleshooting

### Webhook Not Triggering
- Verify webhook URL is correct in Freshdesk
- Check ngrok is running and URL is active
- Monitor server logs for incoming requests

### API Errors
- Verify API keys are correct
- Check Freshdesk domain URL format
- Ensure API key has required permissions

### LLM Not Generating Notes
- Check OpenAI API key is valid
- Monitor OpenAI API usage/quota
- Check rate limits and retry logic

## Future Enhancements

- [ ] Queue system for handling multiple tickets
- [ ] Configurable LLM models and providers
- [ ] Webhook signature verification
- [ ] Database for storing processed tickets
- [ ] Admin dashboard for monitoring
- [ ] Custom prompt templates
- [ ] Multi-language support

## Support

For issues or questions, refer to:
- Freshdesk API Docs: https://developers.freshdesk.com/api/
- OpenAI API Docs: https://platform.openai.com/docs/
- ngrok Docs: https://ngrok.com/docs

## License

ISC
