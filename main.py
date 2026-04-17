import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables before service imports
load_dotenv()

from services.freshdesk_service import freshdesk_service
from services.groq_service import groq_service
from utils.logger import log_info, log_error, log_warn

# Initialize FastAPI app
app = FastAPI(
    title="Freshdesk Groq Integration",
    description="Freshdesk webhook integration with Groq LLM for automatic ticket note generation",
    version="1.0.0"
)

# Pydantic models
class TicketData(BaseModel):
    id: int
    subject: str
    description: str
    priority: int
    status: int
    tags: List[str] = []

class WebhookPayload(BaseModel):
    ticket: TicketData

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    ticketId: int


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 400 with concise details instead of framework-level 422."""
    log_warn(f"Request validation failed on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "Invalid webhook payload format",
            "errors": exc.errors(),
        },
    )


def _to_int(value: Any, default: int = 0) -> int:
    """Safely cast a value to int for webhook fields that may arrive as strings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_tags(value: Any) -> List[str]:
    """Normalize tags from list/string/null payload variants."""
    if isinstance(value, list):
        return [str(tag) for tag in value if str(tag).strip()]
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    return []


def _pick_first(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Pick the first available key from a dict."""
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return default


def _looks_like_ticket(obj: Dict[str, Any]) -> bool:
    """Heuristic to detect ticket-like dicts in nested webhook payloads."""
    if not isinstance(obj, dict):
        return False
    has_id = any(k in obj for k in ["id", "ticket_id", "display_id"])
    has_subject = any(k in obj for k in ["subject", "ticket_subject", "title"])
    has_desc = any(k in obj for k in ["description", "description_text", "ticket_description", "body"])
    return has_id and has_subject and has_desc


def _find_ticket_object(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Find ticket object across known and nested payload shapes."""
    if not isinstance(payload, dict):
        return {}

    if _looks_like_ticket(payload):
        return payload

    # Common wrappers used by webhook providers/automations.
    common_paths = ["ticket", "data", "event", "payload", "object"]
    for key in common_paths:
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            if _looks_like_ticket(candidate):
                return candidate
            nested_ticket = candidate.get("ticket")
            if isinstance(nested_ticket, dict) and _looks_like_ticket(nested_ticket):
                return nested_ticket

    # Recursive search fallback for unknown nesting.
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_ticket_object(value)
            if found:
                return found

    return {}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    log_info("Health check requested")
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/webhook/ticket-created", response_model=ProcessingResponse)
async def webhook_ticket_created(request: Request):
    """
    Webhook endpoint for Freshdesk ticket creation
    
    This endpoint receives ticket creation events from Freshdesk and:
    1. Fetches relevant knowledge base articles
    2. Sends ticket details to Groq LLM
    3. Adds the LLM response as a private note
    """
    try:
        try:
            payload = await request.json()
        except Exception:
            form_data = await request.form()
            raw_payload = form_data.get("payload") or form_data.get("ticket") or "{}"
            try:
                payload = json.loads(raw_payload) if isinstance(raw_payload, str) else dict(form_data)
            except Exception:
                payload = dict(form_data)

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid payload: expected object JSON")

        ticket_data = _find_ticket_object(payload)
        if not ticket_data:
            ticket_data = payload

        ticket_id = _to_int(_pick_first(ticket_data, ["id", "ticket_id", "display_id"]))
        subject = str(_pick_first(ticket_data, ["subject", "ticket_subject", "title"], "")).strip()
        description = str(_pick_first(ticket_data, ["description", "description_text", "ticket_description", "body"], "")).strip()
        priority = _to_int(_pick_first(ticket_data, ["priority", "ticket_priority"], 1), default=1)
        status = _to_int(_pick_first(ticket_data, ["status", "ticket_status"], 2), default=2)
        tags = _to_tags(_pick_first(ticket_data, ["tags", "ticket_tags"], []))
        
        log_info(f"Received ticket creation webhook")
        log_info(f"Processing ticket: {ticket_id} - {subject}")
        
        # Validate payload
        if not ticket_id or not subject or not description:
            log_warn("Invalid webhook payload: missing required ticket data")
            raise HTTPException(status_code=400, detail="Invalid payload: missing required fields")
        
        # Step 1: Fetch knowledge base from Freshdesk
        knowledge_base = freshdesk_service.get_knowledge_base()
        log_info(f"Retrieved {len(knowledge_base)} knowledge base articles")
        
        # Step 2: Prepare context and send to Groq LLM
        llm_context = {
            "ticketId": ticket_id,
            "subject": subject,
            "description": description,
            "knowledgeBase": knowledge_base,
            "priority": priority,
            "status": status,
            "tags": tags
        }
        
        llm_response = groq_service.generate_ticket_note(llm_context)
        log_info(f"Groq generated response for ticket {ticket_id}")
        
        # Step 3: Add response as private note to the ticket
        freshdesk_service.add_private_note(ticket_id, llm_response)
        log_info(f"Added private note to ticket {ticket_id}")
        
        return {
            "success": True,
            "message": "Ticket processed successfully",
            "ticketId": ticket_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error processing webhook", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/webhook/ticket-created/", response_model=ProcessingResponse)
async def webhook_ticket_created_trailing_slash(request: Request):
    """Support webhook calls that include a trailing slash."""
    return await webhook_ticket_created(request)

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Freshdesk Groq Integration API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook/ticket-created",
            "docs": "/docs"
        },
        "freshdesk_domain": os.getenv("FRESHDESK_DOMAIN"),
        "groq_model": os.getenv("GROQ_MODEL")
    }

@app.on_event("startup")
async def startup_event():
    """Event handler for application startup"""
    log_info("="*50)
    log_info("Freshdesk Groq Integration Server Starting")
    log_info("="*50)
    log_info(f"Freshdesk Domain: {os.getenv('FRESHDESK_DOMAIN')}")
    log_info(f"Groq Model: {os.getenv('GROQ_MODEL')}")
    ngrok_url = os.getenv("NGROK_URL")
    if ngrok_url:
        log_info(f"Webhook URL: {ngrok_url}/webhook/ticket-created")
    log_info("="*50)

@app.on_event("shutdown")
async def shutdown_event():
    """Event handler for application shutdown"""
    log_info("="*50)
    log_info("Freshdesk Groq Integration Server Shutting Down")
    log_info("="*50)

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    log_info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
