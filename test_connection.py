import os
from dotenv import load_dotenv
from utils.logger import log_info, log_error

load_dotenv()

from services.freshdesk_service import freshdesk_service
from services.groq_service import groq_service

async def test_freshdesk_connection():
    """Test Freshdesk API connectivity"""
    try:
        log_info("Testing Freshdesk connection...")
        kb = freshdesk_service.get_knowledge_base()
        log_info(f"✓ Freshdesk connection successful. Found {len(kb)} KB articles")
        return True
    except Exception as e:
        log_error(f"✗ Freshdesk connection failed", e)
        return False

async def test_groq_connection():
    """Test Groq API connectivity"""
    try:
        log_info("Testing Groq API connection...")
        
        test_context = {
            "ticketId": 0,
            "subject": "Test Subject",
            "description": "Test description for connectivity check",
            "knowledgeBase": [],
            "priority": 1,
            "tags": []
        }
        
        response = groq_service.generate_ticket_note(test_context)
        log_info("✓ Groq API connection successful")
        log_info(f"Generated response: {response[:100]}...")
        return True
    except Exception as e:
        log_error(f"✗ Groq API connection failed", e)
        return False

async def run_all_tests():
    """Run all connection tests"""
    log_info("\n" + "="*50)
    log_info("Running Connection Tests")
    log_info("="*50 + "\n")
    
    freshdesk_ok = await test_freshdesk_connection()
    log_info("")
    
    groq_ok = await test_groq_connection()
    
    log_info("\n" + "="*50)
    log_info("Test Summary")
    log_info("="*50)
    log_info(f"Freshdesk: {'✓ OK' if freshdesk_ok else '✗ FAILED'}")
    log_info(f"Groq: {'✓ OK' if groq_ok else '✗ FAILED'}")
    log_info("="*50 + "\n")
    
    if freshdesk_ok and groq_ok:
        log_info("✓ All tests passed! Ready for production.")
    else:
        log_info("⚠ Some tests failed. Check your .env configuration.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_tests())
