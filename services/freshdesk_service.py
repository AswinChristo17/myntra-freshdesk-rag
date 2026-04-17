import os
import re
import html
import requests
from typing import List, Dict, Any
from utils.logger import log_info, log_error, log_warn

class FreshdeskService:
    """Service for Freshdesk API interactions"""
    
    def __init__(self):
        freshdesk_domain = os.getenv("FRESHDESK_DOMAIN")
        freshdesk_api_key = os.getenv("FRESHDESK_API_KEY")
        if not freshdesk_domain or not freshdesk_api_key:
            raise ValueError("FRESHDESK_DOMAIN or FRESHDESK_API_KEY is missing. Set them in .env or environment variables.")
        self.base_url = f"{freshdesk_domain}/api/v2"
        self.auth = (freshdesk_api_key, "X")
        self.headers = {"Content-Type": "application/json"}

    def _markdown_to_html(self, content: str) -> str:
        """Convert simple markdown notes into Freshdesk-friendly HTML."""
        if not content:
            return ""

        normalized_content = str(content).strip()

        # Handle collapsed one-line markdown by forcing section/list boundaries.
        section_markers = [
            "## Private Note",
            "### Ticket Summary",
            "### Steps to Resolve",
            "### Next Update",
        ]
        for marker in section_markers:
            normalized_content = normalized_content.replace(marker, f"\n{marker}\n")
        normalized_content = re.sub(r"\s+(\d+\.\s+)", r"\n\1", normalized_content)
        normalized_content = re.sub(r"\n{3,}", "\n\n", normalized_content).strip()

        lines = normalized_content.splitlines()
        html_lines: List[str] = []
        in_ordered_list = False

        def close_list_if_open():
            nonlocal in_ordered_list
            if in_ordered_list:
                html_lines.append("</ol>")
                in_ordered_list = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                close_list_if_open()
                continue

            if line.startswith("### "):
                close_list_if_open()
                html_lines.append(f"<p><strong>{html.escape(line[4:].strip())}</strong></p>")
                continue

            if line.startswith("## "):
                close_list_if_open()
                html_lines.append(f"<p><strong>{html.escape(line[3:].strip())}</strong></p>")
                continue

            ordered_item = re.match(r"^(\d+)\.\s+(.+)$", line)
            if ordered_item:
                if not in_ordered_list:
                    html_lines.append("<ol>")
                    in_ordered_list = True
                html_lines.append(f"<li>{html.escape(ordered_item.group(2).strip())}</li>")
                continue

            close_list_if_open()
            html_lines.append(f"<p>{html.escape(line)}</p>")

        close_list_if_open()
        return "\n".join(html_lines)
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Freshdesk API"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, auth=self.auth, headers=self.headers)
            elif method.upper() == "POST":
                response = requests.post(url, auth=self.auth, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log_error(f"Freshdesk API request failed for {endpoint}", e)
            raise
    
    def get_knowledge_base(self) -> List[Dict[str, Any]]:
        """Fetch knowledge base articles from Freshdesk"""
        try:
            log_info("Fetching knowledge base from Freshdesk")
            
            categories_response = self._make_request("GET", "/solutions/categories")
            categories = categories_response if isinstance(categories_response, list) else categories_response.get("categories", [])
            all_articles = []
            
            for category in categories:
                try:
                    folders_response = self._make_request("GET", f"/solutions/categories/{category['id']}/folders")
                    folders = folders_response if isinstance(folders_response, list) else folders_response.get("folders", [])
                    
                    for folder in folders:
                        try:
                            articles_response = self._make_request("GET", f"/solutions/folders/{folder['id']}/articles")
                            articles = articles_response if isinstance(articles_response, list) else articles_response.get("articles", [])
                            
                            for article in articles:
                                all_articles.append({
                                    "id": article.get("id"),
                                    "title": article.get("title"),
                                    "description": article.get("description"),
                                    "description_text": article.get("description_text"),
                                    "details": article.get("details"),
                                    "content": article.get("content"),
                                    "keywords": article.get("keywords"),
                                    "tags": article.get("tags"),
                                    "category": category.get("name"),
                                    "folder": folder.get("name"),
                                    "status": article.get("status"),
                                    "created_at": article.get("created_at"),
                                    "updated_at": article.get("updated_at")
                                })
                        except Exception as e:
                            log_warn(f"Error fetching articles from folder {folder['id']}", str(e))
                
                except Exception as e:
                    log_warn(f"Error fetching folders from category {category['id']}", str(e))
            
            log_info(f"Successfully fetched {len(all_articles)} knowledge base articles")
            return all_articles
        
        except Exception as e:
            log_error("Error fetching knowledge base", e)
            raise
    
    def get_ticket(self, ticket_id: int) -> Dict[str, Any]:
        """Get specific ticket details"""
        try:
            log_info(f"Fetching ticket details for ID: {ticket_id}")
            ticket = self._make_request("GET", f"/tickets/{ticket_id}")
            return ticket
        except Exception as e:
            log_error(f"Error fetching ticket {ticket_id}", e)
            raise
    
    def add_note(self, ticket_id: int, note_content: str, is_private: bool = True) -> Dict[str, Any]:
        """Add a note to a ticket. Set is_private=False for a public note."""
        try:
            note_type = "private" if is_private else "public"
            log_info(f"Adding {note_type} note to ticket {ticket_id}")
            
            note_data = {
                "body": self._markdown_to_html(note_content),
                "private": is_private
            }
            
            response = self._make_request("POST", f"/tickets/{ticket_id}/notes", note_data)
            log_info(f"{note_type.capitalize()} note added successfully to ticket {ticket_id}")
            return response
        
        except Exception as e:
            log_error(f"Error adding {note_type} note to ticket {ticket_id}", e)
            raise

    def add_private_note(self, ticket_id: int, note_content: str) -> Dict[str, Any]:
        """Backward-compatible helper to add a private note."""
        return self.add_note(ticket_id, note_content, is_private=True)

    def add_public_note(self, ticket_id: int, note_content: str) -> Dict[str, Any]:
        """Add a public note to a ticket."""
        return self.add_note(ticket_id, note_content, is_private=False)
    
    def search_knowledge_base(self, keyword: str) -> List[Dict[str, Any]]:
        """Search knowledge base articles by keyword"""
        try:
            log_info(f"Searching knowledge base for keyword: {keyword}")
            
            articles_response = self._make_request("GET", f"/solutions/articles/search?query={keyword}&per_page=10")
            articles = articles_response if isinstance(articles_response, list) else articles_response.get("articles", [])
            log_info(f"Found {len(articles)} matching articles")
            return articles
        
        except Exception as e:
            log_error("Error searching knowledge base", e)
            raise


# Create singleton instance
freshdesk_service = FreshdeskService()
