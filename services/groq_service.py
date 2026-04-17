import os
import re
import math
import json
from collections import Counter
from typing import Dict, Any, List
from groq import Groq
from services.rag_service import rag_service
from utils.logger import log_info, log_error

class GroqService:
    """Service for Groq LLM API interactions"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it in .env or environment variables.")
        self.client = Groq(api_key=api_key)
    
    def generate_ticket_note(self, context: Dict[str, Any]) -> str:
        """Generate an internal private note for a ticket using Groq LLM"""
        try:
            ticket_id = context.get("ticketId")
            subject = self._clean_text(context.get("subject", ""), max_len=300)
            description = self._clean_text(context.get("description", ""), max_len=900)
            knowledge_base = context.get("knowledgeBase", [])
            priority = context.get("priority")
            tags = context.get("tags", [])
            
            log_info(f"Generating Groq response for ticket {ticket_id} using model {self.model}")
            
            # Retrieve most relevant KB content via lightweight vector search.
            query_text = f"{subject} {description}".strip()
            relevant_kb = rag_service.filter_relevant_articles(query_text, knowledge_base)
            relevant_kb = self._ensure_contact_kb_coverage(query_text, knowledge_base, relevant_kb)
            
            # Build the prompt
            prompt = self.build_prompt(subject, description, relevant_kb, priority, tags)
            
            # Call Groq API
            message = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a support operations assistant. Return concise internal notes only in valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=1000,
            )
            
            raw_output = message.choices[0].message.content or ""
            note_data = self._parse_note_json(raw_output)
            key_details = self._extract_key_details(f"{subject} {description}")
            note_data = self._enrich_note_data(note_data, key_details, relevant_kb, query_text)
            generated_note = self._format_private_note(note_data)
            log_info(f"Successfully generated note for ticket {ticket_id} ({len(generated_note)} chars)")

            return generated_note
        
        except Exception as e:
            log_error(f"Error generating Groq response", e)
            raise

    def _clean_text(self, text: str, max_len: int = 5000) -> str:
        """Strip HTML/noise and cap text length to keep prompts within model limits."""
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", str(text))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:max_len]

    def _extract_key_details(self, text: str) -> List[str]:
        """Extract key operational details to keep summary useful at-a-glance."""
        if not text:
            return []

        details: List[str] = []
        normalized = str(text)

        order_id_match = re.search(
            r"(?:order\s*id\s*[:#-]?\s*)([A-Za-z]{1,6}\s*[-]?\s*\d{3,}|[A-Za-z0-9\-]{5,30})",
            normalized,
            flags=re.IGNORECASE,
        )
        if order_id_match:
            order_id = self._clean_text(order_id_match.group(1), max_len=40)
            order_id = re.sub(r"\s{2,}", " ", order_id).strip(" -")
            details.append(f"Order ID: {order_id}")

        date_match = re.search(r"(?:expected\s*delivery\s*date\s*[:#-]?\s*)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", normalized, flags=re.IGNORECASE)
        if date_match:
            details.append(f"Expected Delivery Date: {date_match.group(1)}")

        return details

    def _is_contact_support_intent(self, text: str) -> bool:
        """Detect support-contact requests where KB channels (phone/chat/email) must be surfaced."""
        if not text:
            return False
        lowered = text.lower()
        triggers = [
            "contact support",
            "customer care",
            "customer service",
            "helpline",
            "phone number",
            "contact us",
            "reach myntra",
            "get in touch",
            "chat support",
        ]
        return any(trigger in lowered for trigger in triggers)

    def _ensure_contact_kb_coverage(self, query_text: str, knowledge_base: List[Dict[str, Any]], relevant_kb: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure contact-related KB article is included for contact-intent tickets."""
        if not self._is_contact_support_intent(query_text):
            return relevant_kb
        if not knowledge_base:
            return relevant_kb

        existing_text = " ".join(
            f"{a.get('title', '')} {a.get('description', '')}" for a in relevant_kb
        ).lower()
        if any(token in existing_text for token in ["contact", "customer care", "helpline", "support"]):
            return relevant_kb

        def score_article(article: Dict[str, Any]) -> int:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            keywords = ["contact", "customer care", "customer service", "helpline", "phone", "chat", "email", "help center"]
            return sum(1 for k in keywords if k in text)

        ranked = sorted(knowledge_base, key=score_article, reverse=True)
        if not ranked or score_article(ranked[0]) == 0:
            return relevant_kb

        merged = [ranked[0]] + relevant_kb
        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for article in merged:
            article_id = article.get("id")
            dedupe_key = f"id:{article_id}" if article_id is not None else f"title:{article.get('title', '')}"
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            deduped.append(article)
        return deduped[:3]

    def _extract_contact_steps_from_kb(self, relevant_kb: List[Dict[str, Any]]) -> List[str]:
        """Build concrete contact steps from KB details (phone/chat/email/help center)."""
        if not relevant_kb:
            return []

        kb_text = " ".join(self._article_text_for_retrieval(article, max_len=1800) for article in relevant_kb)
        lowered = kb_text.lower()

        phones = []
        for phone in re.findall(r"\+?\d[\d\-\u2010-\u2015\s]{7,}\d", kb_text):
            normalized = re.sub(r"\s{2,}", " ", phone).strip()
            if normalized and normalized not in phones:
                phones.append(normalized)

        steps: List[str] = []
        if "contact us" in lowered:
            steps.append("Guide the customer to App/Website > 'Contact Us' under the main menu.")
        if "chat" in lowered:
            steps.append("Offer live chat support via the Myntra app/website chat option.")
        if "email" in lowered or "help center" in lowered:
            steps.append("Share email support path via the Help Center on myntra.com.")
        if phones:
            steps.append(f"For urgent issues or shipping address changes, advise calling {phones[0]}.")
        if "insider elite" in lowered or "icon members" in lowered:
            steps.append("If the customer is Insider Elite/Icon, route through priority support handling.")
        return steps


    def _enrich_note_data(self, note_data: Dict[str, Any], key_details: List[str], relevant_kb: List[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
        """Normalize note quality: detailed summary + KB-grounded resolution steps."""
        summary = self._clean_text(str(note_data.get("summary", "")), max_len=350)
        next_action = self._clean_text(str(note_data.get("next_action", "")), max_len=220)

        kb_steps_raw = note_data.get("kb_steps", [])
        if isinstance(kb_steps_raw, str):
            kb_steps = [self._clean_text(kb_steps_raw, max_len=180)]
        elif isinstance(kb_steps_raw, list):
            kb_steps = [self._clean_text(str(step), max_len=180) for step in kb_steps_raw if str(step).strip()]
        else:
            kb_steps = []

        for detail in key_details:
            detail_value = detail.split(":", 1)[-1].strip().lower()
            if detail_value and detail_value not in summary.lower():
                summary = f"{summary} ({detail})" if summary else detail

        if relevant_kb:
            top_article = self._clean_text(relevant_kb[0].get("title", ""), max_len=120)
            if not kb_steps:
                kb_steps = [
                    f"Open KB article '{top_article}' and follow the documented troubleshooting path for this issue.",
                    "Verify the latest status update in the order system and compare it with the expected timeline.",
                ]

        if self._is_contact_support_intent(query_text):
            if not next_action:
                next_action = "Confirm the channel shared with customer and update the ticket with follow-up ownership."

        return {
            "summary": self._clean_text(summary, max_len=350),
            "kb_steps": kb_steps,
            "next_action": next_action,
        }

    def _parse_note_json(self, raw_output: str) -> Dict[str, Any]:
        """Parse model output as JSON with a resilient fallback."""
        if not raw_output:
            return {
                "summary": "Ticket details received. Review issue context and proceed with KB-guided troubleshooting.",
                "kb_steps": ["Review the matched KB article and apply the documented resolution steps."],
                "next_action": "If unresolved, gather latest error screenshot/logs and update the ticket."
            }

        # Try direct JSON parsing first.
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try extracting the first JSON object from markdown/code fences.
        match = re.search(r"\{[\s\S]*\}", raw_output)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Try parsing markdown-style notes (including collapsed one-line output).
        markdown_parsed = self._parse_markdown_note(raw_output)
        if markdown_parsed:
            return markdown_parsed

        return {
            "summary": self._clean_text(raw_output, max_len=300),
            "kb_steps": ["Use the most relevant KB article steps to investigate and resolve."],
            "next_action": "If unresolved, request additional details from requester and escalate if needed."
        }

    def _parse_markdown_note(self, raw_output: str) -> Dict[str, Any]:
        """Parse markdown note content into structured summary/steps/next_action."""
        if not raw_output:
            return {}

        normalized = str(raw_output).replace("\r", " ").strip()
        markers = [
            "## Private Note",
            "### Ticket Summary",
            "### Steps to Resolve",
            "### Next Update",
            "**Ticket Summary**",
            "**Steps to Resolve**",
            "**Next Update**",
        ]
        for marker in markers:
            normalized = normalized.replace(marker, f"\n{marker}\n")
        normalized = re.sub(r"\s+(\d+\.\s+)", r"\n\1", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

        def _extract_section(text: str, starts: List[str], ends: List[str]) -> str:
            start_re = "(?:" + "|".join(starts) + ")"
            end_re = "(?:" + "|".join(ends) + ")" if ends else "$"
            match = re.search(
                start_re + r"\s*(.*?)\s*(?=" + end_re + r"|$)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return match.group(1).strip() if match else ""

        summary = _extract_section(
            normalized,
            [r"###\s*Ticket Summary", r"\*\*Ticket Summary\*\*"],
            [r"###\s*Steps to Resolve", r"\*\*Steps to Resolve\*\*", r"###\s*Next Update", r"\*\*Next Update\*\*"],
        )
        steps_block = _extract_section(
            normalized,
            [r"###\s*Steps to Resolve", r"\*\*Steps to Resolve\*\*"],
            [r"###\s*Next Update", r"\*\*Next Update\*\*"],
        )
        next_action = _extract_section(
            normalized,
            [r"###\s*Next Update", r"\*\*Next Update\*\*"],
            [],
        )

        steps = [
            self._clean_text(step, max_len=180)
            for step in re.findall(r"(?:^|\n)\s*\d+\.\s*(.+?)(?=(?:\n\s*\d+\.\s)|$)", steps_block, flags=re.DOTALL)
            if self._clean_text(step, max_len=180)
        ]

        if not summary and not steps and not next_action:
            return {}

        return {
            "summary": self._clean_text(summary, max_len=350),
            "kb_steps": steps,
            "next_action": self._clean_text(next_action, max_len=220),
        }

    def _format_private_note(self, note_data: Dict[str, Any]) -> str:
        """Render a clean, consistent private note body."""
        summary = self._clean_text(str(note_data.get("summary", "")), max_len=350)
        next_action = self._clean_text(str(note_data.get("next_action", "")), max_len=220)

        kb_steps_raw = note_data.get("kb_steps", [])
        if isinstance(kb_steps_raw, str):
            kb_steps = [self._clean_text(kb_steps_raw, max_len=180)]
        elif isinstance(kb_steps_raw, list):
            kb_steps = [self._clean_text(str(step), max_len=180) for step in kb_steps_raw if str(step).strip()]
        else:
            kb_steps = []

        if not summary:
            summary = "Ticket details reviewed. Refer to relevant KB guidance for execution."
        if not kb_steps:
            kb_steps = ["No strong KB match found. Apply standard troubleshooting workflow."]
        if not next_action:
            next_action = "Update ticket with findings; escalate if issue remains unresolved."

        lines = [
            "### Ticket Summary",
            f"{summary}",
            "",
            "### Steps to Resolve",
        ]
        for idx, step in enumerate(kb_steps, 1):
            lines.append(f"{idx}. {step}")

        lines.extend([
            "",
            "### Next Update",
            f"{next_action}"
        ])

        return "\n".join(lines)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for simple lexical vector search."""
        normalized = self._clean_text(text.lower(), max_len=4000)
        tokens = re.findall(r"[a-z0-9]{3,}", normalized)
        stop_words = {
            "the", "and", "for", "that", "with", "this", "from", "your", "have", "been",
            "are", "was", "were", "can", "not", "you", "our", "but", "all", "any", "into",
            "how", "when", "what", "where", "why", "who", "ticket", "issue", "please", "help"
        }
        return [t for t in tokens if t not in stop_words]

    
    def build_prompt(self, subject: str, description: str, relevant_articles: List[Dict], 
                     priority: int, tags: List[str]) -> str:
        """Build the prompt for Groq LLM"""
        prompt = "Create a PRIVATE INTERNAL NOTE for support agents using the ticket details and knowledge base content.\n\n"
        
        prompt += f"**Ticket Subject:** {subject}\n"
        prompt += f"**Ticket Description:** {description}\n"
        prompt += f"**Priority:** {priority}\n"
        
        if tags:
            prompt += f"**Tags:** {', '.join(tags)}\n"
        
        if relevant_articles:
            prompt += "\n**Relevant Knowledge Base Articles:**\n"
            for idx, article in enumerate(relevant_articles, 1):
                title = self._clean_text(article.get("title", "N/A"), max_len=120)
                category = self._clean_text(article.get("category", "N/A"), max_len=60)
                desc = rag_service._article_text_for_retrieval(article, max_len=5000) or "No description"
                prompt += f"\n{idx}. **{title}** (Category: {category})\n"
                prompt += f"   {desc}\n"
        else:
            prompt += "\nNo specific knowledge base articles were found for this topic.\n"
        
        prompt += "\nReturn ONLY valid JSON with this exact schema (no markdown, no extra text):\n"
        prompt += "{\n"
        prompt += "  \"summary\": \"string\",\n"
        prompt += "  \"kb_steps\": [\"string\", \"string\"],\n"
        prompt += "  \"next_action\": \"string\"\n"
        prompt += "}\n"
        prompt += (
            "Rules: keep it internal and agent-focused, avoid customer-facing language, avoid apologies. "
            "Summary must be a detailed, comprehensive paragraph that deeply analyzes the core problem, user intent, and context behind the ticket. Do not just restate the subject. Write at least 3-4 sentences of descriptive analysis. Include key identifiers when present (e.g., Order ID, expected date). "
            "kb_steps must accurately extract step-by-step guidance directly from the most relevant KB article. "
            "CRITICAL EXPLICIT INSTRUCTION: If the KB article contains any specific information like email addresses or phone numbers, YOU MUST explicitly include them correctly in the kb_steps exactly as they appear in the text! "
            "next_action should state a clear escalation/owner checkpoint."
        )
        
        return prompt
    
    def generate_ticket_summary(self, tickets: List[Dict]) -> str:
        """Generate a summary of multiple tickets"""
        try:
            log_info(f"Generating summary for {len(tickets)} tickets")
            
            tickets_list = "\n".join([f"{i}. [#{t.get('id')}] {t.get('subject')}" 
                                     for i, t in enumerate(tickets, 1)])
            
            message = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a support team analyst. Summarize ticket trends and provide insights."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize these support tickets and identify common issues:\n\n{tickets_list}"
                    }
                ],
                temperature=0.7,
                max_tokens=300,
            )
            
            return message.choices[0].message.content
        
        except Exception as e:
            log_error("Error generating ticket summary", e)
            raise


# Create singleton instance
groq_service = GroqService()