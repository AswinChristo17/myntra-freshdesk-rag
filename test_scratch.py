import os
from dotenv import load_dotenv
load_dotenv()

from services.freshdesk_service import freshdesk_service
from services.rag_service import rag_service

kb = freshdesk_service.get_knowledge_base()
subject = 'Inquiry Regarding Ongoing Sales and Promotions – Myntra'
description = 'I would like to get more information about the ongoing sales and promotional offers on Myntra. I noticed there are discounts available, but I need clarification regarding the eligibility, applicable products, and terms & conditions.'

query_text = subject + ' ' + description
relevant_kb = rag_service.filter_relevant_articles(query_text, kb)

for i, article in enumerate(relevant_kb):
    print(f"Index: {i} | Score: {article.get('score'):.3f} | Snippet: {str(article.get('content', ''))[:200].strip()}")
