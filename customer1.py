"""
Customer Support & FAQ AI Agent System
=====================================
A complete AI-powered support system with:
- Website Chat Widget
- WhatsApp Integration
- Email Auto-Responder
- Ticketing System
- Multilingual Support
- Analytics Dashboard
"""

# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

import os
import json
import datetime
import pytz
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """System configuration"""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FAISS_INDEX_PATH: str = "knowledge_base/faiss_index"
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "mr"]  # English, Hindi, Marathi
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_RESPONSE_TIME: int = 2  # seconds
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///support_agent.db")
    
config = Config()

# ============================================================================
# DATA MODELS
# ============================================================================

class QueryCategory(Enum):
    """Categories for customer queries"""
    BILLING = "billing"
    TECHNICAL = "technical"
    PRODUCT_INFO = "product_info"
    WARRANTY = "warranty"
    GENERAL = "general"
    COMPLAINT = "complaint"
    SERVICE = "service"

class CommunicationChannel(Enum):
    """Communication channels"""
    WEBSITE_CHAT = "website_chat"
    WHATSAPP = "whatsapp"
    EMAIL = "email"

class TicketStatus(Enum):
    """Support ticket status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class CustomerQuery:
    """Represents a customer query"""
    record_id: str
    business_unit: str
    customer_query: str
    query_category: QueryCategory
    language: str
    communication_channel: CommunicationChannel
    timestamp: datetime.datetime
    
    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "business_unit": self.business_unit,
            "customer_query": self.customer_query,
            "query_category": self.query_category.value,
            "language": self.language,
            "communication_channel": self.communication_channel.value,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class AIResponse:
    """AI-generated response"""
    query: CustomerQuery
    response_text: str
    confidence_score: float
    ticket_created: bool
    ticket_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query.to_dict(),
            "response_text": self.response_text,
            "confidence_score": self.confidence_score,
            "ticket_created": self.ticket_created,
            "ticket_id": self.ticket_id
        }

@dataclass
class SupportTicket:
    """Support ticket for escalation"""
    ticket_id: str
    query: CustomerQuery
    ai_response: AIResponse
    assigned_to: Optional[str] = None
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime.datetime = datetime.datetime.now(pytz.UTC)
    
    def to_dict(self) -> Dict:
        return {
            "ticket_id": self.ticket_id,
            "query": self.query.to_dict(),
            "ai_response": self.ai_response.to_dict(),
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at.isoformat()
        }

# ============================================================================
# KNOWLEDGE BASE INGESTION MODULE
# ============================================================================

class KnowledgeBaseIngestor:
    """Handles ingestion of various document types into searchable knowledge base"""
    
    def __init__(self):
        self.vector_db = None
        self.knowledge_chunks = []
        
    def ingest_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF files"""
        try:
            import PyPDF2
            text_chunks = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        chunks = self._chunk_text(text)
                        text_chunks.extend(chunks)
            logger.info(f"Ingested PDF: {pdf_path} - {len(text_chunks)} chunks")
            return text_chunks
        except Exception as e:
            logger.error(f"Error ingesting PDF {pdf_path}: {e}")
            return []
    
    def ingest_website(self, url: str) -> List[str]:
        """Crawl website and extract content"""
        try:
            from bs4 import BeautifulSoup
            import requests
            
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            chunks = self._chunk_text(text)
            logger.info(f"Ingested website: {url} - {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error ingesting website {url}: {e}")
            return []
    
    def ingest_docx(self, docx_path: str) -> List[str]:
        """Extract text from DOCX files"""
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            chunks = self._chunk_text(text)
            logger.info(f"Ingested DOCX: {docx_path} - {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error ingesting DOCX {docx_path}: {e}")
            return []
    
    def add_faq(self, faq_list: List[Dict[str, str]]) -> List[str]:
        """Add FAQ entries to knowledge base"""
        chunks = []
        for faq in faq_list:
            chunk = f"Q: {faq.get('question')}\nA: {faq.get('answer')}"
            chunks.append(chunk)
        logger.info(f"Added {len(chunks)} FAQ entries")
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]):
        """Create vector embeddings for knowledge chunks"""
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            # Load embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))
            
            # Save index
            faiss.write_index(index, config.FAISS_INDEX_PATH)
            self.knowledge_chunks = chunks
            logger.info(f"Created embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """Search knowledge base for relevant information"""
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            if not os.path.exists(config.FAISS_INDEX_PATH):
                return []
            
            # Load model and index
            model = SentenceTransformer('all-MiniLM-L6-v2')
            index = faiss.read_index(config.FAISS_INDEX_PATH)
            
            # Encode query
            query_embedding = model.encode([query])
            
            # Search
            distances, indices = index.search(
                np.array(query_embedding).astype('float32'), 
                top_k
            )
            
            results = []
            for idx in indices[0]:
                if idx < len(self.knowledge_chunks):
                    results.append(self.knowledge_chunks[idx])
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

# ============================================================================
# AI RESPONSE GENERATOR
# ============================================================================

class AIResponseGenerator:
    """Generates AI responses using LLM"""
    
    def __init__(self, knowledge_base: KnowledgeBaseIngestor):
        self.knowledge_base = knowledge_base
        
    def generate_response(self, query: CustomerQuery) -> AIResponse:
        """Generate AI response for customer query"""
        
        # Search knowledge base
        context_chunks = self.knowledge_base.search_knowledge(query.customer_query)
        context = "\n\n".join(context_chunks)
        
        # Prepare prompt
        prompt = self._build_prompt(query, context)
        
        try:
            # Call LLM API (using OpenAI as example)
            import openai
            
            openai.api_key = config.OPENAI_API_KEY
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # Calculate confidence score (simplified)
            confidence = self._calculate_confidence(query.customer_query, context_chunks, response_text)
            
            # Decide if ticket needs to be created
            ticket_created = confidence < config.CONFIDENCE_THRESHOLD
            ticket_id = None
            
            if ticket_created:
                ticket_id = self._generate_ticket_id(query)
                response_text = f"{response_text}\n\nYour query has been escalated to our support team. Ticket ID: {ticket_id}"
            
            return AIResponse(
                query=query,
                response_text=response_text,
                confidence_score=confidence,
                ticket_created=ticket_created,
                ticket_id=ticket_id
            )
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            # Fallback response
            return AIResponse(
                query=query,
                response_text="I apologize, but I'm having trouble processing your request. Our support team will contact you shortly.",
                confidence_score=0.0,
                ticket_created=True,
                ticket_id=self._generate_ticket_id(query)
            )
    
    def _build_prompt(self, query: CustomerQuery, context: str) -> str:
        """Build prompt for LLM"""
        language_map = {
            "en": "English",
            "hi": "Hindi",
            "mr": "Marathi"
        }
        
        prompt = f"""
        Customer Query: {query.customer_query}
        Query Language: {language_map.get(query.language, 'English')}
        Business Unit: {query.business_unit}
        
        Context from Knowledge Base:
        {context}
        
        Instructions:
        1. Respond in the same language as the query
        2. Use the provided context to answer accurately
        3. If information is not available, admit you don't know
        4. Be helpful and professional
        5. Keep response concise but complete
        
        Response:
        """
        return prompt
    
    def _calculate_confidence(self, query: str, context: List[str], response: str) -> float:
        """Calculate confidence score for AI response"""
        # Simplified confidence calculation
        # In production, this would use more sophisticated methods
        
        if not context:
            return 0.0
        
        # Check if response contains "I don't know" type phrases
        low_confidence_phrases = [
            "I don't know",
            "I'm not sure",
            "I cannot answer",
            "please contact support",
            "escalated to our team"
        ]
        
        for phrase in low_confidence_phrases:
            if phrase.lower() in response.lower():
                return 0.3
        
        # Higher confidence if context was found
        return 0.8 if len(context) > 0 else 0.2
    
    def _generate_ticket_id(self, query: CustomerQuery) -> str:
        """Generate unique ticket ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        business_prefix = query.business_unit[:3].upper()
        return f"TICKET-{business_prefix}-{timestamp}"

# ============================================================================
# CHAT SUPPORT AGENT (WEBSITE WIDGET)
# ============================================================================

class ChatSupportAgent:
    """Website chat widget implementation"""
    
    def __init__(self, ai_generator: AIResponseGenerator):
        self.ai_generator = ai_generator
        self.chat_sessions = {}
    
    def process_message(self, session_id: str, message: str, 
                       business_unit: str = "default") -> Dict:
        """Process incoming chat message"""
        
        # Detect language
        language = self._detect_language(message)
        
        # Create query object
        query = CustomerQuery(
            record_id=self._generate_record_id(),
            business_unit=business_unit,
            customer_query=message,
            query_category=self._categorize_query(message),
            language=language,
            communication_channel=CommunicationChannel.WEBSITE_CHAT,
            timestamp=datetime.datetime.now(pytz.UTC)
        )
        
        # Generate AI response
        response = self.ai_generator.generate_response(query)
        
        # Store in session
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []
        
        self.chat_sessions[session_id].append({
            "query": query.to_dict(),
            "response": response.to_dict()
        })
        
        # Format response for chat
        chat_response = {
            "session_id": session_id,
            "message": response.response_text,
            "ticket_created": response.ticket_created,
            "ticket_id": response.ticket_id,
            "confidence": response.confidence_score,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }
        
        return chat_response
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        # Simplified language detection
        # In production, use libraries like langdetect or fasttext
        
        hindi_chars = set("अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथधनपफबभमयरलवशषसह")
        marathi_chars = set("अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथधनपफबभमयरलवशषसह")
        
        text_chars = set(text)
        
        if text_chars.intersection(hindi_chars):
            return "hi"
        elif text_chars.intersection(marathi_chars):
            return "mr"
        else:
            return "en"
    
    def _categorize_query(self, text: str) -> QueryCategory:
        """Categorize the query"""
        text_lower = text.lower()
        
        category_keywords = {
            QueryCategory.BILLING: ["payment", "bill", "invoice", "refund", "price", "cost"],
            QueryCategory.TECHNICAL: ["error", "bug", "technical", "issue", "problem", "not working"],
            QueryCategory.PRODUCT_INFO: ["what is", "how to", "feature", "specification", "details"],
            QueryCategory.WARRANTY: ["warranty", "guarantee", "repair", "replace"],
            QueryCategory.SERVICE: ["service", "install", "setup", "configure", "maintenance"],
            QueryCategory.COMPLAINT: ["complaint", "bad", "poor", "terrible", "angry", "frustrated"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return QueryCategory.GENERAL
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID"""
        return f"CHAT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"

# ============================================================================
# WHATSAPP SUPPORT AGENT
# ============================================================================

class WhatsAppSupportAgent:
    """WhatsApp Business API integration"""
    
    def __init__(self, ai_generator: AIResponseGenerator):
        self.ai_generator = ai_generator
    
    def process_whatsapp_message(self, from_number: str, message: str,
                                business_unit: str = "default") -> Dict:
        """Process incoming WhatsApp message"""
        
        language = self._detect_language(message)
        
        query = CustomerQuery(
            record_id=self._generate_record_id(),
            business_unit=business_unit,
            customer_query=message,
            query_category=self._categorize_query(message),
            language=language,
            communication_channel=CommunicationChannel.WHATSAPP,
            timestamp=datetime.datetime.now(pytz.UTC)
        )
        
        response = self.ai_generator.generate_response(query)
        
        # Format for WhatsApp API
        whatsapp_response = {
            "to": from_number,
            "message": response.response_text,
            "ticket_created": response.ticket_created,
            "ticket_id": response.ticket_id,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }
        
        return whatsapp_response
    
    def _detect_language(self, text: str) -> str:
        """Detect language for WhatsApp messages"""
        return ChatSupportAgent._detect_language(self, text)
    
    def _categorize_query(self, text: str) -> QueryCategory:
        """Categorize WhatsApp queries"""
        return ChatSupportAgent._categorize_query(self, text)
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID for WhatsApp"""
        return f"WA-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"

# ============================================================================
# EMAIL AUTO-RESPONDER
# ============================================================================

class EmailAutoResponder:
    """Email auto-responder system"""
    
    def __init__(self, ai_generator: AIResponseGenerator):
        self.ai_generator = ai_generator
    
    def process_email(self, from_email: str, subject: str, body: str,
                     business_unit: str = "default") -> Dict:
        """Process incoming support email"""
        
        # Combine subject and body for processing
        full_text = f"{subject}\n\n{body}"
        language = self._detect_language(full_text)
        
        query = CustomerQuery(
            record_id=self._generate_record_id(),
            business_unit=business_unit,
            customer_query=full_text,
            query_category=self._categorize_email(subject, body),
            language=language,
            communication_channel=CommunicationChannel.EMAIL,
            timestamp=datetime.datetime.now(pytz.UTC)
        )
        
        response = self.ai_generator.generate_response(query)
        
        # Format email response
        email_response = {
            "to": from_email,
            "subject": f"Re: {subject}",
            "body": response.response_text,
            "ticket_created": response.ticket_created,
            "ticket_id": response.ticket_id,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }
        
        return email_response
    
    def _categorize_email(self, subject: str, body: str) -> QueryCategory:
        """Categorize email content"""
        full_text = f"{subject} {body}".lower()
        
        if any(word in full_text for word in ["invoice", "payment", "bill", "refund"]):
            return QueryCategory.BILLING
        elif any(word in full_text for word in ["error", "bug", "technical", "issue"]):
            return QueryCategory.TECHNICAL
        elif any(word in full_text for word in ["warranty", "guarantee"]):
            return QueryCategory.WARRANTY
        elif any(word in full_text for word in ["service", "install", "maintenance"]):
            return QueryCategory.SERVICE
        
        return QueryCategory.GENERAL
    
    def _detect_language(self, text: str) -> str:
        """Detect language for emails"""
        return ChatSupportAgent._detect_language(self, text)
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID for emails"""
        return f"EMAIL-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"

# ============================================================================
# TICKET MANAGEMENT SYSTEM
# ============================================================================

class TicketManagementSystem:
    """Manages support ticket creation and tracking"""
    
    def __init__(self):
        self.tickets = {}
        self.support_staff = ["support1@company.com", "support2@company.com"]
    
    def create_ticket(self, query: CustomerQuery, 
                     ai_response: AIResponse) -> SupportTicket:
        """Create a new support ticket"""
        
        ticket_id = ai_response.ticket_id or self._generate_ticket_id()
        
        ticket = SupportTicket(
            ticket_id=ticket_id,
            query=query,
            ai_response=ai_response,
            assigned_to=self._assign_to_staff(),
            status=TicketStatus.OPEN
        )
        
        self.tickets[ticket_id] = ticket
        
        # Notify support staff
        self._notify_staff(ticket)
        
        logger.info(f"Created ticket: {ticket_id}")
        return ticket
    
    def update_ticket_status(self, ticket_id: str, status: TicketStatus,
                            assigned_to: Optional[str] = None) -> bool:
        """Update ticket status"""
        if ticket_id in self.tickets:
            self.tickets[ticket_id].status = status
            if assigned_to:
                self.tickets[ticket_id].assigned_to = assigned_to
            return True
        return False
    
    def get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        """Get ticket by ID"""
        return self.tickets.get(ticket_id)
    
    def get_all_tickets(self, status: Optional[TicketStatus] = None) -> List[SupportTicket]:
        """Get all tickets, optionally filtered by status"""
        if status:
            return [ticket for ticket in self.tickets.values() 
                   if ticket.status == status]
        return list(self.tickets.values())
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID"""
        return f"TICKET-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _assign_to_staff(self) -> str:
        """Assign ticket to support staff (round-robin)"""
        if not hasattr(self, '_staff_index'):
            self._staff_index = 0
        
        staff = self.support_staff[self._staff_index]
        self._staff_index = (self._staff_index + 1) % len(self.support_staff)
        return staff
    
    def _notify_staff(self, ticket: SupportTicket):
        """Notify support staff about new ticket"""
        # In production, this would send email/WhatsApp notification
        logger.info(f"Notified {ticket.assigned_to} about ticket {ticket.ticket_id}")

# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

class AnalyticsDashboard:
    """Analytics and reporting dashboard"""
    
    def __init__(self, chat_agent: ChatSupportAgent, 
                 whatsapp_agent: WhatsAppSupportAgent,
                 email_agent: EmailAutoResponder,
                 ticket_system: TicketManagementSystem):
        
        self.chat_agent = chat_agent
        self.whatsapp_agent = whatsapp_agent
        self.email_agent = email_agent
        self.ticket_system = ticket_system
        self.analytics_data = {
            "daily_queries": [],
            "resolved_vs_escalated": {"resolved": 0, "escalated": 0},
            "category_distribution": {},
            "language_distribution": {},
            "channel_distribution": {},
            "satisfaction_scores": {"thumbs_up": 0, "thumbs_down": 0}
        }
    
    def generate_report(self, date: Optional[datetime.date] = None) -> Dict:
        """Generate analytics report"""
        if date is None:
            date = datetime.date.today()
        
        report = {
            "date": date.isoformat(),
            "total_queries": self._count_total_queries(date),
            "auto_resolved": self._count_auto_resolved(date),
            "escalated_tickets": self._count_escalated_tickets(date),
            "category_breakdown": self._get_category_breakdown(date),
            "language_distribution": self._get_language_distribution(date),
            "channel_distribution": self._get_channel_distribution(date),
            "top_queries": self._get_top_queries(date, limit=10),
            "average_response_time": self._calculate_avg_response_time(date),
            "satisfaction_score": self._calculate_satisfaction_score()
        }
        
        return report
    
    def record_feedback(self, session_id: str, thumbs_up: bool):
        """Record customer feedback"""
        if thumbs_up:
            self.analytics_data["satisfaction_scores"]["thumbs_up"] += 1
        else:
            self.analytics_data["satisfaction_scores"]["thumbs_down"] += 1
    
    def _count_total_queries(self, date: datetime.date) -> int:
        """Count total queries for a date"""
        # Simplified - in production, this would query a database
        return 0
    
    def _count_auto_resolved(self, date: datetime.date) -> int:
        """Count auto-resolved queries"""
        return 0
    
    def _count_escalated_tickets(self, date: datetime.date) -> int:
        """Count escalated tickets"""
        return 0
    
    def _get_category_breakdown(self, date: datetime.date) -> Dict:
        """Get query category breakdown"""
        return {}
    
    def _get_language_distribution(self, date: datetime.date) -> Dict:
        """Get language distribution"""
        return {}
    
    def _get_channel_distribution(self, date: datetime.date) -> Dict:
        """Get channel distribution"""
        return {}
    
    def _get_top_queries(self, date: datetime.date, limit: int = 10) -> List[str]:
        """Get top queries"""
        return []
    
    def _calculate_avg_response_time(self, date: datetime.date) -> float:
        """Calculate average response time"""
        return 0.0
    
    def _calculate_satisfaction_score(self) -> float:
        """Calculate satisfaction score"""
        total = (self.analytics_data["satisfaction_scores"]["thumbs_up"] + 
                self.analytics_data["satisfaction_scores"]["thumbs_down"])
        
        if total == 0:
            return 0.0
        
        return (self.analytics_data["satisfation_scores"]["thumbs_up"] / total) * 100

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class CustomerSupportAIApplication:
    """Main application orchestrator"""
    
    def __init__(self):
        # Initialize components
        self.knowledge_base = KnowledgeBaseIngestor()
        self.ai_generator = AIResponseGenerator(self.knowledge_base)
        self.chat_agent = ChatSupportAgent(self.ai_generator)
        self.whatsapp_agent = WhatsAppSupportAgent(self.ai_generator)
        self.email_agent = EmailAutoResponder(self.ai_generator)
        self.ticket_system = TicketManagementSystem()
        self.dashboard = AnalyticsDashboard(
            self.chat_agent, self.whatsapp_agent, 
            self.email_agent, self.ticket_system
        )
        
        logger.info("Customer Support AI Application initialized")
    
    def run_demo(self):
        """Run a demonstration of the system"""
        print("=" * 60)
        print("CUSTOMER SUPPORT AI AGENT - DEMONSTRATION")
        print("=" * 60)
        
        # 1. Ingest knowledge base
        print("\n1. Knowledge Base Ingestion")
        print("-" * 40)
        
        # Example FAQ ingestion
        faqs = [
            {"question": "What is the warranty period?", 
             "answer": "Our products come with a 1-year warranty."},
            {"question": "How can I book a service?", 
             "answer": "Visit our website at example.com/service-booking."}
        ]
        
        faq_chunks = self.knowledge_base.add_faq(faqs)
        print(f"Ingested {len(faq_chunks)} FAQ entries")
        
        # 2. Process website chat
        print("\n2. Website Chat Support")
        print("-" * 40)
        
        chat_response = self.chat_agent.process_message(
            session_id="demo_session_1",
            message="What is the warranty period for your products?",
            business_unit="Electronics"
        )
        
        print(f"Query: {chat_response.get('message', '')}")
        print(f"Response: {chat_response.get('response_text', '')}")
        print(f"Confidence: {chat_response.get('confidence', 0):.2f}")
        
        # 3. Process WhatsApp message
        print("\n3. WhatsApp Support (Hindi)")
        print("-" * 40)
        
        whatsapp_response = self.whatsapp_agent.process_whatsapp_message(
            from_number="+911234567890",
            message="Warranty kitna hai?",
            business_unit="Home Appliances"
        )
        
        print(f"Query (Hindi): Warranty kitna hai?")
        print(f"Response: {whatsapp_response.get('message', '')}")
        
        # 4. Process email
        print("\n4. Email Auto-Responder")
        print("-" * 40)
        
        email_response = self.email_agent.process_email(
            from_email="customer@example.com",
            subject="Service Booking Inquiry",
            body="Hello, I would like to book a service for my AC unit.",
            business_unit="AC Services"
        )
        
        print(f"Email Subject: {email_response.get('subject', '')}")
        print(f"Auto-response: {email_response.get('body', '')[:100]}...")
        
        # 5. Generate analytics report
        print("\n5. Analytics Dashboard")
        print("-" * 40)
        
        report = self.dashboard.generate_report()
        print(f"Date: {report.get('date')}")
        print(f"Total Queries: {report.get('total_queries')}")
        print(f"Auto Resolved: {report.get('auto_resolved')}")
        print(f"Escalated Tickets: {report.get('escalated_tickets')}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)

# ============================================================================
# WEB API (Django-like structure)
# ============================================================================

class SupportAPI:
    """REST API for the support system"""
    
    def __init__(self, app: CustomerSupportAIApplication):
        self.app = app
    
    def chat_endpoint(self, request: Dict) -> Dict:
        """Handle chat requests"""
        session_id = request.get("session_id", "default")
        message = request.get("message", "")
        business_unit = request.get("business_unit", "default")
        
        response = self.app.chat_agent.process_message(
            session_id, message, business_unit
        )
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }
    
    def whatsapp_endpoint(self, request: Dict) -> Dict:
        """Handle WhatsApp webhook requests"""
        from_number = request.get("from")
        message = request.get("message")
        business_unit = request.get("business_unit", "default")
        
        response = self.app.whatsapp_agent.process_whatsapp_message(
            from_number, message, business_unit
        )
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }
    
    def email_endpoint(self, request: Dict) -> Dict:
        """Handle email processing"""
        from_email = request.get("from")
        subject = request.get("subject", "")
        body = request.get("body", "")
        business_unit = request.get("business_unit", "default")
        
        response = self.app.email_agent.process_email(
            from_email, subject, body, business_unit
        )
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }
    
    def analytics_endpoint(self, request: Dict) -> Dict:
        """Get analytics data"""
        date_str = request.get("date")
        date = None
        
        if date_str:
            date = datetime.datetime.fromisoformat(date_str).date()
        
        report = self.app.dashboard.generate_report(date)
        
        return {
            "status": "success",
            "data": report,
            "timestamp": datetime.datetime.now(pytz.UTC).isoformat()
        }

# ============================================================================
# DEPLOYMENT AND MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    # Initialize the application
    app = CustomerSupportAIApplication()
    
    # Run demonstration
    app.run_demo()
    
    # Initialize API
    api = SupportAPI(app)
    
    print("\n\nAPI Endpoints Available:")
    print("- /api/chat - For website chat")
    print("- /api/whatsapp - For WhatsApp integration")
    print("- /api/email - For email processing")
    print("- /api/analytics - For dashboard data")
    
    return app

if __name__ == "__main__":
    main()
