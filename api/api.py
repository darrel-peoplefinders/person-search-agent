from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging

# Add error handling for imports
try:
    from main import (
        person_search_agent,
        create_initial_context,
    )
    print("✓ Successfully imported from main")
except ImportError as e:
    print(f"❌ Failed to import from main: {e}")
    # Create mock implementations for testing
    class MockAgent:
        name = "person_search_agent"
        input_guardrails = []
        tools = []
    
    person_search_agent = MockAgent()
    
    def create_initial_context():
        from pydantic import BaseModel
        class MockContext(BaseModel):
            query: str = ""
            results: List[Dict[str, Any]] = []
        return MockContext()

try:
    from agents import (
        Runner,
        ItemHelpers,
        MessageOutputItem,
        HandoffOutputItem,
        ToolCallItem,
        ToolCallOutputItem,
        InputGuardrailTripwireTriggered,
        Handoff,
    )
    print("✓ Successfully imported from agents")
except ImportError as e:
    print(f"❌ Failed to import from agents: {e}")
    # Create mock implementations for testing
    class MockRunner:
        @staticmethod
        async def run(agent, input_items, context=None):
            class MockResult:
                new_items = []
                context_wrapper = None
                def to_input_list(self):
                    return []
            return MockResult()
    
    class MockItemHelpers:
        @staticmethod
        def text_message_output(item):
            return "Mock response"
    
    class MockMessageOutputItem:
        def __init__(self, agent_name="mock"):
            self.agent = type('obj', (object,), {'name': agent_name})
    
    class MockInputGuardrailTripwireTriggered(Exception):
        def __init__(self):
            self.guardrail_result = type('obj', (object,), {
                'guardrail': None,
                'output': type('obj', (object,), {
                    'output_info': type('obj', (object,), {'reasoning': 'Mock reasoning'})
                })
            })
    
    Runner = MockRunner()
    ItemHelpers = MockItemHelpers()
    MessageOutputItem = MockMessageOutputItem
    HandoffOutputItem = None
    ToolCallItem = None
    ToolCallOutputItem = None
    InputGuardrailTripwireTriggered = MockInputGuardrailTripwireTriggered
    Handoff = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Person Search Engine", version="1.0.0")

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle preflight OPTIONS requests
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:8080",
        "https://lovable.dev",
        "https://*.lovable.dev",  # Allow all lovable subdomains
        "*"  # Temporarily allow all for testing (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class SearchResult(BaseModel):
    id: str
    confidence: int
    first_name: str
    last_name: str
    age: int
    city: str
    state: str
    previous_addresses: List[str]
    relatives: List[str]
    timeline_match: str
    professional_background: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []
    search_results: Optional[List[SearchResult]] = None

# =========================
# In-memory store for conversation state
# =========================

class ConversationStore:
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass

class InMemoryConversationStore(ConversationStore):
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

conversation_store = InMemoryConversationStore()

# =========================
# Helpers
# =========================

def _get_guardrail_name(g) -> str:
    """Extract a friendly guardrail name."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of available agents and their metadata."""
    def make_agent_dict(agent):
        return {
            "name": getattr(agent, "name", "person_search_agent"),
            "description": "AI agent for conversational person search",
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
        }
    return [make_agent_dict(person_search_agent)]

def _extract_search_results(context: Dict[str, Any]) -> Optional[List[SearchResult]]:
    """Extract search results from context if available."""
    results = context.get("results")
    if not results:
        return None
    
    search_results = []
    for result in results:
        try:
            search_results.append(SearchResult(
                id=result.get("id", str(uuid4())),
                confidence=result.get("confidence", 0),
                first_name=result.get("first_name", ""),
                last_name=result.get("last_name", ""),
                age=result.get("age", 0),
                city=result.get("city", ""),
                state=result.get("state", ""),
                previous_addresses=result.get("previous_addresses", []),
                relatives=result.get("relatives", []),
                timeline_match=result.get("timeline_match", ""),
                professional_background=result.get("professional_background")
            ))
        except Exception as e:
            logger.warning(f"Failed to parse search result: {e}")
    
    return search_results

# =========================
# Main Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for person search conversations.
    Handles conversation state, search queries, and result presentation.
    """
    try:
        # Initialize or retrieve conversation state
        is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
        
        if is_new:
            conversation_id: str = req.conversation_id or uuid4().hex
            ctx = create_initial_context()
            current_agent_name = getattr(person_search_agent, "name", "person_search_agent")
            state: Dict[str, Any] = {
                "input_items": [],
                "context": ctx,
                "current_agent": current_agent_name,
            }
            
            # Handle empty initial message
            if req.message.strip() == "":
                conversation_store.save(conversation_id, state)
                return ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=current_agent_name,
                    messages=[MessageResponse(
                        content="Hi! I'm here to help you find someone. Who are you looking for?",
                        agent=current_agent_name
                    )],
                    events=[],
                    context=ctx.model_dump() if hasattr(ctx, 'model_dump') else {},
                    agents=_build_agents_list(),
                    guardrails=[],
                    search_results=None,
                )
        else:
            conversation_id = req.conversation_id
            state = conversation_store.get(conversation_id)
            logger.info(f"Retrieved conversation {conversation_id}")

        current_agent = person_search_agent
        
        # Build input items for this turn
        input_items = state["input_items"] + [{"content": req.message, "role": "user"}]
        
        old_context = state["context"].model_dump().copy() if hasattr(state["context"], 'model_dump') else {}
        guardrail_checks: List[GuardrailCheck] = []

        try:
            # Run agent with conversation history and context
            result = await Runner.run(
                current_agent, 
                input_items, 
                context=state["context"]
            )
        except Exception as e:
            if "InputGuardrailTripwireTriggered" in str(type(e)):
                # Handle guardrail violations
                refusal = "I can only help with legitimate person search requests. Please ensure your search intent is appropriate."
                
                state["input_items"] = input_items + [{"role": "assistant", "content": refusal}]
                conversation_store.save(conversation_id, state)
                
                return ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=getattr(current_agent, "name", "person_search_agent"),
                    messages=[MessageResponse(content=refusal, agent=getattr(current_agent, "name", "person_search_agent"))],
                    events=[],
                    context=state["context"].model_dump() if hasattr(state["context"], 'model_dump') else {},
                    agents=_build_agents_list(),
                    guardrails=guardrail_checks,
                    search_results=None,
                )
            else:
                # Handle other errors
                logger.error(f"Error running agent: {e}")
                error_msg = "I'm having trouble processing your request right now. Please try again."
                return ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=getattr(current_agent, "name", "person_search_agent"),
                    messages=[MessageResponse(content=error_msg, agent=getattr(current_agent, "name", "person_search_agent"))],
                    events=[],
                    context=state["context"].model_dump() if hasattr(state["context"], 'model_dump') else {},
                    agents=_build_agents_list(),
                    guardrails=[],
                    search_results=None,
                )

        messages: List[MessageResponse] = []
        events: List[AgentEvent] = []

        # Process all new items from the agent response
        for item in getattr(result, 'new_items', []):
            if hasattr(item, 'agent') and hasattr(item.agent, 'name'):
                agent_name = item.agent.name
            else:
                agent_name = getattr(current_agent, "name", "person_search_agent")
                
            if "MessageOutputItem" in str(type(item)):
                text = ItemHelpers.text_message_output(item) if hasattr(ItemHelpers, 'text_message_output') else str(item)
                messages.append(MessageResponse(content=text, agent=agent_name))
                events.append(AgentEvent(
                    id=uuid4().hex, 
                    type="message", 
                    agent=agent_name, 
                    content=text,
                    timestamp=time.time() * 1000
                ))
            elif "ToolCallItem" in str(type(item)):
                tool_name = getattr(getattr(item, 'raw_item', None), "name", "tool_call")
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_call",
                        agent=agent_name,
                        content=tool_name,
                        metadata={"tool_name": tool_name},
                        timestamp=time.time() * 1000,
                    )
                )
            elif "ToolCallOutputItem" in str(type(item)):
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_output",
                        agent=agent_name,
                        content=str(getattr(item, 'output', '')),
                        metadata={"tool_result": getattr(item, 'output', None)},
                        timestamp=time.time() * 1000,
                    )
                )

        # Update conversation state with new context
        if hasattr(result, 'context_wrapper') and result.context_wrapper:
            state["context"] = result.context_wrapper.context

        # Update conversation state
        if hasattr(result, 'to_input_list'):
            state["input_items"] = result.to_input_list()
        else:
            state["input_items"] = input_items
            
        state["current_agent"] = getattr(current_agent, "name", "person_search_agent")
        conversation_store.save(conversation_id, state)
        
        logger.info(f"Saved conversation {conversation_id}")

        # Extract search results if available
        context_dict = state["context"].model_dump() if hasattr(state["context"], 'model_dump') else {}
        search_results = _extract_search_results(context_dict)

        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=getattr(current_agent, "name", "person_search_agent"),
            messages=messages,
            events=events,
            context=context_dict,
            agents=_build_agents_list(),
            guardrails=[],
            search_results=search_results,
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return ChatResponse(
            conversation_id=req.conversation_id or uuid4().hex,
            current_agent="person_search_agent",
            messages=[MessageResponse(
                content="I'm experiencing technical difficulties. Please try again later.",
                agent="person_search_agent"
            )],
            events=[],
            context={},
            agents=_build_agents_list(),
            guardrails=[],
            search_results=None,
        )

# =========================
# Additional Endpoints
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Person Search Engine"}

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "AI Person Search Engine",
        "version": "1.0.0",
        "description": "Conversational AI-powered person search using natural language queries",
        "endpoints": {
            "chat": "POST /chat - Main conversation endpoint",
            "health": "GET /health - Health check"
        }
    }

# Debug endpoint to check imports
@app.get("/debug")
async def debug_imports():
    """Debug endpoint to check import status."""
    return {
        "imports": {
            "main_module": "main" in globals(),
            "agents_module": "Runner" in globals(),
            "person_search_agent": person_search_agent is not None,
        },
        "app_status": "FastAPI app initialized successfully"
    }

# Ensure app is available for uvicorn
print("✓ FastAPI app created and configured")

@app.post("/debug/enformion")
async def debug_enformion_raw(request: dict):
    """
    Debug endpoint to get raw EnformionGo API response
    
    Request body:
    {
        "first_name": "Jennifer",
        "last_name": "Diaz", 
        "city": "Lancaster",
        "state": "California"
    }
    """
    try:
        from enformion_api import create_enformion_client, PersonSearchParams
        
        # Extract search parameters
        first_name = request.get("first_name", "")
        last_name = request.get("last_name", "")
        city = request.get("city")
        state = request.get("state")
        age = request.get("age")
        
        if not first_name or not last_name:
            return {"error": "first_name and last_name are required"}
        
        # Create search parameters
        search_params = PersonSearchParams(
            first_name=first_name,
            last_name=last_name,
            city=city,
            state=state,
            age=age
        )
        
        # Try EnformionGo API
        enformion_client = create_enformion_client()
        
        if not enformion_client:
            return {
                "error": "EnformionGo client not configured",
                "message": "Missing ENFORMION_AP_NAME or ENFORMION_AP_PASSWORD environment variables",
                "using_mock": True,
                "mock_data": "Would return mock data here"
            }
        
        # Make raw API call and capture response
        import httpx
        from datetime import datetime
        
        # Build the exact same request that would be sent to EnformionGo
        payload = {
            "FirstName": search_params.first_name,
            "LastName": search_params.last_name,
            "Page": 1,
            "ResultsPerPage": 2,
            "Includes": [
                "Addresses", 
                "PhoneNumbers", 
                "EmailAddresses", 
                "DatesOfBirth",
                "RelativesSummary",
                "AssociatesSummary"
            ]
        }
        
        # Add optional parameters
        if search_params.city or search_params.state:
            addresses = []
            address_line2_parts = []
            
            if search_params.city:
                address_line2_parts.append(search_params.city)
            if search_params.state:
                address_line2_parts.append(search_params.state)
                
            if address_line2_parts:
                addresses.append({
                    "AddressLine2": ", ".join(address_line2_parts)
                })
                payload["Addresses"] = addresses
        
        if search_params.age:
            payload["Age"] = search_params.age
        
        # Make the actual API call
        try:
            async with httpx.AsyncClient(
                timeout=30,
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "galaxy-ap-name": enformion_client.config.ap_name,
                    "galaxy-ap-password": enformion_client.config.ap_password,
                    "galaxy-search-type": enformion_client.config.search_type,
                    "galaxy-client-type": enformion_client.config.client_type,
                    "galaxy-client-session-id": f"debug_session_{int(datetime.now().timestamp())}"
                }
            ) as client:
                
                response = await client.post(
                    f"{enformion_client.config.base_url}/PersonSearch",
                    json=payload
                )
                
                return {
                    "request_payload": payload,
                    "response_status": response.status_code,
                    "response_headers": dict(response.headers),
                    "raw_response": response.text,
                    "parsed_json": response.json() if response.status_code == 200 else None,
                    "config": {
                        "base_url": enformion_client.config.base_url,
                        "search_type": enformion_client.config.search_type,
                        "ap_name": enformion_client.config.ap_name,
                        "ap_password": "***" # Don't expose password
                    }
                }
                
        except httpx.TimeoutException:
            return {"error": "Request timeout", "message": "EnformionGo API did not respond in time"}
        except Exception as e:
            return {"error": "API call failed", "message": str(e)}
        
    except Exception as e:
        import traceback
        return {
            "error": "Debug endpoint failed", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/enformion/config")
async def debug_enformion_config():
    """Check EnformionGo configuration"""
    import os
    from enformion_api import create_enformion_client
    
    client = create_enformion_client()
    
    return {
        "configured": client is not None,
        "environment_variables": {
            "ENFORMION_AP_NAME": "SET" if os.getenv("ENFORMION_AP_NAME") else "MISSING",
            "ENFORMION_AP_PASSWORD": "SET" if os.getenv("ENFORMION_AP_PASSWORD") else "MISSING",
            "ENFORMION_BASE_URL": os.getenv("ENFORMION_BASE_URL", "DEFAULT"),
            "ENFORMION_SEARCH_TYPE": os.getenv("ENFORMION_SEARCH_TYPE", "DEFAULT")
        },
        "config": {
            "base_url": client.config.base_url if client else None,
            "search_type": client.config.search_type if client else None,
            "max_results": client.config.max_results if client else None
        } if client else None
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)