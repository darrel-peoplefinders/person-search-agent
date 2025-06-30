from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging
import os

def check_environment():
    """Check critical environment variables"""
    required_vars = [
        "OPENAI_API_KEY",
        "ENFORMION_AP_NAME", 
        "ENFORMION_AP_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {missing_vars}")
        print("⚠️ Some features may not work properly")
    else:
        print("✅ All required environment variables are set")

# Call this at startup
check_environment()

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
        
        guardrail_checks: List[GuardrailCheck] = []

        try:
            # IMPORTANT: Ensure Runner.run is called with proper async context
            result = await Runner.run(
                current_agent, 
                input_items, 
                context=state["context"]
            )
        except Exception as e:
            error_type = str(type(e).__name__)
            logger.error(f"Error running agent ({error_type}): {e}")
            
            # Handle different types of errors
            if "InputGuardrailTripwireTriggered" in error_type:
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
                # Handle other errors with more detailed logging
                import traceback
                traceback.print_exc()
                
                error_msg = f"I'm having trouble processing your request right now. Please try again. (Error: {error_type})"
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
            try:
                if hasattr(item, 'agent') and hasattr(item.agent, 'name'):
                    agent_name = item.agent.name
                else:
                    agent_name = getattr(current_agent, "name", "person_search_agent")
                    
                item_type = str(type(item).__name__)
                
                if "MessageOutputItem" in item_type:
                    text = ItemHelpers.text_message_output(item) if hasattr(ItemHelpers, 'text_message_output') else str(item)
                    messages.append(MessageResponse(content=text, agent=agent_name))
                    events.append(AgentEvent(
                        id=uuid4().hex, 
                        type="message", 
                        agent=agent_name, 
                        content=text,
                        timestamp=time.time() * 1000
                    ))
                elif "ToolCallItem" in item_type:
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
                elif "ToolCallOutputItem" in item_type:
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
            except Exception as item_error:
                logger.error(f"Error processing item {item}: {item_error}")
                continue

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
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            conversation_id=req.conversation_id or uuid4().hex,
            current_agent="person_search_agent",
            messages=[MessageResponse(
                content=f"I'm experiencing technical difficulties. Please try again later. (Error: {str(e)})",
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

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify deployment"""
    return {
        "status": "OK",
        "message": "API is running",
        "environment": {
            "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "has_enformion_name": bool(os.getenv("ENFORMION_AP_NAME")),
            "has_enformion_password": bool(os.getenv("ENFORMION_AP_PASSWORD"))
        }
    }

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
    Debug endpoint to get raw EnformionGo API response with proper pagination
    
    Request body (supports both formats):
    {
        "first_name": "Jennifer",
        "last_name": "Andrus", 
        "city": "Lancaster",
        "state": "CA",
        "age": 45,
        "ResultsPerPage": 5,     # EnformionGo native format
        "Page": 1                # EnformionGo native format
    }
    
    OR legacy format:
    {
        "first_name": "Jennifer",
        "last_name": "Andrus", 
        "state": "CA",
        "limit": 5,              # Converted to ResultsPerPage
        "page": 1                # Converted to Page
    }
    """
    try:
        from enformion_api import create_enformion_client, PersonSearchParams
        
        # Extract search parameters - support both native and legacy formats
        first_name = request.get("first_name", "") or request.get("FirstName", "")
        last_name = request.get("last_name", "") or request.get("LastName", "")
        city = request.get("city") or request.get("City")
        state = request.get("state") or request.get("State")
        age = request.get("age") or request.get("Age")
        
        # Handle pagination parameters - prefer EnformionGo native format
        results_per_page = request.get("ResultsPerPage") or request.get("limit", 10)
        page = request.get("Page") or request.get("page", 1)
        
        # Ensure they're integers
        try:
            results_per_page = int(results_per_page)
            page = int(page)
        except (ValueError, TypeError):
            return {"error": "ResultsPerPage and Page must be integers"}
        
        # Validate required parameters
        if not first_name or not last_name:
            return {"error": "first_name and last_name (or FirstName and LastName) are required"}
        
        # Validate pagination parameters
        if results_per_page < 1 or results_per_page > 50:
            return {"error": "ResultsPerPage must be between 1 and 50"}
        
        if page < 1:
            return {"error": "Page must be >= 1"}
        
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
                "debug_info": {
                    "ENFORMION_AP_NAME": "SET" if os.getenv("ENFORMION_AP_NAME") else "MISSING",
                    "ENFORMION_AP_PASSWORD": "SET" if os.getenv("ENFORMION_AP_PASSWORD") else "MISSING"
                },
                "using_mock": False
            }
        
        # Build the exact EnformionGo API payload
        payload = {
            "FirstName": search_params.first_name,
            "LastName": search_params.last_name,
            "Page": page,
            "ResultsPerPage": results_per_page,
            "Includes": [
                "Addresses", 
                "PhoneNumbers", 
                "EmailAddresses", 
                "DatesOfBirth",
                "RelativesSummary",
                "AssociatesSummary"
            ]
        }
        
        # Add middle name if provided
        if hasattr(search_params, 'middle_name') and search_params.middle_name:
            payload["MiddleName"] = search_params.middle_name
        
        # Add location information
        if search_params.city or search_params.state:
            address_line2_parts = []
            
            if search_params.city:
                address_line2_parts.append(search_params.city)
            if search_params.state:
                address_line2_parts.append(search_params.state)
                
            if address_line2_parts:
                payload["Addresses"] = [{
                    "AddressLine2": ", ".join(address_line2_parts)
                }]
        
        # Add age if provided
        if search_params.age:
            try:
                payload["Age"] = int(search_params.age)
            except (ValueError, TypeError):
                return {"error": "Age must be a valid integer"}
        
        # Make the actual API call
        try:
            import httpx
            from datetime import datetime
            
            # Create session ID for debugging
            session_id = f"debug_session_{int(datetime.now().timestamp())}"
            
            async with httpx.AsyncClient(
                timeout=30,
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "galaxy-ap-name": enformion_client.config.ap_name,
                    "galaxy-ap-password": enformion_client.config.ap_password,
                    "galaxy-search-type": enformion_client.config.search_type,
                    "galaxy-client-type": enformion_client.config.client_type,
                    "galaxy-client-session-id": session_id
                }
            ) as client:
                
                # Log the request for debugging
                print(f"🔍 EnformionGo API Request:")
                print(f"   URL: {enformion_client.config.base_url}/PersonSearch")
                print(f"   Payload: {payload}")
                print(f"   Session ID: {session_id}")
                
                response = await client.post(
                    f"{enformion_client.config.base_url}/PersonSearch",
                    json=payload
                )
                
                print(f"🔍 EnformionGo API Response: {response.status_code}")
                
                # Build response data
                response_data = {
                    "debug_info": {
                        "request_payload_sent": payload,
                        "request_headers_sent": {
                            "galaxy-search-type": enformion_client.config.search_type,
                            "galaxy-client-type": enformion_client.config.client_type,
                            "galaxy-client-session-id": session_id
                        },
                        "api_endpoint": f"{enformion_client.config.base_url}/PersonSearch"
                    },
                    "response_status": response.status_code,
                    "response_headers": dict(response.headers),
                    "raw_response_text": response.text[:1000] + "..." if len(response.text) > 1000 else response.text,
                    "config": {
                        "base_url": enformion_client.config.base_url,
                        "search_type": enformion_client.config.search_type,
                        "client_type": enformion_client.config.client_type,
                        "ap_name": enformion_client.config.ap_name,
                        "ap_password": "***" # Don't expose password
                    }
                }
                
                # Try to parse JSON if successful
                if response.status_code == 200:
                    try:
                        parsed_json = response.json()
                        response_data["parsed_json"] = parsed_json
                        
                        # Add detailed summary info
                        if isinstance(parsed_json, dict):
                            # Check for persons array
                            if "persons" in parsed_json:
                                persons = parsed_json.get("persons", [])
                                response_data["summary"] = {
                                    "total_persons_returned": len(persons),
                                    "requested_results_per_page": results_per_page,
                                    "requested_page": page,
                                    "pagination_working": len(persons) <= results_per_page,
                                    "response_structure": "persons array found"
                                }
                                
                                # Show structure of first person if available
                                if persons and len(persons) > 0:
                                    first_person = persons[0]
                                    if isinstance(first_person, dict):
                                        response_data["first_person_sample"] = {
                                            "available_fields": sorted(list(first_person.keys())),
                                            "name_structure": first_person.get("name", "No name field"),
                                            "age": first_person.get("age", "No age field"),
                                            "addresses_count": len(first_person.get("addresses", [])),
                                            "phone_numbers_count": len(first_person.get("phoneNumbers", [])),
                                            "email_addresses_count": len(first_person.get("emailAddresses", []))
                                        }
                            else:
                                # Look for other possible structures
                                response_data["summary"] = {
                                    "response_structure": "No 'persons' array found",
                                    "top_level_keys": sorted(list(parsed_json.keys())),
                                    "requested_results_per_page": results_per_page,
                                    "requested_page": page
                                }
                        else:
                            response_data["summary"] = {
                                "response_structure": f"Unexpected response type: {type(parsed_json)}",
                                "requested_results_per_page": results_per_page,
                                "requested_page": page
                            }
                                
                    except Exception as parse_error:
                        response_data["json_parse_error"] = str(parse_error)
                        response_data["summary"] = {
                            "error": "Failed to parse JSON response",
                            "requested_results_per_page": results_per_page,
                            "requested_page": page
                        }
                elif response.status_code == 400:
                    response_data["error_details"] = {
                        "message": "Bad Request - Check your payload parameters",
                        "likely_cause": "Invalid search parameters or missing required fields"
                    }
                elif response.status_code == 401:
                    response_data["error_details"] = {
                        "message": "Unauthorized - Check your API credentials",
                        "likely_cause": "Invalid galaxy-ap-name or galaxy-ap-password"
                    }
                elif response.status_code == 403:
                    response_data["error_details"] = {
                        "message": "Forbidden - Check your API permissions",
                        "likely_cause": "Account doesn't have access to PersonSearch endpoint"
                    }
                else:
                    response_data["error_details"] = {
                        "message": f"HTTP {response.status_code} error",
                        "likely_cause": "Server error or network issue"
                    }
                
                return response_data
                
        except httpx.TimeoutException:
            return {
                "error": "Request timeout", 
                "message": "EnformionGo API did not respond within 30 seconds",
                "debug_info": {
                    "request_payload": payload,
                    "timeout_duration": 30
                }
            }
        except httpx.ConnectError:
            return {
                "error": "Connection failed", 
                "message": "Could not connect to EnformionGo API",
                "debug_info": {
                    "api_url": f"{enformion_client.config.base_url}/PersonSearch",
                    "request_payload": payload
                }
            }
        except Exception as e:
            import traceback
            return {
                "error": "API call failed", 
                "message": str(e),
                "debug_info": {
                    "request_payload": payload,
                    "full_traceback": traceback.format_exc()
                }
            }
        
    except Exception as e:
        import traceback
        return {
            "error": "Debug endpoint failed", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Use port 8080 for Cloud Run
