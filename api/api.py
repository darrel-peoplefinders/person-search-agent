from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging

from api.main import (
    person_search_agent,
    create_initial_context,
)

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Person Search Engine", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
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
            "name": agent.name,
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
        search_results.append(SearchResult(
            id=result.get("id", ""),
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
    # Initialize or retrieve conversation state
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    
    if is_new:
        conversation_id: str = req.conversation_id or uuid4().hex
        ctx = create_initial_context()
        current_agent_name = person_search_agent.name
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
                context=ctx.model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
                search_results=None,
            )
    else:
        conversation_id = req.conversation_id
        state = conversation_store.get(conversation_id)
        print(f"Retrieved conversation {conversation_id} with context: {state['context'].model_dump() if state else 'NOT FOUND'}")

    current_agent = person_search_agent
    
    # Build input items for this turn
    input_items = state["input_items"] + [{"content": req.message, "role": "user"}]
    
    old_context = state["context"].model_dump().copy()
    guardrail_checks: List[GuardrailCheck] = []

    try:
        # Run agent with conversation history and context
        result = await Runner.run(
            current_agent, 
            input_items, 
            context=state["context"]
        )
    except InputGuardrailTripwireTriggered as e:
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        
        if "safety" in _get_guardrail_name(failed).lower():
            refusal = "I can only help with legitimate person search requests. Please ensure your search intent is appropriate."
        else:
            refusal = "I can only help you find people. How can I assist you in locating someone?"
            
        state["input_items"] = input_items + [{"role": "assistant", "content": refusal}]
        conversation_store.save(conversation_id, state)
        
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
            search_results=None,
        )

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    # Process all new items from the agent response
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(
                id=uuid4().hex, 
                type="message", 
                agent=item.agent.name, 
                content=text,
                timestamp=time.time() * 1000
            ))
        elif isinstance(item, ToolCallItem):
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    import json
                    tool_args = json.loads(raw_args)
                except Exception:
                    pass
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                    timestamp=time.time() * 1000,
                )
            )
        elif isinstance(item, ToolCallOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                    timestamp=time.time() * 1000,
                )
            )

    # Update conversation state with new context
    if hasattr(result, 'context_wrapper') and result.context_wrapper:
        state["context"] = result.context_wrapper.context
    
    # Track context changes
    new_context = state["context"].model_dump()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="Search criteria updated",
                metadata={"changes": changes},
                timestamp=time.time() * 1000,
            )
        )

    # Update conversation state
    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    conversation_store.save(conversation_id, state)
    
    print(f"Saved conversation {conversation_id} with updated context")

    # Build guardrail results
    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        if failed:
            final_guardrails.append(failed)
        else:
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=req.message,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))

    # Extract search results if available
    search_results = _extract_search_results(state["context"].model_dump())

    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].model_dump(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
        search_results=search_results,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)