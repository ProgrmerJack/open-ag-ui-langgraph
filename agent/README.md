# Stock Analysis Agent with AG-UI Protocol

A FastAPI-based backend agent demonstrating the AG-UI (Agent User Interaction) protocol implementation with LangGraph workflow orchestration for stock analysis and investment simulation.

## Overview

This agent demonstrates how to build AI agents using the AG-UI protocol, featuring:

- **Event-driven communication** through AG-UI's streaming protocol
- **LangGraph workflow orchestration** with state management
- **Tool-based interactions** with real-time progress tracking
- **Structured state synchronization** between agent and frontend
- **Multi-step agent workflows** with intermediate feedback

### Core Capabilities

- Stock data extraction and analysis using Yahoo Finance
- Investment simulation with portfolio tracking
- AI-powered bull/bear insights generation using Google Gemini
- Performance comparison with SPY benchmark
- Real-time streaming updates via AG-UI events

## AG-UI Protocol Features

This implementation showcases key AG-UI protocol capabilities:

- **Event Streaming**: Real-time communication via Server-Sent Events (SSE)
- **State Management**: Synchronized state between agent and frontend using `STATE_SNAPSHOT` and `STATE_DELTA` events
- **Tool Integration**: Frontend-defined tools executed by the agent with progress tracking
- **Message History**: Persistent conversation context across interactions
- **Lifecycle Events**: Clear workflow progression indicators (`RUN_STARTED`, `RUN_FINISHED`)
- **Error Handling**: Robust error propagation and recovery mechanisms

### AG-UI Event Types Used

- **Lifecycle Events**: `RUN_STARTED`, `RUN_FINISHED` for agent execution boundaries
- **Text Message Events**: `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` for streaming responses
- **Tool Call Events**: `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END` for tool execution tracking
- **State Events**: `STATE_SNAPSHOT`, `STATE_DELTA` for real-time state synchronization

## Architecture

This agent demonstrates the AG-UI protocol architecture with:

### AG-UI Protocol Layer

- **FastAPI Server**: Implements AG-UI HTTP endpoint (`/langgraph-agent`)
- **Event Encoder**: Converts internal events to AG-UI SSE format
- **State Synchronization**: Real-time state updates between agent and frontend
- **Tool Protocol**: Standardized tool call/response handling

### LangGraph Integration

- **Workflow Orchestration**: Multi-node agent workflow with state transitions
- **State Management**: Persistent state across workflow nodes using `AgentState`
- **Node-based Processing**: Modular agent logic with clear separation of concerns
- **Command Routing**: Dynamic workflow navigation based on processing results

### Technology Stack

- **AG-UI Protocol**: Event-driven agent communication standard
- **LangGraph**: Workflow orchestration and state management
- **FastAPI**: Web server with SSE streaming support
- **Google Gemini 2.5 Pro**: Large language model for AI capabilities
- **Yahoo Finance API**: Real-time and historical stock market data
- **CopilotKit State**: Extended state management for UI integration

## Prerequisites

- Python 3.12
- Poetry (for dependency management)
- Google API Key for Gemini access

## Setup

### 1. Install Dependencies

```bash
poetry install
```

### 2. Environment Configuration

Create a `.env` file in the agent directory:

```env
# Required: Google API Key for Gemini 2.5 Pro
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Server port (default: 8000)
PORT=8000
```

Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Run the Agent

```bash
poetry run python main.py
```

The agent will start on `http://localhost:8000`.

## AG-UI Protocol Implementation

### HTTP Endpoint

The agent implements the AG-UI HTTP protocol via:

**POST `/langgraph-agent`**

Accepts `RunAgentInput` and returns a stream of AG-UI events.

**Request Body (RunAgentInput):**

```json
{
  "thread_id": "conversation-123",
  "run_id": "run-456",
  "messages": [
    {
      "id": "msg-1",
      "role": "user",
      "content": "Analyze AAPL and GOOGL with $10000 investment each"
    }
  ],
  "tools": [
    {
      "name": "extract_relevant_data_from_user_prompt",
      "description": "Gets ticker symbols, amounts, and investment parameters"
    }
  ],
  "state": {
    "available_cash": 50000,
    "investment_portfolio": [],
    "investment_summary": {}
  }
}
```

**Response:** Server-Sent Events stream containing AG-UI protocol events:

```
data: {"type": "RUN_STARTED", "thread_id": "conversation-123", "run_id": "run-456"}

data: {"type": "STATE_SNAPSHOT", "snapshot": {"available_cash": 50000, "tool_logs": []}}

data: {"type": "TOOL_CALL_START", "tool_call_id": "call-123", "toolCallName": "extract_relevant_data_from_user_prompt"}

data: {"type": "STATE_DELTA", "delta": [{"op": "add", "path": "/tool_logs/-", "value": {"message": "Extracting investment parameters", "status": "processing"}}]}

data: {"type": "RUN_FINISHED", "thread_id": "conversation-123", "run_id": "run-456"}
```

## LangGraph Workflow Implementation

The agent uses LangGraph to orchestrate a multi-step workflow with AG-UI event streaming:

### Workflow Nodes

1. **Chat Node** (`chat_node`)

   - Extracts investment parameters using LLM tool calls
   - Emits `TOOL_CALL_*` events for parameter extraction
   - Updates conversation state with structured data

2. **Simulation Node** (`simulation_node`)

   - Fetches historical stock data from Yahoo Finance
   - Emits `STATE_DELTA` events for portfolio updates
   - Validates and adjusts investment dates

3. **Cash Allocation Node** (`cash_allocation_node`)

   - Simulates investment strategies (single-shot vs DCA)
   - Calculates returns and portfolio performance
   - Compares against SPY benchmark

4. **Insights Node** (`insights_node`)

   - Generates AI-powered bull/bear analysis
   - Uses structured tool calls for insight generation
   - Streams final recommendations

5. **End Node** (`end_node`)
   - Workflow termination marker
   - Triggers final state cleanup

### State Flow and AG-UI Integration

```python
class AgentState(CopilotKitState):
    """AG-UI compatible state that flows through LangGraph nodes"""
    tools: list              # Available tools for the agent
    messages: list           # Conversation history (AG-UI format)
    be_stock_data: Any       # Yahoo Finance data
    be_arguments: dict       # Extracted investment parameters
    available_cash: int      # Current cash balance
    investment_summary: dict # Performance metrics
    investment_portfolio: list # Holdings
    tool_logs: list         # Real-time progress tracking
```

Each node in the workflow:

- Receives the current `AgentState`
- Emits AG-UI events via `emit_event()` callback
- Updates state for the next node
- Returns `Command` objects for workflow routing

## Key Components

### AG-UI State Management

The `AgentState` class extends `CopilotKitState` and provides:

- **Message Synchronization**: Maintains conversation history in AG-UI format
- **Real-time Updates**: Emits `STATE_DELTA` events for incremental changes
- **Tool Logging**: Tracks tool execution progress for UI feedback
- **Portfolio Tracking**: Manages investment data with live updates

```python
# Example state delta emission
config.get("configurable").get("emit_event")(
    StateDeltaEvent(
        type=EventType.STATE_DELTA,
        delta=[{
            "op": "replace",
            "path": "/available_cash",
            "value": new_cash_amount
        }]
    )
)
```

### AG-UI Tool Integration

Tools are defined with JSON schema and executed through the AG-UI protocol:

- **Frontend-Defined Tools**: Tools are passed from frontend to agent
- **Structured Execution**: Tool calls follow `START` → `ARGS` → `END` event sequence
- **Progress Tracking**: Each tool execution updates `tool_logs` with status
- **Type Safety**: Full TypeScript/Python type checking for tool parameters

#### Tool: `extract_relevant_data_from_user_prompt`

Extracts structured investment parameters from natural language:

- Ticker symbols array (e.g., `['AAPL', 'GOOGL']`)
- Investment amounts per stock
- Investment date and interval strategy
- Portfolio vs sandbox selection

#### Tool: `generate_insights`

Creates bull/bear case analysis with structured output:

- Positive insights with titles, descriptions, and emojis
- Negative insights with risk assessments
- AI-generated investment thesis for each stock

### Event-Driven Data Processing

The agent combines LangGraph workflow orchestration with AG-UI event streaming:

- **Async Processing**: Each node runs asynchronously with real-time updates
- **Progress Feedback**: Tool logs provide immediate user feedback
- **State Synchronization**: Frontend state stays synchronized via delta events
- **Error Handling**: Graceful error propagation through AG-UI events

#### Data Flow Example:

1. User submits investment query
2. `chat_node` emits `TOOL_CALL_START` event
3. LLM extracts parameters, emits `STATE_DELTA` with results
4. `simulation_node` fetches Yahoo Finance data
5. Portfolio state updated via `STATE_DELTA` events
6. Final insights streamed as `TEXT_MESSAGE_*` events

## AG-UI Protocol Configuration

### Event Encoder Setup

```python
from ag_ui.encoder import EventEncoder

encoder = EventEncoder()
yield encoder.encode(
    StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=initial_state
    )
)
```

### Runtime Configuration

The agent accepts configuration for AG-UI integration:

```python
config = {
    "configurable": {
        "emit_event": event_callback,  # AG-UI event emission
        "message_id": unique_id        # Message tracking
    }
}
```

### Investment Parameters

- **Ticker Symbols**: Stock symbols to analyze (e.g., AAPL, GOOGL)
- **Investment Amount**: Dollar amounts to invest per stock
- **Investment Date**: Starting date for analysis
- **Investment Interval**: `single_shot` or periodic intervals
- **Portfolio Type**: Current portfolio or sandbox

### Data Periods

- Automatically adjusts historical data periods based on investment date
- Limits lookback to 4 years maximum for performance
- Uses quarterly intervals for analysis

## Dependencies

Key dependencies for AG-UI and LangGraph integration:

- **ag-ui-core**: Core AG-UI protocol events and types
- **ag-ui-encoder**: Event encoding for SSE streaming
- **copilotkit**: Extended state management and CopilotKit integration
- **langgraph**: Workflow orchestration with state transitions
- **langchain**: LLM integration and tool calling
- **langchain-gemini**: Google Gemini model integration
- **fastapi**: HTTP server with SSE streaming capabilities
- **yfinance**: Yahoo Finance API for market data
- **pandas**: Data manipulation and analysis
- **uvicorn**: ASGI server for production deployment

### AG-UI Specific Imports

```python
from ag_ui.core import (
    RunAgentInput, StateSnapshotEvent, EventType,
    RunStartedEvent, RunFinishedEvent,
    TextMessageStartEvent, TextMessageContentEvent,
    ToolCallStartEvent, StateDeltaEvent
)
from ag_ui.encoder import EventEncoder
```

## Error Handling

The agent includes robust error handling for:

- Missing or invalid stock data
- Insufficient funds scenarios
- API rate limiting
- Network connectivity issues

## Development

### File Structure

```
agent/
├── main.py              # FastAPI server with AG-UI endpoint
├── stock_analysis.py    # LangGraph workflow and AG-UI integration
├── prompts.py          # System and insight prompts for LLM
├── pyproject.toml      # Dependencies including AG-UI packages
├── poetry.lock         # Locked dependencies
├── .env               # Environment variables
└── README.md          # This documentation
```

### Adding New Features

#### Adding LangGraph Nodes

1. Define new async node function in `stock_analysis.py`
2. Add AG-UI event emissions for progress tracking
3. Update `AgentState` schema if needed
4. Connect node to workflow graph with routing logic

#### Adding AG-UI Events

1. Import event types from `ag_ui.core`
2. Emit events via `config.get("configurable").get("emit_event")()`
3. Use appropriate event types (`STATE_DELTA`, `TOOL_CALL_*`, etc.)
4. Test event streaming in frontend integration

#### Adding Tools

1. Define tool schema with JSON parameters
2. Implement tool logic in workflow nodes
3. Emit `TOOL_CALL_*` events for execution tracking
4. Handle tool results and state updates

### Testing AG-UI Integration

Test the protocol implementation:

```bash
# Start the agent
poetry run python main.py

# Test with curl (example POST)
curl -X POST http://localhost:8000/langgraph-agent \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test", "run_id": "run1", "messages": [...], "tools": [...], "state": {...}}'
```

## Troubleshooting

### AG-UI Protocol Issues

1. **Event Streaming Problems**: Check SSE headers and event encoding format
2. **State Sync Issues**: Verify `STATE_DELTA` events use correct JSON Patch format
3. **Tool Call Failures**: Ensure tool schemas match frontend definitions
4. **Message Format Errors**: Validate AG-UI message structure and types

### LangGraph Issues

1. **Workflow Hangs**: Check node return values and Command routing
2. **State Corruption**: Verify state mutations are properly handled
3. **Node Errors**: Add error handling and recovery in each workflow node
4. **Memory Issues**: Monitor state size and data retention

### Common Issues

1. **Missing API Key**: Ensure `GOOGLE_API_KEY` is set in `.env`
2. **Stock Data Issues**: Yahoo Finance may have temporary outages or rate limits
3. **Event Encoding**: Verify AG-UI event types and encoding format
4. **State Management**: Check JSON Patch operations in `STATE_DELTA` events

### Debugging

Enable detailed logging for AG-UI and LangGraph:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints in workflow nodes
print(f"Current state: {state}")
print(f"Emitting event: {event}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement AG-UI protocol changes or LangGraph workflow improvements
4. Test with both unit tests and integration tests
5. Ensure event streaming works correctly with frontend
6. Submit a pull request with detailed description

### Best Practices

- Follow AG-UI protocol specifications for event types and formats
- Use proper error handling and recovery in LangGraph nodes
- Emit progress events for long-running operations
- Maintain state consistency across workflow transitions
- Document new tools and their AG-UI integration

## License

This project is part of the CopilotKit demo suite demonstrating AG-UI protocol implementation with LangGraph workflow orchestration.
