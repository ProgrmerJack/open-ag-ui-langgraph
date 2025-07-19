# Import necessary libraries and modules
from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming server-sent events
import uuid  # For generating unique message IDs
from typing import Any  # For type hints
import os  # For environment variables
import uvicorn  # ASGI server for running the FastAPI app
import asyncio  # For asynchronous programming

# Import AG UI core components for event-driven communication
from ag_ui.core import (
    RunAgentInput,        # Input data structure for agent runs
    StateSnapshotEvent,   # Event for capturing state snapshots
    EventType,            # Enumeration of all event types
    RunStartedEvent,      # Event to signal run start
    RunFinishedEvent,     # Event to signal run completion
    TextMessageStartEvent,    # Event to start text message streaming
    TextMessageEndEvent,      # Event to end text message streaming
    TextMessageContentEvent,  # Event for text message content chunks
    ToolCallStartEvent,       # Event to start tool call
    ToolCallEndEvent,         # Event to end tool call
    ToolCallArgsEvent,        # Event for tool call arguments
    StateDeltaEvent           # Event for state changes
)
from ag_ui.encoder import EventEncoder  # Encoder for converting events to SSE format
from stock_analysis import agent_graph  # Import the LangGraph agent
from copilotkit import CopilotKitState  # Base state class from CopilotKit

# Initialize FastAPI application instance
app = FastAPI()


class AgentState(CopilotKitState):
    """
    AgentState defines the structure of data that flows through the agent.
    It extends CopilotKitState and contains all the information needed
    for stock analysis and investment operations.
    """
    
    # List of available tools for the agent to use
    tools: list
    # Conversation history between user and assistant
    messages: list
    # Stock data retrieved from backend APIs
    be_stock_data: Any
    # Arguments passed to backend functions
    be_arguments: dict
    # Amount of cash available for investment
    available_cash: int
    # Summary of current investments
    investment_summary: dict
    # Log of tool executions and their results
    tool_logs: list

# FastAPI endpoint that handles agent execution requests
@app.post("/langgraph-agent")
async def langgraph_agent(input_data: RunAgentInput):
    """
    Main endpoint that processes agent requests and streams back events.
    
    Args:
        input_data (RunAgentInput): Contains thread_id, run_id, messages, tools, and state
        
    Returns:
        StreamingResponse: Server-sent events stream with agent execution updates
    """
    try:
        # Define async generator function to produce server-sent events
        async def event_generator():
            # Step 1: Initialize event encoding and communication infrastructure
            encoder = EventEncoder()  # Converts events to SSE format
            event_queue = asyncio.Queue()  # Queue for events from agent execution

            # Helper function to add events to the queue
            def emit_event(event):
                event_queue.put_nowait(event)

            # Generate unique identifier for this message thread
            message_id = str(uuid.uuid4())

            # Step 2: Signal the start of agent execution
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

            # Step 3: Send initial state snapshot to frontend
            yield encoder.encode(
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT, 
                    snapshot={
                        "available_cash": input_data.state["available_cash"],
                        "investment_summary": input_data.state["investment_summary"],
                        "investment_portfolio": input_data.state["investment_portfolio"],
                        "tool_logs": []
                    }
                )
            )
            
            # Step 4: Initialize agent state with input data
            state = AgentState(
                tools=input_data.tools,
                messages=input_data.messages,
                be_stock_data=None,  # Will be populated by agent tools
                be_arguments=None,   # Will be populated by agent tools
                available_cash=input_data.state["available_cash"],
                investment_portfolio=input_data.state["investment_portfolio"],
                tool_logs=[]
            )
            
            # Step 5: Create and configure the LangGraph agent
            agent = await agent_graph()

            # Step 6: Start agent execution asynchronously
            agent_task = asyncio.create_task(
                agent.ainvoke(
                    state, config={"emit_event": emit_event, "message_id": message_id}
                )
            )
            
            # Step 7: Stream events from agent execution as they occur
            while True:
                try:
                    # Wait for events with short timeout to check if agent is done
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield encoder.encode(event)
                except asyncio.TimeoutError:
                    # Check if the agent execution has completed
                    if agent_task.done():
                        break

            # Step 8: Clear tool logs after execution
            yield encoder.encode(
                StateDeltaEvent(
                    type=EventType.STATE_DELTA,
                    delta=[
                        {
                            "op": "replace",
                            "path": "/tool_logs",
                            "value": []
                        }
                    ]
                )
            )
            # Step 9: Handle the final message from the agent
            if state["messages"][-1].role == "assistant":
                # Check if the assistant made tool calls
                if state["messages"][-1].tool_calls:
                    # Step 9a: Stream tool call events if tools were used
                    
                    # Signal the start of tool execution
                    yield encoder.encode(
                        ToolCallStartEvent(
                            type=EventType.TOOL_CALL_START,
                            tool_call_id=state["messages"][-1].tool_calls[0].id,
                            toolCallName=state["messages"][-1]
                            .tool_calls[0]
                            .function.name,
                        )
                    )

                    # Send the tool call arguments
                    yield encoder.encode(
                        ToolCallArgsEvent(
                            type=EventType.TOOL_CALL_ARGS,
                            tool_call_id=state["messages"][-1].tool_calls[0].id,
                            delta=state["messages"][-1]
                            .tool_calls[0]
                            .function.arguments,
                        )
                    )

                    # Signal the end of tool execution
                    yield encoder.encode(
                        ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=state["messages"][-1].tool_calls[0].id,
                        )
                    )
                else:
                    # Step 9b: Stream text message if no tools were used
                    
                    # Signal the start of text message
                    yield encoder.encode(
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START,
                            message_id=message_id,
                            role="assistant",
                        )
                    )

                    # Stream the message content in chunks for better UX
                    if state["messages"][-1].content:
                        content = state["messages"][-1].content
                        
                        # Split content into 5 parts for gradual streaming
                        n_parts = 5
                        part_length = max(1, len(content) // n_parts)
                        parts = [content[i:i+part_length] for i in range(0, len(content), part_length)]
                        
                        # Handle rounding by merging extra parts into the last one
                        if len(parts) > n_parts:
                            parts = parts[:n_parts-1] + [''.join(parts[n_parts-1:])]
                        
                        # Stream each part with a delay for typing effect
                        for part in parts:
                            yield encoder.encode(
                                TextMessageContentEvent(
                                    type=EventType.TEXT_MESSAGE_CONTENT,
                                    message_id=message_id,
                                    delta=part,
                                )
                            )
                            await asyncio.sleep(0.5)  # 500ms delay between chunks
                    else:
                        # Send error message if content is empty
                        yield encoder.encode(
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta="Something went wrong! Please try again.",
                            )
                        )
                    
                    # Signal the end of text message
                    yield encoder.encode(
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id=message_id,
                        )
                    )

            # Step 10: Signal the completion of the entire agent run
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

    except Exception as e:
        # Log any errors that occur during execution
        print(e)

    # Return the event generator as a streaming response
    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main():
    """
    Main function to start the FastAPI server.
    
    This function:
    1. Gets the port from environment variable PORT (defaults to 8000)
    2. Starts the uvicorn ASGI server
    3. Enables hot reload for development
    """
    # Get port from environment variable, default to 8000
    port = int(os.getenv("PORT", "8000"))
    
    # Start the uvicorn server with the FastAPI app
    uvicorn.run(
        "main:app",        # Reference to the FastAPI app instance
        host="0.0.0.0",    # Listen on all available interfaces
        port=port,         # Use the configured port
        reload=True,       # Enable auto-reload for development
    )


# Entry point: run the server when script is executed directly
if __name__ == "__main__":
    main()
