# \n# Import necessary libraries for LangChain, LangGraph, and financial analysis
from langchain_core.runnables import (
    RunnableConfig,
)  # Configuration for LangChain runnables
from ag_ui.core import (
    StateDeltaEvent,
    EventType,
)  # AG UI event system for state updates
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    ToolMessage,
    HumanMessage,
)  # Message types
from ag_ui.core import (
    AssistantMessage,
    ToolMessage as ToolMessageAGUI,
)  # AG UI message types
from langgraph.graph import StateGraph, START, END  # LangGraph workflow components
from langgraph.types import Command  # For controlling workflow flow
import yfinance as yf  # Yahoo Finance API for stock data

# Prefer CopilotKit but allow running without it in this environment
try:
    from copilotkit import CopilotKitState  # Base state class from CopilotKit
except Exception:

    class CopilotKitState(dict):  # minimal shim
        pass


from langchain.chat_models import init_chat_model  # Chat model initialization
from dotenv import load_dotenv  # Environment variable loader
import json  # JSON handling
import pandas as pd  # Data manipulation and analysis
import asyncio  # Asynchronous programming
from prompts import system_prompt, insights_prompt  # Custom prompts for the agent
from datetime import datetime  # Date and time handling
from typing import Any  # Type hints
import uuid  # Unique identifier generation

# Load environment variables (API keys, etc.)
load_dotenv()


class AgentState(CopilotKitState):
    """
    AgentState defines the complete state structure for the stock analysis agent.
    This state flows through all nodes in the LangGraph workflow and contains
    all the data needed for stock analysis, portfolio management, and UI updates.
    """

    # List of available tools that the agent can call
    tools: list
    # Conversation history between user and assistant
    messages: list
    # Stock price data retrieved from Yahoo Finance (pandas DataFrame)
    be_stock_data: Any
    # Parsed arguments from user input (ticker symbols, amounts, dates, etc.)
    be_arguments: dict
    # Current available cash in the user's wallet
    available_cash: int
    # Summary of investment results, returns, and portfolio performance
    investment_summary: dict
    # List of stocks and amounts in the current portfolio
    investment_portfolio: list
    # Log of tool executions with status updates for the UI
    tool_logs: list


def convert_tool_call(tc):
    """
    Convert LangChain tool call format to AG UI tool call format.

    Args:
        tc: Tool call object from LangChain

    Returns:
        dict: Tool call formatted for AG UI with id, type, and function details
    """
    return {
        "id": tc.get("id"),
        "type": "function",
        "function": {
            "name": tc.get("name"),
            "arguments": json.dumps(tc.get("args", {})),
        },
    }


def convert_tool_call_for_model(tc):
    """
    Convert AG UI tool call format back to LangChain model format.

    Args:
        tc: Tool call object from AG UI

    Returns:
        dict: Tool call formatted for LangChain models with parsed arguments
    """
    return {
        "id": tc.id,
        "name": tc.function.name,
        "args": json.loads(tc.function.arguments),
    }


# Tool definition for extracting investment parameters from user input
# This tool is used by the LLM to parse natural language requests into structured data
extract_relevant_data_from_user_prompt = {
    "name": "extract_relevant_data_from_user_prompt",
    "description": "Gets the data like ticker symbols, amount of dollars to be invested, interval of investment.",
    "parameters": {
        "type": "object",
        "properties": {
            # Array of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])
            "ticker_symbols": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "A stock ticker symbol, e.g. 'AAPL', 'GOOGL'.",
                },
                "description": "A list of stock ticker symbols, e.g. ['AAPL', 'GOOGL'].",
            },
            # Date when the investment should be made
            "investment_date": {
                "type": "string",
                "description": "The date of investment, e.g. '2023-01-01'.",
            },
            # Amount of money to invest in each stock (parallel to ticker_symbols)
            "amount_of_dollars_to_be_invested": {
                "type": "array",
                "items": {
                    "type": "number",
                    "description": "The amount of dollars to be invested, e.g. 10000.",
                },
                "description": "The amount of dollars to be invested, e.g. [10000, 20000, 30000].",
            },
            # Investment strategy: single purchase vs dollar-cost averaging
            "interval_of_investment": {
                "type": "string",
                "description": (
                    "The interval of investment, e.g. '1d', '5d', '1mo', '3mo', '6mo', '1y'1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '4y', '5y'. "
                    "If the user did not specify the interval, then assume it as 'single_shot'"
                ),
            },
            # Whether to add to real portfolio or simulate in sandbox
            "to_be_added_in_portfolio": {
                "type": "boolean",
                "description": "If user wants to add it in the current portfolio, then set it to true. If user wants to add it in the sandbox portfolio, then set it to false.",
            },
        },
        "required": [
            "ticker_symbols",
            "investment_date",
            "amount_of_dollars_to_be_invested",
            "to_be_added_in_portfolio",
        ],
    },
}


# Tool definition for generating investment insights (bull and bear cases)
# This tool creates positive and negative analysis for stocks or portfolios
generate_insights = {
    "name": "generate_insights",
    "description": "Generate positive (bull) and negative (bear) insights for a stock or portfolio.",
    "parameters": {
        "type": "object",
        "properties": {
            # Positive investment thesis and opportunities
            "bullInsights": {
                "type": "array",
                "description": "A list of positive insights (bull case) for the stock or portfolio.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short title for the positive insight.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the positive insight.",
                        },
                        "emoji": {
                            "type": "string",
                            "description": "Emoji representing the positive insight.",
                        },
                    },
                    "required": ["title", "description", "emoji"],
                },
            },
            # Negative investment thesis and risks
            "bearInsights": {
                "type": "array",
                "description": "A list of negative insights (bear case) for the stock or portfolio.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short title for the negative insight.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the negative insight.",
                        },
                        "emoji": {
                            "type": "string",
                            "description": "Emoji representing the negative insight.",
                        },
                    },
                    "required": ["title", "description", "emoji"],
                },
            },
        },
        "required": ["bullInsights", "bearInsights"],
    },
}


async def chat_node(state: AgentState, config: RunnableConfig):
    """
    First node in the workflow: Analyzes user input and extracts investment parameters.

    This function:
    1. Creates a tool log entry for UI feedback
    2. Initializes the chat model (Gemini 2.5 Pro)
    3. Converts state messages to LangChain format
    4. Uses the LLM to extract structured data from user input
    5. Handles retries if the model doesn't respond properly
    6. Updates the conversation state with the response

    Args:
        state: Current agent state containing messages and context
        config: Runtime configuration including event emitters
    """
    try:
        # Step 1: Create and emit a tool log entry for UI feedback
        tool_log_id = str(uuid.uuid4())
        state["tool_logs"].append(
            {
                "id": tool_log_id,
                "message": "Analyzing user query",
                "status": "processing",
            }
        )
        config.get("configurable").get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "add",
                        "path": "/tool_logs/-",
                        "value": {
                            "message": "Analyzing user query",
                            "status": "processing",
                            "id": tool_log_id,
                        },
                    }
                ],
            )
        )
        await asyncio.sleep(0)  # Yield control to allow UI updates

        # Step 2: Initialize the chat model (Gemini 2.5 Pro from Google)
        model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")

        # Step 3: Convert state messages to LangChain message format
        messages = []
        for message in state["messages"]:
            match message.role:
                case "user":
                    # Convert user messages to HumanMessage
                    messages.append(HumanMessage(content=message.content))
                case "system":
                    # Convert system messages and inject portfolio data
                    messages.append(
                        SystemMessage(
                            content=system_prompt.replace(
                                "{PORTFOLIO_DATA_PLACEHOLDER}",
                                json.dumps(state["investment_portfolio"]),
                            )
                        )
                    )
                case "assistant" | "ai":
                    # Convert assistant messages and handle tool calls
                    tool_calls_converted = [
                        convert_tool_call_for_model(tc)
                        for tc in message.tool_calls or []
                    ]
                    messages.append(
                        AIMessage(
                            invalid_tool_calls=[],
                            tool_calls=tool_calls_converted,
                            type="ai",
                            content=message.content or "",
                        )
                    )
                case "tool":
                    # Convert tool result messages
                    messages.append(
                        ToolMessage(
                            tool_call_id=message.tool_call_id, content=message.content
                        )
                    )
                case _:
                    raise ValueError(f"Unsupported message role: {message.role}")

        # Step 4: Attempt to get structured response from the model with retries
        retry_counter = 0
        while True:
            # Break after 3 failed attempts to prevent infinite loops
            if retry_counter > 3:
                print("retry_counter", retry_counter)
                break

            # Call the model with the data extraction tool
            response = await model.bind_tools(
                [extract_relevant_data_from_user_prompt]
            ).ainvoke(messages, config=config)

            # Step 5a: Handle successful tool call response
            if response.tool_calls:
                # Convert tool calls to AG UI format
                tool_calls = [convert_tool_call(tc) for tc in response.tool_calls]
                a_message = AssistantMessage(
                    role="assistant", tool_calls=tool_calls, id=response.id
                )
                state["messages"].append(a_message)

                # Update tool log status to completed
                index = len(state["tool_logs"]) - 1
                config.get("configurable").get("emit_event")(
                    StateDeltaEvent(
                        type=EventType.STATE_DELTA,
                        delta=[
                            {
                                "op": "replace",
                                "path": f"/tool_logs/{index}/status",
                                "value": "completed",
                            }
                        ],
                    )
                )
                await asyncio.sleep(0)
                return  # Success - exit the function

            # Step 5b: Handle empty response (retry needed)
            elif response.content == "" and response.tool_calls == []:
                retry_counter += 1

            # Step 5c: Handle text response (no tool call)
            else:
                a_message = AssistantMessage(
                    id=response.id, content=response.content, role="assistant"
                )
                state["messages"].append(a_message)

                # Update tool log status to completed
                index = len(state["tool_logs"]) - 1
                config.get("configurable").get("emit_event")(
                    StateDeltaEvent(
                        type=EventType.STATE_DELTA,
                        delta=[
                            {
                                "op": "replace",
                                "path": f"/tool_logs/{index}/status",
                                "value": "completed",
                            }
                        ],
                    )
                )
                await asyncio.sleep(0)
                return  # Success - exit the function

        # Step 6: Handle case where all retries failed
        print("hello")
        a_message = AssistantMessage(
            id=response.id, content=response.content, role="assistant"
        )
        state["messages"].append(a_message)

    except Exception as e:
        # Step 7: Handle any exceptions that occur during processing
        print(e)
        a_message = AssistantMessage(id=response.id, content="", role="assistant")
        state["messages"].append(a_message)
        return Command(
            goto="end",  # Skip to end node on error
        )

    # Step 8: Final cleanup - mark tool log as completed
    index = len(state["tool_logs"]) - 1
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)
    return


async def end_node(state: AgentState, config: RunnableConfig):
    """
    Terminal node in the workflow: Marks the completion of the agent execution.

    This is a simple placeholder node that signifies the end of the workflow.
    No processing is done here - it's just a marker for workflow completion.
    """
    print("inside end node")


async def simulation_node(state: AgentState, config: RunnableConfig):
    """
    Second node in the workflow: Fetches historical stock data for analysis.

    This function:
    1. Creates a tool log entry for UI feedback
    2. Extracts investment parameters from the previous tool call
    3. Updates the investment portfolio in the state
    4. Validates and adjusts the investment date if too far in the past
    5. Downloads historical stock data from Yahoo Finance
    6. Stores the data for use in subsequent nodes

    Args:
        state: Current agent state with extracted investment parameters
        config: Runtime configuration including event emitters

    Returns:
        Command: Directs workflow to the cash_allocation node
    """
    print("inside simulation node")

    # Step 1: Create and emit tool log entry for UI feedback
    tool_log_id = str(uuid.uuid4())
    state["tool_logs"].append(
        {"id": tool_log_id, "message": "Gathering stock data", "status": "processing"}
    )
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Gathering stock data",
                        "status": "processing",
                        "id": tool_log_id,
                    },
                }
            ],
        )
    )
    await asyncio.sleep(0)

    # Step 2: Extract investment parameters from the last assistant message
    arguments = json.loads(state["messages"][-1].tool_calls[0].function.arguments)
    print("arguments", arguments)

    # Step 3: Update the investment portfolio in the state
    # Create portfolio entries with ticker symbols and investment amounts
    state["investment_portfolio"] = json.dumps(
        [
            {
                "ticker": ticker,
                "amount": arguments["amount_of_dollars_to_be_invested"][index],
            }
            for index, ticker in enumerate(arguments["ticker_symbols"])
        ]
    )

    # Step 4: Emit state change event to update the UI
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": "/investment_portfolio",
                    "value": json.loads(state["investment_portfolio"]),
                }
            ],
        )
    )
    await asyncio.sleep(2)  # Brief delay for UI updates

    # Step 5: Prepare parameters for historical data download
    tickers = arguments["ticker_symbols"]
    investment_date = arguments["investment_date"]
    current_year = datetime.now().year

    # Step 6: Validate and adjust investment date if necessary
    # Limit historical data to maximum of 4 years to avoid API limitations
    if current_year - int(investment_date[:4]) > 4:
        print("investment date is more than 4 years ago")
        investment_date = f"{current_year - 4}-01-01"

    # Step 7: Determine the appropriate time period for data download
    if current_year - int(investment_date[:4]) == 0:
        history_period = "1y"  # Current year - get 1 year of data
    else:
        history_period = f"{current_year - int(investment_date[:4])}y"

    # Step 8: Download historical stock data from Yahoo Finance
    data = yf.download(
        tickers,  # List of ticker symbols
        period=history_period,  # Time period for historical data
        interval="3mo",  # Data interval (quarterly)
        start=investment_date,  # Start date
        end=datetime.today().strftime("%Y-%m-%d"),  # End date (today)
    )

    # Step 9: Store the closing prices and arguments in state for next nodes
    state["be_stock_data"] = data["Close"]  # Extract closing prices only
    state["be_arguments"] = arguments  # Store parsed arguments
    print(state["be_stock_data"])

    # Step 10: Update tool log status to completed
    index = len(state["tool_logs"]) - 1
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)

    # Step 11: Direct workflow to the cash allocation node
    return Command(goto="cash_allocation", update=state)


async def cash_allocation_node(state: AgentState, config: RunnableConfig):
    """
    Third node in the workflow: Performs investment simulation and cash allocation.

    This is the most complex node that handles:
    1. Investment simulation (single-shot vs dollar-cost averaging)
    2. Cash allocation and share purchasing logic
    3. Portfolio performance calculation vs SPY benchmark
    4. Investment logging and error handling
    5. UI data preparation for charts and tables

    Args:
        state: Current agent state with stock data and investment parameters
        config: Runtime configuration including event emitters

    Returns:
        Command: Directs workflow to the insights node
    """
    print("inside cash allocation node")

    # Step 1: Create and emit tool log entry for UI feedback
    tool_log_id = str(uuid.uuid4())
    state["tool_logs"].append(
        {"id": tool_log_id, "message": "Allocating cash", "status": "processing"}
    )
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Allocating cash",
                        "status": "processing",
                        "id": tool_log_id,
                    },
                }
            ],
        )
    )
    await asyncio.sleep(2)

    # Step 2: Import required libraries for numerical computations
    import numpy as np

    # Step 3: Extract data from state for investment simulation
    stock_data = state["be_stock_data"]  # DataFrame: index=date, columns=tickers
    args = state["be_arguments"]
    tickers = args["ticker_symbols"]
    amounts = args["amount_of_dollars_to_be_invested"]  # list, one per ticker
    interval = args.get("interval_of_investment", "single_shot")

    # Step 4: Initialize cash and portfolio tracking variables
    # Use state['available_cash'] as a single integer (total wallet cash)
    if "available_cash" in state and state["available_cash"] is not None:
        total_cash = state["available_cash"]
    else:
        total_cash = sum(amounts)  # Fallback to sum of investment amounts

    # Initialize holdings dictionary to track shares owned
    holdings = {ticker: 0.0 for ticker in tickers}
    investment_log = []  # Log of all investment transactions
    add_funds_needed = False  # Flag for insufficient funds
    add_funds_dates = []  # Dates where additional funds were needed

    # Step 5: Ensure stock data is sorted chronologically
    stock_data = stock_data.sort_index()

    # Step 6: Execute investment strategy based on interval type
    if interval == "single_shot":
        # Single-shot investment: Buy all shares at the first available date
        first_date = stock_data.index[0]
        row = stock_data.loc[first_date]

        # Process each ticker for single-shot investment
        for idx, ticker in enumerate(tickers):
            price = row[ticker]

            # Skip if no price data available
            if np.isnan(price):
                investment_log.append(
                    f"{first_date.date()}: No price data for {ticker}, could not invest."
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, amounts[idx])
                )
                continue

            # Get allocated amount for this specific ticker
            allocated = amounts[idx]

            # Check if we have enough cash and the allocation covers at least one share
            if total_cash >= allocated and allocated >= price:
                shares_to_buy = (
                    allocated // price
                )  # Calculate shares (no fractional shares)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    holdings[ticker] += shares_to_buy
                    total_cash -= cost
                    investment_log.append(
                        f"{first_date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                    )
                else:
                    investment_log.append(
                        f"{first_date.date()}: Not enough allocated cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}"
                    )
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(first_date.date()), ticker, price, allocated)
                    )
            else:
                investment_log.append(
                    f"{first_date.date()}: Not enough total cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}, Available: ${total_cash:.2f}"
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, total_cash)
                )
        # No further purchases on subsequent dates for single-shot strategy
    else:
        # Dollar-Cost Averaging (DCA) or other interval-based investment strategy
        for date, row in stock_data.iterrows():
            for i, ticker in enumerate(tickers):
                price = row[ticker]
                if np.isnan(price):
                    continue  # skip if price is NaN

                # Invest as much as possible for this ticker at this date
                if total_cash >= price:
                    shares_to_buy = total_cash // price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        holdings[ticker] += shares_to_buy
                        total_cash -= cost
                        investment_log.append(
                            f"{date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                        )
                else:
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(date.date()), ticker, price, total_cash)
                    )
                    investment_log.append(
                        f"{date.date()}: Not enough cash to buy {ticker} at ${price:.2f}. Available: ${total_cash:.2f}. Please add more funds."
                    )

    # Step 7: Calculate final portfolio value and performance metrics
    final_prices = stock_data.iloc[-1]  # Get the last available prices
    total_value = 0.0
    returns = {}
    total_invested_per_stock = {}
    percent_allocation_per_stock = {}
    percent_return_per_stock = {}
    total_invested = 0.0

    # Calculate how much was actually invested in each stock
    for idx, ticker in enumerate(tickers):
        if interval == "single_shot":
            # Only one purchase at first date for single-shot strategy
            first_date = stock_data.index[0]
            price = stock_data.loc[first_date][ticker]
            shares_bought = holdings[ticker]
            invested = shares_bought * price
        else:
            # Sum all purchases from the investment log for DCA strategy
            invested = 0.0
            for log in investment_log:
                if f"shares of {ticker}" in log and "Bought" in log:
                    # Extract cost from log string
                    try:
                        cost_str = log.split("(cost: $")[-1].split(")")[0]
                        invested += float(cost_str)
                    except Exception:
                        pass
        total_invested_per_stock[ticker] = invested
        total_invested += invested

    # Step 8: Calculate percentage allocations and returns for each stock
    for ticker in tickers:
        invested = total_invested_per_stock[ticker]
        holding_value = holdings[ticker] * final_prices[ticker]
        returns[ticker] = holding_value - invested  # Absolute return in dollars
        total_value += holding_value

        # Calculate percentage allocation (how much of total was invested in this stock)
        percent_allocation_per_stock[ticker] = (
            (invested / total_invested * 100) if total_invested > 0 else 0.0
        )

        # Calculate percentage return for this stock
        percent_return_per_stock[ticker] = (
            ((holding_value - invested) / invested * 100) if invested > 0 else 0.0
        )

    total_value += total_cash  # Add remaining cash to total portfolio value

    # Step 9: Store investment results in state for UI display
    state["investment_summary"] = {
        "holdings": holdings,  # Shares owned per ticker
        "final_prices": final_prices.to_dict(),  # Current stock prices
        "cash": total_cash,  # Remaining cash
        "returns": returns,  # Dollar returns per stock
        "total_value": total_value,  # Total portfolio value
        "investment_log": investment_log,  # Transaction history
        "add_funds_needed": add_funds_needed,  # Whether more funds needed
        "add_funds_dates": add_funds_dates,  # Dates funds were insufficient
        "total_invested_per_stock": total_invested_per_stock,  # Amount invested per stock
        "percent_allocation_per_stock": percent_allocation_per_stock,  # Allocation percentages
        "percent_return_per_stock": percent_return_per_stock,  # Return percentages
    }
    state["available_cash"] = total_cash  # Update available cash in state

    # Step 10: Portfolio vs SPY (S&P 500) benchmark comparison
    # Get SPY prices for the same date range to compare portfolio performance
    spy_ticker = "SPY"
    spy_prices = None
    try:
        spy_prices = yf.download(
            spy_ticker,
            period=f"{len(stock_data)//4}y" if len(stock_data) > 4 else "1y",
            interval="3mo",
            start=stock_data.index[0],
            end=stock_data.index[-1],
        )["Close"]
        # Align SPY prices to stock_data dates using forward fill
        spy_prices = spy_prices.reindex(stock_data.index, method="ffill")
    except Exception as e:
        print("Error fetching SPY data:", e)
        spy_prices = pd.Series([None] * len(stock_data), index=stock_data.index)

    # Step 11: Simulate investing the same total amount in SPY for comparison
    spy_shares = 0.0
    spy_cash = total_invested  # Use same total investment amount
    spy_invested = 0.0
    spy_investment_log = []

    if interval == "single_shot":
        # Single-shot SPY investment at first date
        first_date = stock_data.index[0]
        spy_price = spy_prices.loc[first_date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]
        if not pd.isna(spy_price):
            spy_shares = spy_cash // spy_price
            spy_invested = spy_shares * spy_price
            spy_cash -= spy_invested
            spy_investment_log.append(
                f"{first_date.date()}: Bought {spy_shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${spy_invested:.2f})"
            )
    else:
        # DCA strategy for SPY: invest equal portions at each date
        dca_amount = total_invested / len(stock_data)
        for date in stock_data.index:
            spy_price = spy_prices.loc[date]
            if isinstance(spy_price, pd.Series):
                spy_price = spy_price.iloc[0]
            if not pd.isna(spy_price):
                shares = dca_amount // spy_price
                cost = shares * spy_price
                spy_shares += shares
                spy_cash -= cost
                spy_invested += cost
                spy_investment_log.append(
                    f"{date.date()}: Bought {shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${cost:.2f})"
                )

    # Step 12: Build performance comparison data for charting
    # Create time series data comparing portfolio vs SPY performance
    performanceData = []
    running_holdings = holdings.copy()  # Snapshot of final holdings
    running_cash = total_cash  # Remaining cash

    for date in stock_data.index:
        # Calculate portfolio value at this historical date
        port_value = (
            sum(
                running_holdings[t] * stock_data.loc[date][t]
                for t in tickers
                if not pd.isna(stock_data.loc[date][t])
            )
            + running_cash
        )

        # Calculate SPY value at this historical date
        spy_price = spy_prices.loc[date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]
        spy_val = spy_shares * spy_price + spy_cash if not pd.isna(spy_price) else None

        # Add data point for this date
        performanceData.append(
            {
                "date": str(date.date()),
                "portfolio": float(port_value) if port_value is not None else None,
                "spy": float(spy_val) if spy_val is not None else None,
            }
        )

    # Step 13: Add performance comparison data to investment summary
    state["investment_summary"]["performanceData"] = performanceData

    # Step 14: Generate summary message for the user
    if add_funds_needed:
        msg = "Some investments could not be made due to insufficient funds. Please add more funds to your wallet.\n"
        for d, t, p, c in add_funds_dates:
            msg += (
                f"On {d}, not enough cash for {t}: price ${p:.2f}, available ${c:.2f}\n"
            )
    else:
        msg = "All investments were made successfully.\n"

    msg += f"\nFinal portfolio value: ${total_value:.2f}\n"
    msg += "Returns by ticker (percent and $):\n"
    for ticker in tickers:
        percent = percent_return_per_stock[ticker]
        abs_return = returns[ticker]
        msg += f"{ticker}: {percent:.2f}% (${abs_return:.2f})\n"

    # Step 15: Add tool result message to conversation
    state["messages"].append(
        ToolMessageAGUI(
            role="tool",
            id=str(uuid.uuid4()),
            content="The relevant details had been extracted",
            tool_call_id=state["messages"][-1].tool_calls[0].id,
        )
    )

    # Step 16: Add assistant message with chart rendering tool call
    state["messages"].append(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "render_standard_charts_and_table",
                        "arguments": json.dumps(
                            {"investment_summary": state["investment_summary"]}
                        ),
                    },
                }
            ],
            id=str(uuid.uuid4()),
        )
    )

    # Step 17: Update tool log status to completed
    index = len(state["tool_logs"]) - 1
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)

    # Step 18: Direct workflow to the insights generation node
    return Command(goto="ui_decision", update=state)


async def insights_node(state: AgentState, config: RunnableConfig):
    """
    Fourth node in the workflow: Generates investment insights using AI.

    This function:
    1. Creates a tool log entry for UI feedback
    2. Extracts ticker symbols from the investment arguments
    3. Uses Gemini model to generate bull and bear insights
    4. Integrates insights into the existing tool call arguments
    5. Updates the conversation state with the enhanced data

    Args:
        state: Current agent state with investment data and analysis
        config: Runtime configuration including event emitters

    Returns:
        Command: Directs workflow to the end node
    """
    print("inside insights node")

    # Step 1: Create and emit tool log entry for UI feedback
    tool_log_id = str(uuid.uuid4())
    state["tool_logs"].append(
        {
            "id": tool_log_id,
            "message": "Extracting key insights",
            "status": "processing",
        }
    )
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Extracting key insights",
                        "status": "processing",
                        "id": tool_log_id,
                    },
                }
            ],
        )
    )
    await asyncio.sleep(0)

    # Step 2: Extract ticker symbols from investment arguments
    args = state.get("be_arguments") or state.get("arguments")
    tickers = args.get("ticker_symbols", [])

    # Step 3: Initialize AI model and generate insights
    model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
    response = await model.bind_tools(generate_insights).ainvoke(
        [
            {"role": "system", "content": insights_prompt},
            {"role": "user", "content": tickers},  # Send ticker list to model
        ],
        config=config,
    )

    # Step 4: Process the insights response
    if response.tool_calls:
        # Step 4a: Extract current arguments from the last tool call
        args_dict = json.loads(state["messages"][-1].tool_calls[0].function.arguments)

        # Step 4b: Add the generated insights to the arguments
        args_dict["insights"] = response.tool_calls[0]["args"]

        # Step 4c: Update the tool call arguments with insights included
        state["messages"][-1].tool_calls[0].function.arguments = json.dumps(args_dict)
    else:
        # Step 4d: Handle case where no insights were generated
        state["insights"] = {}

    # Step 5: Update tool log status to completed
    index = len(state["tool_logs"]) - 1
    config.get("configurable").get("emit_event")(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)

    # Step 6: Direct workflow to the end node (completion)
    return Command(goto="end", update=state)


def router_function1(state: AgentState, config: RunnableConfig):
    """
    Router function that determines the next node in the workflow.

    This function examines the last message in the conversation to decide
    whether to proceed to the simulation node or end the workflow.

    Args:
        state: Current agent state with conversation messages
        config: Runtime configuration (unused in this router)

    Returns:
        str: Next node name ("end" or "simulation")
    """
    # Check if the last message has tool calls
    if (
        state["messages"][-1].tool_calls == []
        or state["messages"][-1].tool_calls is None
    ):
        # No tool calls means end the workflow (likely a text response)
        return "end"
    else:
        # Tool calls present means proceed to simulation
        return "simulation"


async def agent_graph():
    """
    Creates and configures the LangGraph workflow for stock analysis.

    This function:
    1. Creates a StateGraph with the AgentState structure
    2. Adds all workflow nodes (chat, simulation, cash_allocation, insights, end)
    3. Defines the workflow edges and conditional routing
    4. Sets entry and exit points
    5. Compiles the graph for execution

    Returns:
        CompiledStateGraph: The compiled workflow ready for execution
    """
    # Step 1: Create the workflow graph with AgentState structure
    workflow = StateGraph(AgentState)

    # Step 2: Add all nodes to the workflow
    workflow.add_node("chat", chat_node)  # Initial chat and parameter extraction
    workflow.add_node("simulation", simulation_node)  # Stock data gathering
    workflow.add_node(
        "cash_allocation", cash_allocation_node
    )  # Investment simulation and analysis
    workflow.add_node("insights", insights_node)  # AI-generated insights
    workflow.add_node("end", end_node)  # Terminal node

    # Step 3: Set workflow entry and exit points
    workflow.set_entry_point("chat")  # Always start with chat node
    workflow.set_finish_point("end")  # Always end with end node

    # Step 4: Define workflow edges and routing logic
    workflow.add_edge(START, "chat")  # Entry: START -> chat
    workflow.add_conditional_edges(
        "chat", router_function1
    )  # Conditional: chat -> (simulation|end)
    workflow.add_edge(
        "simulation", "cash_allocation"
    )  # Direct: simulation -> cash_allocation
    workflow.add_edge(
        "cash_allocation", "insights"
    )  # Direct: cash_allocation -> insights
    workflow.add_edge("insights", "end")  # Direct: insights -> end
    workflow.add_edge("end", END)  # Exit: end -> END

    # Step 5: Compile the workflow graph
    # Note: Memory persistence is commented out for simplicity
    # from langgraph.checkpoint.memory import MemorySaver
    # graph = workflow.compile(MemorySaver())
    graph = workflow.compile()

    return graph
