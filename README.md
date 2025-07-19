# Open AG UI Demo

A full-stack AI-powered stock analysis and portfolio management application that demonstrates the integration of CopilotKit with LangGraph for intelligent financial analysis. The project features a Next.js frontend with an interactive chat interface and a FastAPI backend powered by Google's Gemini AI for real-time stock analysis and investment recommendations.

## 🚀 Features

- **🤖 AI-Powered Stock Analysis**: Intelligent stock analysis using Google Gemini AI
- **📊 Interactive Charts**: Real-time portfolio performance visualization with Recharts
- **💬 Chat Interface**: Natural language conversation with AI investment advisor
- **📈 Portfolio Management**: Track investments, allocations, and performance metrics
- **🎯 Investment Insights**: Bull and bear market insights for informed decision-making
- **🔄 Real-time Updates**: Live portfolio tracking and state management
- **📱 Responsive Design**: Modern UI built with Next.js 15 and Tailwind CSS
- **🔧 Tool Integration**: Yahoo Finance API integration for real-time stock data
- **🌐 AG-UI Protocol**: Event-driven communication between frontend and LangGraph agent
- **📡 Real-time Streaming**: Server-sent events for live agent interactions

## 🛠 Tech Stack

### Frontend

- **Framework**: Next.js 15 with React 19
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts for data visualization
- **AI Integration**: CopilotKit React components
- **Icons**: Lucide React
- **Language**: TypeScript

### Backend

- **Framework**: FastAPI with Python 3.12
- **AI Engine**: LangChain with Google Gemini
- **Workflow**: LangGraph for agent orchestration
- **AG-UI Protocol**: Event-driven agent communication framework
- **Data**: Yahoo Finance (yfinance) for stock data
- **Search**: Tavily for web research
- **Data Processing**: Pandas for financial analysis
- **Environment**: Poetry for dependency management

## 📁 Project Structure

```
open-ag-ui-demo-langgraph/
├── frontend/                 # Next.js 15 React application
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/   # UI components
│   │   │   │   ├── chart-components/  # Chart visualizations
│   │   │   │   ├── cash-panel.tsx     # Cash management UI
│   │   │   │   ├── generative-canvas.tsx  # AI canvas
│   │   │   │   └── prompt-panel.tsx   # Chat interface
│   │   │   ├── api/
│   │   │   │   └── copilotkit/       # CopilotKit API routes
│   │   │   └── page.tsx              # Main application page
│   │   └── utils/
│   │       └── prompts.ts            # Frontend prompt templates
│   ├── package.json          # Node.js dependencies
│   └── next.config.ts        # Next.js configuration
└── agent/                    # FastAPI backend agent
    ├── main.py              # FastAPI application entry point
    ├── stock_analysis.py    # LangGraph agent implementation
    ├── prompts.py           # AI prompt templates
    ├── pyproject.toml       # Python dependencies
    └── README.md            # Backend documentation
```

---

## 🏗 AG-UI Architecture

This project demonstrates the **Agent User Interaction Protocol (AG-UI)**, which provides a standardized, event-driven communication framework between frontend applications and AI agents. The AG-UI protocol enables real-time, streaming interactions with LangGraph-powered agents.

### Core AG-UI Components

#### Event-Driven Communication

The AG-UI protocol uses a streaming event-based architecture where all communication between the frontend and LangGraph agent happens through typed events:

- **Lifecycle Events**: `RunStarted`, `RunFinished`, `StepStarted`, `StepFinished`
- **Text Message Events**: `TextMessageStart`, `TextMessageContent`, `TextMessageEnd`
- **Tool Call Events**: `ToolCallStart`, `ToolCallArgs`, `ToolCallEnd`
- **State Management Events**: `StateSnapshot`, `StateDelta`, `MessagesSnapshot`

#### LangGraph Agent Integration

The backend implements a LangGraph workflow that communicates through AG-UI events:

```python
# Agent state flows through LangGraph nodes
class AgentState(CopilotKitState):
    tools: list
    messages: list
    be_stock_data: Any
    available_cash: int
    # ... other state properties

# LangGraph workflow with AG-UI event emission
agent_graph = StateGraph(AgentState)
agent_graph.add_node("stock_analysis", stock_analysis_node)
agent_graph.add_node("portfolio_optimization", portfolio_node)
```

#### Real-time Streaming

- **Server-Sent Events (SSE)**: Enables real-time streaming of agent responses
- **Progressive Content Delivery**: Text and data stream incrementally as generated
- **Live State Updates**: Portfolio data and charts update in real-time
- **Tool Execution Visibility**: Users see AI actions as they happen

#### State Synchronization

AG-UI provides efficient state management between frontend and backend:

- **State Snapshots**: Complete state synchronization at key points
- **State Deltas**: Incremental updates using JSON Patch (RFC 6902)
- **Message History**: Conversation state maintained across interactions
- **Tool Results**: Bidirectional data flow for tool executions

### Implementation Details

#### Backend (FastAPI + LangGraph)

```python
@app.post("/run_agent")
async def run_agent_endpoint(input_data: RunAgentInput):
    """
    AG-UI compatible endpoint that:
    1. Receives RunAgentInput with tools and context
    2. Executes LangGraph workflow
    3. Streams AG-UI events via SSE
    """
    return StreamingResponse(
        run_agent_stream(input_data),
        media_type="text/event-stream"
    )
```

#### Frontend (CopilotKit + AG-UI)

```typescript
// CopilotKit integration with AG-UI events
const { agent } = useCoAgent({
  name: "stock_analysis_agent",
  initialState: portfolioState,
});

// Real-time state updates via AG-UI events
useCoAgentStateRender({
  name: "stock_analysis_agent",
  render: ({ state }) => <PortfolioVisualization data={state} />,
});
```

### Benefits of AG-UI Integration

1. **Standardized Protocol**: Consistent communication interface regardless of AI backend
2. **Real-time Interactions**: Streaming responses create responsive user experiences
3. **Tool Integration**: Seamless bidirectional tool execution between AI and frontend
4. **State Management**: Efficient synchronization of complex application state
5. **Extensibility**: Easy to add new agent capabilities and frontend features
6. **Debugging**: Event-driven architecture provides clear visibility into agent operations

---

## 🚀 Getting Started

### Prerequisites

Before running the application, ensure you have the following installed:

- **Node.js** (v18 or later)
- **pnpm** (recommended package manager for frontend)
- **Python** (3.12)
- **Poetry** (for Python dependency management)
- **Google Gemini API Key** (for AI functionality)

### 1. Environment Configuration

Create a `.env` file in each relevant directory with the required API keys.

#### Backend (`agent/.env`):

```env
GOOGLE_API_KEY=<<your-gemini-key-here>>
# Optional: Add other API keys for enhanced functionality
# TAVILY_API_KEY=<<your-tavily-key-here>>
```

#### Frontend (`frontend/.env`):

```env
GOOGLE_API_KEY=<<your-gemini-key-here>>
```

### 2. Start the Backend Agent

Navigate to the agent directory and install dependencies:

```bash
cd agent
poetry install
poetry run python main.py
```

The backend will start on `http://localhost:8000` with the following endpoints:

- `/run_agent` - Main agent execution endpoint
- `/docs` - FastAPI interactive documentation

### 3. Start the Frontend

In a new terminal, navigate to the frontend directory:

```bash
cd frontend
pnpm install
pnpm run dev
```

The frontend will be available at [http://localhost:3000](http://localhost:3000).

### 4. Using the Application

1. **Open your browser** to `http://localhost:3000`
2. **Set your investment budget** using the cash panel
3. **Start chatting** with the AI agent about stocks and investments
4. **View real-time charts** and portfolio performance
5. **Get AI insights** on bull and bear market conditions

---

## 📊 Core Components

### Frontend Components

- **`GenerativeCanvas`** - Main AI chat interface
- **`CashPanel`** - Investment budget management
- **`ComponentTree`** - UI component hierarchy viewer
- **`PromptPanel`** - Chat input and suggestions
- **`BarChart`** - Portfolio allocation visualization
- **`LineChart`** - Performance tracking over time
- **`AllocationTable`** - Detailed portfolio breakdown
- **`ToolLogs`** - AI agent action logging

### Backend Agent Features

- **Stock Data Retrieval** - Yahoo Finance integration
- **AI Analysis** - Google Gemini-powered insights
- **Portfolio Optimization** - Investment allocation recommendations
- **Market Research** - Web search capabilities via Tavily
- **State Management** - LangGraph workflow orchestration

---

## 🔧 API Endpoints

### Backend (FastAPI)

- `POST /run_agent` - Execute the stock analysis agent
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

### Frontend (Next.js API Routes)

- `POST /api/copilotkit` - CopilotKit integration endpoint

---

## 🎯 Usage Examples

### Stock Analysis Query

```
"Analyze AAPL stock and suggest whether I should invest $10,000"
```

### Portfolio Creation

```
"Create a diversified portfolio for $50,000 with tech and healthcare stocks"
```

### Market Insights

```
"What are the current market trends and risks I should be aware of?"
```

---

## 🔑 Environment Variables

| Variable         | Description                                | Required |
| ---------------- | ------------------------------------------ | -------- |
| `GOOGLE_API_KEY` | Google Gemini API key for AI functionality | Yes      |
| `TAVILY_API_KEY` | Tavily API key for web search (optional)   | No       |

---

## 🛠 Development

### Frontend Development

```bash
cd frontend
pnpm dev          # Start development server
pnpm build        # Build for production
pnpm start        # Start production server
pnpm lint         # Run ESLint
```

### Backend Development

```bash
cd agent
poetry install    # Install dependencies
poetry run python main.py  # Start development server
poetry run pytest # Run tests (if available)
```

---

## 🚀 Deployment

### Frontend (Vercel)

The frontend is optimized for Vercel deployment:

1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Backend (Self-hosted)

For production deployment:

1. Use a production ASGI server like Gunicorn with Uvicorn workers
2. Set up proper environment variable management
3. Configure CORS settings for your frontend domain

---

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Google Gemini API key is valid and has sufficient quota
2. **Backend Connection**: Verify the backend is running on port 8000 before starting frontend
3. **Dependencies**: Run `poetry install` and `pnpm install` to ensure all dependencies are installed
4. **Python Version**: Ensure you're using Python 3.12 as specified in pyproject.toml

### Debug Mode

Enable debug logging by setting environment variables:

```env
DEBUG=true
LOG_LEVEL=debug
```

---

## 📝 Notes

- **Backend Dependency**: Ensure the backend agent is running before using the frontend
- **API Keys**: Update environment variables as needed for your deployment
- **Performance**: The application fetches real-time stock data, so response times may vary
- **Rate Limits**: Be mindful of API rate limits for Google Gemini and Yahoo Finance
- **Data Accuracy**: Stock data is for demonstration purposes; consult financial advisors for real investments

---

## 🔗 Links

- **Live Demo**: [https://open-ag-ui-demo.vercel.app/](https://open-ag-ui-demo.vercel.app/)
- **AG-UI Documentation**: [https://docs.ag-ui.com/](https://docs.ag-ui.com/)
- **CopilotKit**: [https://copilotkit.ai/](https://copilotkit.ai/)
- **LangGraph**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **Google Gemini**: [https://ai.google.dev/](https://ai.google.dev/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Open an issue in the GitHub repository
3. Contact the CopilotKit team for framework-specific questions

---

**Built with ❤️ using CopilotKit, LangGraph, and modern web technologies**
