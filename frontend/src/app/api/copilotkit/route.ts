// Import the HttpAgent for making HTTP requests to the backend
import { HttpAgent } from "@ag-ui/client";

// Import CopilotKit runtime components for setting up the API endpoint
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
  GoogleGenerativeAIAdapter,
} from "@copilotkit/runtime";

// Import NextRequest type for handling Next.js API requests
import { NextRequest } from "next/server";

const serviceAdapter = new GoogleGenerativeAIAdapter();

// Create a new HttpAgent instance that connects to the LangGraph stock backend running locally
const stockAgent = new HttpAgent({
  url: "http://127.0.0.1:8000/langgraph-agent",
});

// Initialize the CopilotKit runtime with our stock agent
const runtime = new CopilotRuntime({
  agents: {
    stockAgent, // Register the stock agent with the runtime
  },
});

/**
 * Define the POST handler for the API endpoint
 * This function handles incoming POST requests to the /api/copilotkit endpoint
 */
export const POST = async (req: NextRequest) => {
  // Configure the CopilotKit endpoint for the Next.js app router
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime, // Use the runtime with our research agent
    serviceAdapter,
    endpoint: "/api/copilotkit", // Define the API endpoint path
  });

  // Process the incoming request with the CopilotKit handler
  return handleRequest(req);
};
