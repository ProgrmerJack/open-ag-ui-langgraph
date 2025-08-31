// Import the HttpAgent for making HTTP requests to the backend
import { HttpAgent } from "@ag-ui/client";

// Import CopilotKit runtime components for setting up the API endpoint
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
  GoogleGenerativeAIAdapter,
  OpenAIAdapter,
} from "@copilotkit/runtime";

// Import NextRequest type for handling Next.js API requests
import { NextRequest } from "next/server";

// Determine provider and construct an appropriate adapter.
// Defaults to "auto": prefer OpenAI if configured; else Google (ADC-based);
// otherwise fall back to an EmptyAdapter and ensure the client disables chat.
let resolvedProvider = (
  process.env.COPILOT_PROVIDER ??
  process.env.NEXT_PUBLIC_COPILOT_PROVIDER ??
  "auto"
).toLowerCase(); // 'openai' | 'google' | 'none' | 'auto'

// Helper flags for configuration detection
const hasOpenAI = !!process.env.OPENAI_API_KEY;
const hasGoogleADC = !!process.env.GOOGLE_APPLICATION_CREDENTIALS || !!process.env.GOOGLE_AUTH_SCOPES || !!process.env.GOOGLE_CLIENT_EMAIL;

if (resolvedProvider === "auto") {
  if (hasOpenAI) {
    resolvedProvider = "openai";
  } else if (hasGoogleADC) {
    resolvedProvider = "google";
  } else {
    resolvedProvider = "none";
  }
}

let serviceAdapter;
switch (resolvedProvider) {
  case "openai":
    serviceAdapter = new OpenAIAdapter({
      apiKey: process.env.OPENAI_API_KEY,
      model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
    });
    break;
  case "google":
    // Note: Google adapter uses Google Auth (ADC). Ensure GOOGLE_APPLICATION_CREDENTIALS etc. are configured.
    serviceAdapter = new GoogleGenerativeAIAdapter({
      model: process.env.GEMINI_MODEL ?? "gemini-1.5-pro",
    });
    break;
  default:
    serviceAdapter = new ExperimentalEmptyAdapter();
}

if (process.env.NODE_ENV !== "production") {
  const advisory =
    resolvedProvider === "none"
      ? "UI chat/suggestions disabled (no LLM configured)."
      : resolvedProvider === "openai"
      ? "Using OpenAIAdapter."
      : "Using GoogleGenerativeAIAdapter (ADC required).";
  console.warn(`[copilotkit] Provider: ${resolvedProvider}. ${advisory}`);
}

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
  try {
    return await handleRequest(req);
  } catch (err) {
    if (process.env.NODE_ENV !== "production") {
      // Surface real cause in dev logs to diagnose CombinedError
      console.error("[copilotkit] API error:", err);
    }
    throw err;
  }
};
