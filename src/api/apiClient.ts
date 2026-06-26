/**
 * AGRISENSE Service API Client
 * Wraps clean Fetch operations with pre/post interceptor queues.
 * Fully prepared for secure production-grade requests to the FastAPI backend.
 */

import { VITE_API_URL } from "../config/appConfig";

// Interceptor lists for pre and post execution pipelines
type RequestInterceptor = (config: RequestInit) => RequestInit | Promise<RequestInit>;
type ResponseInterceptor = (response: Response) => Response | Promise<Response>;

const requestInterceptors: RequestInterceptor[] = [];
const responseInterceptors: ResponseInterceptor[] = [];

// Inject default telemetry/content-type headers
requestInterceptors.push((config) => {
  const headers = new Headers(config.headers || {});
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  // Standard identifier header
  headers.set("X-Agrisense-Client", "React-19-Frontend");
  return { ...config, headers };
});

export const apiClient = {
  // Allow page components or services to register custom interceptors (e.g. bearer tokens, telemetry metrics)
  addRequestInterceptor(interceptor: RequestInterceptor) {
    requestInterceptors.push(interceptor);
  },
  
  addResponseInterceptor(interceptor: ResponseInterceptor) {
    responseInterceptors.push(interceptor);
  },

  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${VITE_API_URL}${endpoint}`;
    let config = { ...options };

    // Process Request Interceptors
    for (const interceptor of requestInterceptors) {
      config = await interceptor(config);
    }

    const response = await fetch(url, config);

    // Process Response Interceptors
    let processedResponse = response;
    for (const interceptor of responseInterceptors) {
      processedResponse = await interceptor(processedResponse);
    }

    if (!processedResponse.ok) {
      const errorText = await processedResponse.text().catch(() => "Unknown error");
      throw new Error(`API HTTP Error ${processedResponse.status}: ${errorText}`);
    }

    return processedResponse.json() as Promise<T>;
  },

  async get<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    return apiClient.request<T>(endpoint, { ...options, method: "GET" });
  },

  async post<T>(endpoint: string, data?: any, options: RequestInit = {}): Promise<T> {
    return apiClient.request<T>(endpoint, {
      ...options,
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
  }
};
