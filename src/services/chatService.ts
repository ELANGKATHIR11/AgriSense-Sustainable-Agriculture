/**
 * AGRISENSE AgriGPT Chat Bot Copilot Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { ChatMessage } from "../types";

export const chatService = {
  async sendMessage(history: ChatMessage[]): Promise<string> {
    interface ChatResponse {
      text: string;
    }
    
    const response = await apiClient.post<ChatResponse>(API_ENDPOINTS.chat, {
      messages: history.map(m => ({ role: m.role, content: m.content }))
    });
    return response.text;
  }
};
