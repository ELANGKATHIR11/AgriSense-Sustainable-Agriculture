// AGRISENSE AgriGPT Conversation History Store
import create from "zustand";
import { ChatMessage } from "../types";
import { initialMockChatHistory } from "../mocks/mockChat";

interface ChatState {
  messages: ChatMessage[];
  loading: boolean;
  addMessage: (msg: Omit<ChatMessage, "id" | "timestamp">) => void;
  setLoading: (loading: boolean) => void;
  clearHistory: () => void;
}

export const useChatStore = create((set) => ({
  messages: initialMockChatHistory,
  loading: false,
  addMessage: (msg) => set((state) => ({
    messages: [
      ...state.messages,
      {
        ...msg,
        id: "msg-" + Math.random().toString(36).substring(4, 9),
        timestamp: new Date().toISOString()
      }
    ]
  })),
  setLoading: (loading) => set({ loading }),
  clearHistory: () => set({ messages: [initialMockChatHistory[0]] })
}));
