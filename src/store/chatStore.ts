/**
 * License: GNU Affero General Public License v3.0 (AGPL-3.0)
 * This file is part of AgriSense.
 * 
 * TERMS OF USE:
 * This project is licensed under the AGPL-3.0. Private modifications or private use
 * without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
 * AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
 * Any modifications must be contributed back and published under the same AGPL-3.0 license.
 */

import { create } from "zustand";
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
