/**
 * AGRISENSE React Query Hooks - AgriGPT Advising Conversations
 */

import { useMutation } from "@tanstack/react-query";
import { chatService } from "../services/chatService";
import { ChatMessage } from "../types";
import { useChatStore } from "../store/chatStore";

export function useChat() {
  const { messages, addMessage, setLoading } = useChatStore();

  const chatMutation = useMutation({
    mutationFn: (history: ChatMessage[]) => chatService.sendMessage(history),
    onMutate: () => {
      setLoading(true);
    },
    onSuccess: (botAnswerText) => {
      addMessage({ role: "model", content: botAnswerText });
    },
    onSettled: () => {
      setLoading(false);
    }
  });

  const sendUserMessage = (text: string) => {
    if (!text.trim()) return;
    
    // 1. Append user message
    const userMsg: ChatMessage = {
      id: "u-" + Math.random().toString(36).substring(4, 9),
      role: "user",
      content: text,
      timestamp: new Date().toISOString()
    };
    
    addMessage({ role: "user", content: text });

    // 2. Transmit historical package
    const currentHistory = [...messages, userMsg];
    chatMutation.mutate(currentHistory);
  };

  return {
    messages,
    sendMessage: sendUserMessage,
    isPending: chatMutation.isPending
  };
}
