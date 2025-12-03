import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sql?: string | null;
  data?: any;
  executionTime?: number | null;
  confidence?: number | null;
  provider?: string | null;
  isError?: boolean;
  showApiKeySuggestion?: boolean;
  exampleQuestions?: string[];
}

export interface ChatConversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

// Serialized versions for localStorage
interface SerializedChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string; // ISO string
  sql?: string | null;
  data?: any;
  executionTime?: number | null;
  confidence?: number | null;
  provider?: string | null;
  isError?: boolean;
  showApiKeySuggestion?: boolean;
  exampleQuestions?: string[];
}

interface SerializedChatConversation {
  id: string;
  title: string;
  messages: SerializedChatMessage[];
  createdAt: string; // ISO string
  updatedAt: string; // ISO string
}

// Convert serialized data back to Date objects
const deserializeConversation = (conv: SerializedChatConversation | ChatConversation): ChatConversation => {
  // If already deserialized (Date objects), return as-is
  if (conv.createdAt instanceof Date && conv.updatedAt instanceof Date) {
    // Still need to check messages
    const chatConv = conv as ChatConversation;
    return {
      ...chatConv,
      messages: chatConv.messages.map((msg) => ({
        ...msg,
        timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp as string),
      })),
    };
  }
  
  // Otherwise, deserialize from ISO strings
  const serialized = conv as SerializedChatConversation;
  return {
    ...serialized,
    createdAt: new Date(serialized.createdAt),
    updatedAt: new Date(serialized.updatedAt),
    messages: serialized.messages.map((msg) => ({
      ...msg,
      timestamp: new Date(msg.timestamp),
    })),
  };
};

// Convert Date objects to ISO strings for storage
const serializeConversation = (conv: ChatConversation): SerializedChatConversation => ({
  ...conv,
  createdAt: conv.createdAt.toISOString(),
  updatedAt: conv.updatedAt.toISOString(),
  messages: conv.messages.map((msg) => ({
    ...msg,
    timestamp: msg.timestamp.toISOString(),
  })),
});

interface ChatStore {
  conversations: ChatConversation[];
  activeConversationId: string | null;
  searchQuery: string;
  loading: boolean;
  messages: ChatMessage[]; // Backward compatibility - returns current conversation messages
  createConversation: () => string;
  deleteConversation: (id: string) => void;
  setActiveConversation: (id: string) => void;
  setSearchQuery: (query: string) => void;
  addMessage: (message: ChatMessage, conversationId?: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  setLoading: (loading: boolean) => void;
  clearMessages: () => void;
  getCurrentMessages: () => ChatMessage[];
  getFilteredConversations: () => ChatConversation[];
}

const generateId = () => `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

interface PersistedState {
  conversations: SerializedChatConversation[];
  activeConversationId: string | null;
  searchQuery: string;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,
      searchQuery: '',
      loading: false,
      messages: [], // Backward compatibility

  createConversation: () => {
    const id = generateId();
    const now = new Date();
    const newConversation: ChatConversation = {
      id,
      title: 'New Conversation',
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    set((state) => ({
      conversations: [newConversation, ...state.conversations],
      activeConversationId: id,
      messages: [], // Update backward compatibility messages
    }));
    return id;
  },

  deleteConversation: (id: string) => {
    set((state) => {
      const newConversations = state.conversations.filter((conv) => conv.id !== id);
      const newActiveId =
        state.activeConversationId === id
          ? newConversations.length > 0
            ? newConversations[0].id
            : null
          : state.activeConversationId;
      const activeConv = newConversations.find((c) => c.id === newActiveId);
      return {
        conversations: newConversations,
        activeConversationId: newActiveId,
        messages: activeConv?.messages || [], // Update backward compatibility messages
      };
    });
  },

  setActiveConversation: (id: string) => {
    set((state) => {
      const activeConv = state.conversations.find((c) => c.id === id);
      return {
        activeConversationId: id,
        messages: activeConv?.messages || [], // Update backward compatibility messages
      };
    });
  },

  setSearchQuery: (query: string) => {
    set({ searchQuery: query });
  },

  addMessage: (message: ChatMessage, conversationId?: string) => {
    const state = get();
    const targetId = conversationId || state.activeConversationId;

    if (!targetId) {
      // If no active conversation, create one
      const newId = state.createConversation();
      set((s) => {
        const conv = s.conversations.find((c) => c.id === newId);
        if (conv) {
          conv.messages.push(message);
          conv.updatedAt = new Date();
          // Update title from first user message
          if (message.role === 'user' && conv.title === 'New Conversation') {
            conv.title = message.content.substring(0, 50) || 'New Conversation';
          }
        }
        return {
          conversations: [...s.conversations],
          messages: conv?.messages || [], // Update backward compatibility messages
        };
      });
      return;
    }

    set((s) => {
      const conv = s.conversations.find((c) => c.id === targetId);
      if (conv) {
        conv.messages.push(message);
        conv.updatedAt = new Date();
        // Update title from first user message
        if (message.role === 'user' && conv.title === 'New Conversation') {
          conv.title = message.content.substring(0, 50) || 'New Conversation';
        }
      }
      return {
        conversations: [...s.conversations],
        messages: conv?.messages || [], // Update backward compatibility messages
      };
    });
  },

  setMessages: (messages: ChatMessage[]) => {
    const state = get();
    const activeId = state.activeConversationId;
    if (activeId) {
      set((s) => {
        const conv = s.conversations.find((c) => c.id === activeId);
        if (conv) {
          conv.messages = messages;
          conv.updatedAt = new Date();
        }
        return {
          conversations: [...s.conversations],
          messages, // Update backward compatibility messages
        };
      });
    } else {
      set({ messages }); // Update backward compatibility messages only
    }
  },

  setLoading: (loading: boolean) => {
    set({ loading });
  },

  clearMessages: () => {
    const state = get();
    const activeId = state.activeConversationId;
    if (activeId) {
      set((s) => {
        const conv = s.conversations.find((c) => c.id === activeId);
        if (conv) {
          conv.messages = [];
          conv.updatedAt = new Date();
        }
        return {
          conversations: [...s.conversations],
          messages: [], // Update backward compatibility messages
        };
      });
    } else {
      set({ messages: [] }); // Update backward compatibility messages only
    }
  },

  getCurrentMessages: () => {
    const state = get();
    if (state.activeConversationId) {
      const conv = state.conversations.find((c) => c.id === state.activeConversationId);
      return conv?.messages || [];
    }
    return [];
  },

  getFilteredConversations: () => {
    const state = get();
    if (!state.searchQuery.trim()) {
      return state.conversations;
    }
    const query = state.searchQuery.toLowerCase();
    return state.conversations.filter(
      (conv) =>
        conv.title.toLowerCase().includes(query) ||
        conv.messages.some((msg) => msg.content.toLowerCase().includes(query))
    );
  },
    }),
    {
      name: 'chat-store', // localStorage key
      storage: createJSONStorage(() => localStorage),
      // Serialize: Convert Date objects to ISO strings
      partialize: (state) => ({
        conversations: state.conversations.map(serializeConversation),
        activeConversationId: state.activeConversationId,
        searchQuery: state.searchQuery,
      }),
      // Deserialize: Convert ISO strings back to Date objects
      onRehydrateStorage: () => (state) => {
        if (state) {
          // Deserialize conversations (convert ISO strings to Date objects)
          state.conversations = (state.conversations as any[]).map((conv: any) => {
            // Handle both serialized and already-deserialized formats
            if (conv.createdAt instanceof Date && conv.updatedAt instanceof Date) {
              // Already deserialized, just ensure messages are correct
              return {
                ...conv,
                messages: conv.messages.map((msg: any) => ({
                  ...msg,
                  timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp),
                })),
              };
            }
            // Deserialize from ISO strings
            return {
              ...conv,
              createdAt: new Date(conv.createdAt),
              updatedAt: new Date(conv.updatedAt),
              messages: conv.messages.map((msg: any) => ({
                ...msg,
                timestamp: new Date(msg.timestamp),
              })),
            };
          });
          
          // Update backward compatibility messages
          if (state.activeConversationId) {
            const activeConv = state.conversations.find(
              (c) => c.id === state.activeConversationId
            );
            state.messages = activeConv?.messages || [];
          } else {
            state.messages = [];
          }
        }
      },
    }
  )
);
