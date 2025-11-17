/**
 * AI Analyst Page
 * 
 * Split-screen chat interface with conversation history sidebar.
 * Features:
 * - 25% left sidebar with chat history
 * - 75% right area for active chat
 * - Message bubbles with proper styling
 * - Loading states with animated dots
 * - Suggested questions when empty
 * - Multi-line input support
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  PlusOutlined,
  SearchOutlined,
  DeleteOutlined,
  SendOutlined,
  RobotOutlined,
  ExclamationCircleOutlined,
  SettingOutlined,
  CloseOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import {
  Input,
  Button,
  Spin,
  message as antdMessage,
} from 'antd';
import EmptyState from '@/components/EmptyState';
import { getDataStatistics } from '@/services/dataService';
import { chatService, ChatResponse } from '@/services/api';
import { apiKeyService } from '@/services/apiKeyService';
import {
  useChatStore,
  ChatMessage,
  ChatConversation,
} from '@/store/chatStore';
import {
  SPACING,
  COLORS,
  BORDER_RADIUS,
  SHADOWS,
} from '@/styles/designTokens';
import { AxiosError } from 'axios';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

dayjs.extend(relativeTime);

const { TextArea } = Input;

// Suggested questions when chat is empty
const SUGGESTED_QUESTIONS = [
  'What was my total revenue last month?',
  'Show me top 10 products by sales',
  'Which items have high refund rates?',
  'What is my net revenue?',
  'Show revenue by city',
  'Which products are declining?',
];

// Group conversations by date
const groupConversationsByDate = (conversations: ChatConversation[]) => {
  const today = dayjs().startOf('day');
  const yesterday = dayjs().subtract(1, 'day').startOf('day');
  const last7Days = dayjs().subtract(7, 'days').startOf('day');

  const groups: {
    today: ChatConversation[];
    yesterday: ChatConversation[];
    last7Days: ChatConversation[];
    older: ChatConversation[];
  } = {
    today: [],
    yesterday: [],
    last7Days: [],
    older: [],
  };

  conversations.forEach((conv) => {
    const updatedAt = dayjs(conv.updatedAt);
    if (updatedAt.isSame(today, 'day')) {
      groups.today.push(conv);
    } else if (updatedAt.isSame(yesterday, 'day')) {
      groups.yesterday.push(conv);
    } else if (updatedAt.isAfter(last7Days)) {
      groups.last7Days.push(conv);
    } else {
      groups.older.push(conv);
    }
  });

  return groups;
};

const AIAnalyst: React.FC = () => {
  const navigate = useNavigate();
  const {
    conversations,
    activeConversationId,
    loading,
    searchQuery,
    createConversation,
    deleteConversation,
    setActiveConversation,
    addMessage,
    setLoading,
    setSearchQuery,
    getCurrentMessages,
    getFilteredConversations,
  } = useChatStore();

  const [inputValue, setInputValue] = useState('');
  const [hasData, setHasData] = useState<boolean | null>(null);
  const [checkingData, setCheckingData] = useState(true);
  const [hoveredConversationId, setHoveredConversationId] = useState<string | null>(null);
  const [lowConfidenceCount, setLowConfidenceCount] = useState(0);
  const [showBanner, setShowBanner] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const [hasOpenAIKey, setHasOpenAIKey] = useState<boolean | null>(null);
  const [hasAnthropicKey, setHasAnthropicKey] = useState<boolean | null>(null);
  const [apiKeyStatus, setApiKeyStatus] = useState<{
    has_key: boolean;
    masked_key: string | null;
    is_valid: boolean;
  }>({
    has_key: false,
    masked_key: null,
    is_valid: false,
  });
  const chatEndRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const elapsedTimeRef = useRef<NodeJS.Timeout | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  // Timeout duration: 25 seconds
  const TIMEOUT_DURATION = 25000;

  const currentMessages = getCurrentMessages();
  const filteredConversations = getFilteredConversations();
  const conversationGroups = groupConversationsByDate(filteredConversations);

  // Scroll to bottom when new message is added
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentMessages, loading]);

  // Check if data exists
  const checkDataExists = async () => {
    setCheckingData(true);
    try {
      const stats = await getDataStatistics();
      if (stats && stats.total_records > 0) {
        setHasData(true);
      } else {
        setHasData(false);
      }
    } catch (error) {
      console.error('Error checking data existence:', error);
      setHasData(false);
    } finally {
      setCheckingData(false);
    }
  };

  useEffect(() => {
    checkDataExists();
    checkAPIKeys();
  }, []);

  // Check API key status
  const checkAPIKeys = async (): Promise<{ hasOpenAI: boolean; hasAnthropic: boolean }> => {
    try {
      const userId = localStorage.getItem('userId') || 'user-123';
      const [openaiKey, anthropicKey] = await Promise.all([
        apiKeyService.getAPIKey('openai', userId).catch(() => null),
        apiKeyService.getAPIKey('anthropic', userId).catch(() => null),
      ]);
      const hasOpenAI = openaiKey !== null;
      const hasAnthropic = anthropicKey !== null;
      setHasOpenAIKey(hasOpenAI);
      setHasAnthropicKey(hasAnthropic);
      
      // Update API key status for indicator (prioritize OpenAI)
      if (openaiKey) {
        setApiKeyStatus({
          has_key: true,
          masked_key: openaiKey.masked_key,
          is_valid: true, // Assume valid if key exists (backend validates on save)
        });
      } else if (anthropicKey) {
        setApiKeyStatus({
          has_key: true,
          masked_key: anthropicKey.masked_key,
          is_valid: true,
        });
      } else {
        setApiKeyStatus({
          has_key: false,
          masked_key: null,
          is_valid: false,
        });
      }
      
      return { hasOpenAI, hasAnthropic };
    } catch (error) {
      console.error('Error checking API keys:', error);
      setHasOpenAIKey(false);
      setHasAnthropicKey(false);
      setApiKeyStatus({
        has_key: false,
        masked_key: null,
        is_valid: false,
      });
      return { hasOpenAI: false, hasAnthropic: false };
    }
  };

  // Listen for data upload events
  useEffect(() => {
    const handleDataUpload = () => {
      checkDataExists();
    };
    window.addEventListener('dataUploaded', handleDataUpload);
    return () => {
      window.removeEventListener('dataUploaded', handleDataUpload);
    };
  }, []);

  // Create new conversation if none exists
  useEffect(() => {
    if (!activeConversationId && conversations.length === 0 && hasData) {
      createConversation();
    }
  }, [hasData, activeConversationId, conversations.length, createConversation]);

  // Handle new chat button
  const handleNewChat = () => {
    const newId = createConversation();
    setActiveConversation(newId);
    setInputValue('');
  };

  // Handle conversation click
  const handleConversationClick = (id: string) => {
    setActiveConversation(id);
  };

  // Handle conversation delete
  const handleDeleteConversation = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    deleteConversation(id);
    antdMessage.success('Conversation deleted');
  };

  // Add error message to chat
  const addErrorMessage = useCallback((
    conversationId: string,
    errorText: string,
    exampleQuestions?: string[]
  ) => {
    const errorMessage: ChatMessage = {
      id: `error-${Date.now()}`,
      role: 'assistant',
      content: errorText,
      timestamp: new Date(),
      isError: true,
      exampleQuestions,
    };
    addMessage(conversationId, errorMessage);
  }, [addMessage]);

  // Handle chat question submission
  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim() || loading || !activeConversationId) return;

    const userQuestion = inputValue.trim();
    setInputValue('');

    // Ensure we have an active conversation
    let conversationId = activeConversationId;
    if (!conversationId) {
      conversationId = createConversation();
    }

    // Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userQuestion,
      timestamp: new Date(),
    };
    addMessage(conversationId, userMessage);

    // Check if data exists before sending
    try {
      const stats = await getDataStatistics();
      if (!stats || stats.total_records === 0) {
        addErrorMessage(
          conversationId,
          "Please upload data first before asking questions. Click the 'Upload Data' button to get started.",
          ['Upload Data']
        );
        return;
      }
    } catch (error) {
      addErrorMessage(
        conversationId,
        "Unable to verify data. Please upload data first before asking questions.",
        ['Upload Data']
      );
      return;
    }

    // Set loading state and reset elapsed time
    setLoading(true);
    setElapsedTime(0);
    
    // Start elapsed time counter
    elapsedTimeRef.current = setInterval(() => {
      setElapsedTime((prev) => {
        const newTime = prev + 1;
        // Clear interval if we've exceeded timeout
        if (newTime * 1000 >= TIMEOUT_DURATION) {
          if (elapsedTimeRef.current) {
            clearInterval(elapsedTimeRef.current);
            elapsedTimeRef.current = null;
          }
        }
        return newTime;
      });
    }, 1000);

    // Set timeout for query (25 seconds)
    timeoutRef.current = setTimeout(async () => {
      setLoading(false);
      
      // Clear elapsed time interval
      if (elapsedTimeRef.current) {
        clearInterval(elapsedTimeRef.current);
        elapsedTimeRef.current = null;
      }
      setElapsedTime(0);
      
      // Check API key status before showing timeout message
      const { hasOpenAI, hasAnthropic } = await checkAPIKeys();
      const hasAnyKey = hasOpenAI || hasAnthropic;
      
      let timeoutMessage: string;
      let exampleQuestions: string[] = [];
      
      if (!hasAnyKey) {
        // No API key - suggest adding one
        timeoutMessage = "This is taking longer than expected. Add your OpenAI API key in Settings for faster responses.";
        exampleQuestions = ['Add API Key'];
      } else {
        // Has key but still timing out - might be invalid or data issue
        timeoutMessage = "This is taking longer than expected. Your data might be very large, or you're using the local AI model which can be slow. Try asking a simpler question or filter your data.";
        exampleQuestions = ['Show revenue last month', 'Top 10 products'];
      }
      
      addErrorMessage(
        conversationId,
        timeoutMessage,
        exampleQuestions
      );
    }, TIMEOUT_DURATION);

    try {
      // Call chat API
      const response: ChatResponse | null = await chatService.askQuestion(userQuestion);

      // Clear timeout if response received
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      
      // Clear elapsed time interval
      if (elapsedTimeRef.current) {
        clearInterval(elapsedTimeRef.current);
        elapsedTimeRef.current = null;
      }
      setElapsedTime(0);

      if (!response) {
        addErrorMessage(
          conversationId,
          "I couldn't understand that question. Try asking: 'Show revenue last month' or 'Top 10 products'",
          ['Show revenue last month', 'Top 10 products', 'Which items have high refund rates?']
        );
        setLoading(false);
        return;
      }

      // Check if response has error
      if (response.error) {
        addErrorMessage(
          conversationId,
          response.error || "I couldn't process that question. Please try rephrasing it.",
          ['Show revenue last month', 'Top 10 products']
        );
        setLoading(false);
        return;
      }

      // Check if answer is empty or unclear
      if (!response.answer || response.answer.trim().length === 0) {
        addErrorMessage(
          conversationId,
          "I couldn't understand that question. Try asking: 'Show revenue last month' or 'Top 10 products'",
          ['Show revenue last month', 'Top 10 products', 'Which items have high refund rates?']
        );
        setLoading(false);
        return;
      }

      // Success - add assistant response
      const confidence = response.confidence ?? 1.0;
      const provider = response.provider;
      const executionTime = response.execution_time ?? 0;
      
      // Check if we should show API key suggestion
      const shouldShowSuggestion = 
        (provider === 'ollama' && (confidence < 0.7 || executionTime > 30)) ||
        (response.suggestion && response.suggestion.length > 0);
      
      // Track low-confidence responses
      if (shouldShowSuggestion) {
        const newCount = lowConfidenceCount + 1;
        setLowConfidenceCount(newCount);
        
        // Show banner after 3 consecutive low-confidence responses
        if (newCount >= 3 && !bannerDismissed) {
          setShowBanner(true);
        }
      } else {
        // Reset counter on successful response
        setLowConfidenceCount(0);
      }
      
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sql: response.sql,
        data: response.data,
        executionTime: executionTime,
        confidence: confidence,
        provider: provider,
        showApiKeySuggestion: shouldShowSuggestion,
      };
      addMessage(conversationId, assistantMessage);
      setLoading(false);

    } catch (error: any) {
      // Clear timeout on error
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      
      // Clear elapsed time interval
      if (elapsedTimeRef.current) {
        clearInterval(elapsedTimeRef.current);
        elapsedTimeRef.current = null;
      }
      setElapsedTime(0);

      console.error('Chat error:', error);

      let errorMessage = "Oops, something went wrong. We've been notified. Please try again in a moment.";

      // Handle different error types
      if (error instanceof AxiosError) {
        if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
          // Check API key status for timeout errors
          const { hasOpenAI, hasAnthropic } = await checkAPIKeys();
          const hasAnyKey = hasOpenAI || hasAnthropic;
          
          if (!hasAnyKey) {
            errorMessage = "This is taking longer than expected. Add your OpenAI API key in Settings for faster responses.";
          } else {
            errorMessage = "This is taking longer than expected. Your data might be very large, or you're using the local AI model which can be slow. Try asking a simpler question or filter your data.";
          }
        } else if (error.code === 'ERR_NETWORK' || error.message?.includes('Network Error')) {
          errorMessage = "Connection issue. Please check your internet and try again.";
        } else if (error.response?.status >= 500) {
          errorMessage = "Server error occurred. Please try again later or contact support if the problem persists.";
        } else if (error.response?.status >= 400 && error.response?.status < 500) {
          const serverMessage = error.response?.data?.detail || error.response?.data?.message;
          errorMessage = serverMessage || "I couldn't process that question. Please try rephrasing it.";
        }
      }

      addErrorMessage(
        conversationId,
        errorMessage,
        ['Show revenue last month', 'Top 10 products']
      );
      setLoading(false);
    }
  }, [inputValue, loading, activeConversationId, addMessage, setLoading, addErrorMessage]);

  // Handle example question click
  const handleExampleQuestion = (question: string) => {
    setInputValue(question);
    // Auto-send after a short delay
    setTimeout(() => {
      handleSendMessage();
    }, 100);
  };

  // Handle Enter key press (Shift+Enter for new line, Enter to send)
  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle example question clicks from error messages
  useEffect(() => {
    const handleExampleQuestionEvent = (e: CustomEvent) => {
      const question = e.detail;
      setInputValue(question);
      setTimeout(() => {
        handleSendMessage();
      }, 100);
    };
    
    window.addEventListener('exampleQuestion', handleExampleQuestionEvent as EventListener);
    return () => {
      window.removeEventListener('exampleQuestion', handleExampleQuestionEvent as EventListener);
    };
  }, [handleSendMessage]);

  // Cleanup timeout and interval on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (elapsedTimeRef.current) {
        clearInterval(elapsedTimeRef.current);
      }
    };
  }, []);

  // Reset low-confidence counter and refresh API keys when user navigates to settings
  useEffect(() => {
    const handleLocationChange = () => {
      const currentPath = window.location.pathname;
      if (currentPath === '/settings') {
        // Reset counters when user goes to settings
        setLowConfidenceCount(0);
        setShowBanner(false);
      } else if (currentPath === '/ai-analyst') {
        // Refresh API key status when returning from settings
        checkAPIKeys();
      }
    };
    
    // Check on mount and listen for navigation
    handleLocationChange();
    const interval = setInterval(handleLocationChange, 1000);
    
    return () => clearInterval(interval);
  }, []);

  // Show loading state while checking for data
  if (checkingData || hasData === null) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '400px',
          padding: SPACING.lg,
        }}
      >
        <Spin size="large" />
      </div>
    );
  }

  // Show empty state if no data exists
  if (hasData === false) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 'calc(100vh - 64px)',
          width: '100%',
          padding: SPACING.lg,
        }}
      >
        <div
          style={{
            maxWidth: '500px',
            width: '100%',
          }}
        >
          <EmptyState
            icon={<RobotOutlined />}
            title="AI Analyst Waiting for Data"
            description={
              <div>
                <p
                  style={{
                    marginBottom: SPACING.md,
                    marginTop: 0,
                    fontSize: '16px',
                    fontWeight: 400,
                    color: '#6B7280',
                    lineHeight: 1.6,
                    textAlign: 'center',
                  }}
                >
                  Upload a CSV file first, then ask me anything about your data!
                </p>
              </div>
            }
            primaryButton={{
              text: 'Upload Data',
              onClick: () => navigate('/workspace'),
            }}
          />
        </div>
      </div>
    );
  }

  // Main chat interface with split screen
  return (
    <div
      style={{
        display: 'flex',
        height: 'calc(100vh - 64px)',
        width: '100%',
        overflow: 'hidden',
      }}
    >
      {/* Left Sidebar - 25% */}
      <div
        style={{
          width: '25%',
          borderRight: `1px solid #E5E7EB`,
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: COLORS.neutralLight,
        }}
      >
        {/* New Chat Button */}
        <div
          style={{
            padding: SPACING.md,
            borderBottom: `1px solid #E5E7EB`,
          }}
        >
          <Button
            type="primary"
            icon={<PlusOutlined />}
            block
            onClick={handleNewChat}
            style={{
              height: '40px',
              borderRadius: BORDER_RADIUS.md,
              transition: 'all 0.2s ease',
            }}
          >
            New Chat
          </Button>
        </div>

        {/* Search Box */}
        <div
          style={{
            padding: SPACING.md,
            borderBottom: `1px solid #E5E7EB`,
          }}
        >
          <Input
            prefix={<SearchOutlined />}
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            allowClear
            style={{
              borderRadius: BORDER_RADIUS.md,
            }}
          />
        </div>

        {/* Conversation List */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: SPACING.sm,
          }}
        >
          {/* Today */}
          {conversationGroups.today.length > 0 && (
            <div style={{ marginBottom: SPACING.md }}>
              <div
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#6B7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  marginBottom: SPACING.xs,
                  padding: `0 ${SPACING.xs}`,
                }}
              >
                Today
              </div>
              {conversationGroups.today.map((conv) => (
                <ConversationItem
                  key={conv.id}
                  conversation={conv}
                  isActive={conv.id === activeConversationId}
                  isHovered={conv.id === hoveredConversationId}
                  onHover={setHoveredConversationId}
                  onClick={() => handleConversationClick(conv.id)}
                  onDelete={(e) => handleDeleteConversation(e, conv.id)}
                />
              ))}
            </div>
          )}

          {/* Yesterday */}
          {conversationGroups.yesterday.length > 0 && (
            <div style={{ marginBottom: SPACING.md }}>
              <div
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#6B7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  marginBottom: SPACING.xs,
                  padding: `0 ${SPACING.xs}`,
                }}
              >
                Yesterday
              </div>
              {conversationGroups.yesterday.map((conv) => (
                <ConversationItem
                  key={conv.id}
                  conversation={conv}
                  isActive={conv.id === activeConversationId}
                  isHovered={conv.id === hoveredConversationId}
                  onHover={setHoveredConversationId}
                  onClick={() => handleConversationClick(conv.id)}
                  onDelete={(e) => handleDeleteConversation(e, conv.id)}
                />
              ))}
            </div>
          )}

          {/* Last 7 Days */}
          {conversationGroups.last7Days.length > 0 && (
            <div style={{ marginBottom: SPACING.md }}>
              <div
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#6B7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  marginBottom: SPACING.xs,
                  padding: `0 ${SPACING.xs}`,
                }}
              >
                Last 7 Days
              </div>
              {conversationGroups.last7Days.map((conv) => (
                <ConversationItem
                  key={conv.id}
                  conversation={conv}
                  isActive={conv.id === activeConversationId}
                  isHovered={conv.id === hoveredConversationId}
                  onHover={setHoveredConversationId}
                  onClick={() => handleConversationClick(conv.id)}
                  onDelete={(e) => handleDeleteConversation(e, conv.id)}
                />
              ))}
            </div>
          )}

          {/* Older */}
          {conversationGroups.older.length > 0 && (
            <div style={{ marginBottom: SPACING.md }}>
              <div
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#6B7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  marginBottom: SPACING.xs,
                  padding: `0 ${SPACING.xs}`,
                }}
              >
                Older
              </div>
              {conversationGroups.older.map((conv) => (
                <ConversationItem
                  key={conv.id}
                  conversation={conv}
                  isActive={conv.id === activeConversationId}
                  isHovered={conv.id === hoveredConversationId}
                  onHover={setHoveredConversationId}
                  onClick={() => handleConversationClick(conv.id)}
                  onDelete={(e) => handleDeleteConversation(e, conv.id)}
                />
              ))}
            </div>
          )}

          {/* Empty state */}
          {filteredConversations.length === 0 && (
            <div
              style={{
                textAlign: 'center',
                color: '#9CA3AF',
                padding: SPACING.lg,
                fontSize: '14px',
              }}
            >
              {searchQuery ? 'No conversations found' : 'No conversations yet'}
            </div>
          )}
        </div>
      </div>

      {/* Right Chat Area - 75% */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#FFFFFF',
        }}
      >
        {/* API Key Status Indicator - Top Right Header */}
        <div
          style={{
            padding: `${SPACING.xs} ${SPACING.md}`,
            borderBottom: `1px solid #E5E7EB`,
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            backgroundColor: '#FFFFFF',
          }}
        >
          <div
            onClick={() => navigate('/settings')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: SPACING.xs,
              padding: `${SPACING.xs} ${SPACING.sm}`,
              borderRadius: BORDER_RADIUS.sm,
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 500,
              transition: 'all 0.2s ease',
              backgroundColor: 'transparent',
              border: 'none',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#F3F4F6';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            {/* Status Dot */}
            <div
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor:
                  apiKeyStatus.has_key && apiKeyStatus.is_valid
                    ? COLORS.success // Green
                    : apiKeyStatus.has_key && !apiKeyStatus.is_valid
                    ? COLORS.danger // Red
                    : COLORS.warning, // Yellow
                flexShrink: 0,
                boxShadow: `0 0 0 2px ${
                  apiKeyStatus.has_key && apiKeyStatus.is_valid
                    ? `${COLORS.success}20`
                    : apiKeyStatus.has_key && !apiKeyStatus.is_valid
                    ? `${COLORS.danger}20`
                    : `${COLORS.warning}20`
                }`,
              }}
            />
            {/* Status Text */}
            <span
              style={{
                fontSize: '14px',
                color:
                  apiKeyStatus.has_key && apiKeyStatus.is_valid
                    ? COLORS.success
                    : apiKeyStatus.has_key && !apiKeyStatus.is_valid
                    ? COLORS.danger
                    : COLORS.warning,
                fontWeight: 500,
              }}
            >
              {apiKeyStatus.has_key && apiKeyStatus.is_valid
                ? 'OpenAI API key active'
                : apiKeyStatus.has_key && !apiKeyStatus.is_valid
                ? 'API key invalid'
                : 'Using local AI (slower)'}
            </span>
          </div>
        </div>

        {/* Banner Notification - After 3 low-confidence responses */}
        {showBanner && !bannerDismissed && (
          <div
            style={{
              padding: `${SPACING.md} ${SPACING.lg}`,
              backgroundColor: '#FEF3C7',
              borderBottom: `1px solid #FCD34D`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: SPACING.md,
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: SPACING.sm,
                flex: 1,
              }}
            >
              <ThunderboltOutlined
                style={{
                  fontSize: '18px',
                  color: '#D97706',
                }}
              />
              <span
                style={{
                  fontSize: '14px',
                  color: '#92400E',
                  lineHeight: 1.5,
                }}
              >
                Having trouble? Add your OpenAI or Anthropic API key for faster, more accurate results.
              </span>
            </div>
            <div
              style={{
                display: 'flex',
                gap: SPACING.sm,
                alignItems: 'center',
              }}
            >
              <Button
                type="primary"
                size="small"
                icon={<SettingOutlined />}
                onClick={() => {
                  navigate('/settings');
                  setShowBanner(false);
                  setBannerDismissed(true);
                }}
                style={{
                  backgroundColor: '#D97706',
                  borderColor: '#D97706',
                  borderRadius: BORDER_RADIUS.sm,
                }}
              >
                Add API Key
              </Button>
              <Button
                type="text"
                size="small"
                icon={<CloseOutlined />}
                onClick={() => {
                  setShowBanner(false);
                  setBannerDismissed(true);
                }}
                style={{
                  color: '#92400E',
                }}
              >
                Dismiss
              </Button>
            </div>
          </div>
        )}

        {/* Chat Messages Area */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: SPACING.md,
            display: 'flex',
            flexDirection: 'column',
            gap: SPACING.md,
          }}
        >
          {currentMessages.length === 0 ? (
            /* Empty State with Suggested Questions */
            <div
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: SPACING.lg,
              }}
            >
              <RobotOutlined
                style={{
                  fontSize: '64px',
                  color: COLORS.primary,
                  opacity: 0.6,
                }}
              />
              <div
                style={{
                  textAlign: 'center',
                  maxWidth: '500px',
                }}
              >
                <h3
                  style={{
                    fontSize: '20px',
                    fontWeight: 600,
                    color: COLORS.neutralDark,
                    marginBottom: SPACING.sm,
                    marginTop: 0,
                  }}
                >
                  Start a conversation
                </h3>
                <p
                  style={{
                    fontSize: '14px',
                    color: '#6B7280',
                    marginBottom: SPACING.md,
                    marginTop: 0,
                  }}
                >
                  Ask questions about your data in plain language
                </p>
              </div>
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: SPACING.sm,
                  width: '100%',
                  maxWidth: '500px',
                }}
              >
                {SUGGESTED_QUESTIONS.map((question, index) => (
                  <Button
                    key={index}
                    type="default"
                    onClick={() => handleExampleQuestion(question)}
                    style={{
                      textAlign: 'left',
                      height: 'auto',
                      padding: `${SPACING.sm} ${SPACING.md}`,
                      borderRadius: BORDER_RADIUS.md,
                      border: `1px solid #E5E7EB`,
                      transition: 'all 0.2s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = COLORS.primary;
                      e.currentTarget.style.backgroundColor = `${COLORS.primary}10`;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = '#E5E7EB';
                      e.currentTarget.style.backgroundColor = '#FFFFFF';
                    }}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            /* Messages */
            <>
              {currentMessages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              
              {/* Loading indicator with animated dots and progress */}
              {loading && (() => {
                const elapsedSeconds = elapsedTime;
                const progressPercent = Math.min((elapsedSeconds * 1000 / TIMEOUT_DURATION) * 100, 100);
                
                // Determine message based on elapsed time
                let message = "AI is thinking...";
                if (elapsedSeconds >= 10 && elapsedSeconds < 20) {
                  message = `Still thinking... (${elapsedSeconds}s)`;
                } else if (elapsedSeconds >= 20 && elapsedSeconds < 25) {
                  message = `Almost there... (${elapsedSeconds}s)`;
                } else if (elapsedSeconds >= 25) {
                  message = `Taking longer than expected... (${elapsedSeconds}s)`;
                } else {
                  message = `Thinking (${elapsedSeconds}s)...`;
                }
                
                return (
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: SPACING.xs,
                      color: '#6B7280',
                      padding: `${SPACING.sm} ${SPACING.md}`,
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: SPACING.sm,
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          gap: '4px',
                          alignItems: 'center',
                        }}
                      >
                        <div
                          style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: COLORS.primary,
                            animation: 'bounce 1.4s infinite ease-in-out',
                            animationDelay: '0s',
                          }}
                        />
                        <div
                          style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: COLORS.primary,
                            animation: 'bounce 1.4s infinite ease-in-out',
                            animationDelay: '0.2s',
                          }}
                        />
                        <div
                          style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: COLORS.primary,
                            animation: 'bounce 1.4s infinite ease-in-out',
                            animationDelay: '0.4s',
                          }}
                        />
                      </div>
                      <span style={{ fontSize: '14px' }}>{message}</span>
                    </div>
                    {/* Progress indicator */}
                    <div
                      style={{
                        width: '100%',
                        height: '4px',
                        backgroundColor: '#E5E7EB',
                        borderRadius: '2px',
                        overflow: 'hidden',
                        marginTop: '4px',
                      }}
                    >
                      <div
                        style={{
                          width: `${progressPercent}%`,
                          height: '100%',
                          backgroundColor: COLORS.primary,
                          borderRadius: '2px',
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>
                  </div>
                );
              })()}
              
              <div ref={chatEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div
          style={{
            padding: SPACING.md,
            borderTop: `1px solid #E5E7EB`,
            backgroundColor: '#FFFFFF',
          }}
        >
          <div
            style={{
              maxWidth: '1200px',
              margin: '0 auto',
              border: '1px solid #D1D5DB', // gray-300
              borderRadius: '8px',
              padding: '8px',
              display: 'flex',
              gap: '8px',
              alignItems: 'center',
              backgroundColor: '#FFFFFF',
            }}
          >
            <TextArea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask a question about your data... (Shift+Enter for new line, Enter to send)"
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={loading}
              style={{
                flex: 1,
                padding: '12px',
                minHeight: '44px',
                borderRadius: BORDER_RADIUS.sm,
                fontSize: '14px',
                lineHeight: 1.5,
                border: 'none',
                boxShadow: 'none',
                transition: 'all 0.2s ease',
              }}
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || loading}
              loading={loading}
              style={{
                minWidth: '80px',
                minHeight: '44px',
                padding: '8px 16px',
                borderRadius: BORDER_RADIUS.sm,
                backgroundColor: '#2563eb', // blue-600 for better contrast
                borderColor: '#2563eb',
                color: '#ffffff', // white text
                fontSize: '14px',
                fontWeight: 500,
                transition: 'all 0.2s ease',
                flexShrink: 0,
              }}
              onMouseEnter={(e) => {
                if (!e.currentTarget.disabled) {
                  e.currentTarget.style.backgroundColor = '#1d4ed8'; // blue-700 on hover
                  e.currentTarget.style.borderColor = '#1d4ed8';
                }
              }}
              onMouseLeave={(e) => {
                if (!e.currentTarget.disabled) {
                  e.currentTarget.style.backgroundColor = '#2563eb'; // blue-600 default
                  e.currentTarget.style.borderColor = '#2563eb';
                }
              }}
            >
              Send
            </Button>
          </div>
        </div>
      </div>

      {/* CSS for animated dots and Send button states */}
      <style>{`
        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
        
        /* Ensure Send button disabled state has proper contrast */
        .ant-btn-primary:disabled,
        .ant-btn-primary.ant-btn-loading {
          background-color: #D1D5DB !important; /* gray-300 */
          border-color: #D1D5DB !important;
          color: #6B7280 !important; /* gray-500 */
          cursor: not-allowed;
        }
        
        /* Ensure loading spinner is visible on disabled button */
        .ant-btn-primary.ant-btn-loading .anticon {
          color: #6B7280 !important;
        }
        
        /* Ensure textarea focus doesn't show border (container handles it) */
        .ant-input:focus,
        .ant-input-focused {
          border: none !important;
          box-shadow: none !important;
        }
      `}</style>
    </div>
  );
};

// Conversation Item Component
interface ConversationItemProps {
  conversation: ChatConversation;
  isActive: boolean;
  isHovered: boolean;
  onHover: (id: string | null) => void;
  onClick: () => void;
  onDelete: (e: React.MouseEvent) => void;
}

const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  isActive,
  isHovered,
  onHover,
  onClick,
  onDelete,
}) => {
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => onHover(conversation.id)}
      onMouseLeave={() => onHover(null)}
      style={{
        padding: `${SPACING.sm} ${SPACING.xs}`,
        borderRadius: BORDER_RADIUS.sm,
        cursor: 'pointer',
        backgroundColor: isActive
          ? `${COLORS.primary}15`
          : isHovered
          ? '#F3F4F6'
          : 'transparent',
        border: isActive ? `1px solid ${COLORS.primary}40` : '1px solid transparent',
        marginBottom: SPACING.xs,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        transition: 'all 0.2s ease',
        position: 'relative',
      }}
    >
      <div
        style={{
          flex: 1,
          minWidth: 0,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            fontSize: '14px',
            fontWeight: isActive ? 600 : 400,
            color: COLORS.neutralDark,
            marginBottom: '2px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {conversation.title}
        </div>
        <div
          style={{
            fontSize: '12px',
            color: '#9CA3AF',
          }}
        >
          {dayjs(conversation.updatedAt).fromNow()}
        </div>
      </div>
      {isHovered && (
        <Button
          type="text"
          icon={<DeleteOutlined />}
          size="small"
          danger
          onClick={onDelete}
          style={{
            opacity: 0.7,
            transition: 'opacity 0.2s ease',
          }}
        />
      )}
    </div>
  );
};

// Message Bubble Component
interface MessageBubbleProps {
  message: ChatMessage;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const navigate = useNavigate();
  const isUser = message.role === 'user';
  const isError = message.isError;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        gap: SPACING.xs,
        transition: 'all 0.2s ease',
      }}
    >
      {/* Message Bubble */}
      <div
        style={{
          maxWidth: '70%',
          padding: `${SPACING.sm} ${SPACING.md}`,
          borderRadius: BORDER_RADIUS.md,
          backgroundColor: isUser
            ? COLORS.primary
            : isError
            ? '#FEF2F2'
            : '#F3F4F6',
          color: isUser
            ? '#FFFFFF'
            : isError
            ? COLORS.danger
            : COLORS.neutralDark,
          borderLeft: isError ? `3px solid ${COLORS.danger}` : 'none',
          border: isError ? `1px solid ${COLORS.danger}20` : 'none',
          transition: 'all 0.2s ease',
        }}
      >
        {isError && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: SPACING.xs,
              marginBottom: SPACING.xs,
            }}
          >
            <ExclamationCircleOutlined
              style={{
                color: COLORS.danger,
                fontSize: '16px',
              }}
            />
            <span
              style={{
                fontWeight: 600,
                fontSize: '14px',
                color: COLORS.danger,
              }}
            >
              Query Error
            </span>
          </div>
        )}
        <p
          style={{
            margin: 0,
            fontSize: '14px',
            lineHeight: 1.6,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            marginBottom: isError && message.sql ? SPACING.sm : 0,
          }}
        >
          {message.content}
        </p>
        
        {/* Show SQL query for errors (collapsible) */}
        {isError && message.sql && (
          <details
            style={{
              marginTop: SPACING.xs,
              fontSize: '12px',
            }}
          >
            <summary
              style={{
                cursor: 'pointer',
                color: COLORS.danger,
                fontWeight: 500,
                padding: `${SPACING.xs} 0`,
                userSelect: 'none',
              }}
            >
              View SQL Query
            </summary>
            <div
              style={{
                marginTop: SPACING.xs,
                padding: SPACING.sm,
                backgroundColor: '#1F2937',
                color: '#E5E7EB',
                borderRadius: BORDER_RADIUS.sm,
                fontFamily: 'monospace',
                fontSize: '12px',
                lineHeight: 1.6,
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {message.sql}
            </div>
          </details>
        )}
        
        {/* API Key Suggestion - Inline Message */}
        {!isError && message.showApiKeySuggestion && (
          <div
            style={{
              marginTop: SPACING.sm,
              padding: SPACING.sm,
              backgroundColor: '#FEF3C7',
              border: `1px solid #FCD34D`,
              borderRadius: BORDER_RADIUS.sm,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: SPACING.sm,
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: SPACING.xs,
                flex: 1,
              }}
            >
              <ThunderboltOutlined
                style={{
                  fontSize: '14px',
                  color: '#D97706',
                }}
              />
              <span
                style={{
                  fontSize: '13px',
                  color: '#92400E',
                  lineHeight: 1.5,
                }}
              >
                I couldn't generate a confident answer with the local model.
              </span>
            </div>
            <Button
              type="primary"
              size="small"
              icon={<SettingOutlined />}
              onClick={() => navigate('/settings')}
              style={{
                backgroundColor: '#D97706',
                borderColor: '#D97706',
                borderRadius: BORDER_RADIUS.sm,
                fontSize: '12px',
                height: '28px',
                padding: `0 ${SPACING.sm}`,
              }}
            >
              Add API Key for Better Results
            </Button>
          </div>
        )}
        
        {/* Example Questions for Errors */}
        {isError && message.exampleQuestions && message.exampleQuestions.length > 0 && (
          <div
            style={{
              marginTop: SPACING.sm,
              display: 'flex',
              flexDirection: 'column',
              gap: SPACING.xs,
            }}
          >
            {message.exampleQuestions[0] === 'Add API Key' ? (
              // Special handling for API key suggestion
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: SPACING.sm,
                  padding: SPACING.sm,
                  backgroundColor: '#FEF3C7',
                  border: `1px solid #FCD34D`,
                  borderRadius: BORDER_RADIUS.sm,
                }}
              >
                <ThunderboltOutlined
                  style={{
                    fontSize: '14px',
                    color: '#D97706',
                  }}
                />
                <span
                  style={{
                    fontSize: '13px',
                    color: '#92400E',
                    flex: 1,
                  }}
                >
                  Add your OpenAI API key for faster responses.
                </span>
                <Button
                  type="primary"
                  size="small"
                  icon={<SettingOutlined />}
                  onClick={() => navigate('/settings')}
                  style={{
                    backgroundColor: '#D97706',
                    borderColor: '#D97706',
                    borderRadius: BORDER_RADIUS.sm,
                    fontSize: '12px',
                    height: '28px',
                    padding: `0 ${SPACING.sm}`,
                  }}
                >
                  Add API Key
                </Button>
              </div>
            ) : (
              // Regular example questions
              <>
                <p
                  style={{
                    fontSize: '12px',
                    fontWeight: 600,
                    color: COLORS.danger,
                    margin: 0,
                    marginBottom: SPACING.xs,
                  }}
                >
                  Try asking:
                </p>
                {message.exampleQuestions.map((question, index) => (
                  <Button
                    key={index}
                    type="text"
                    size="small"
                    onClick={() => {
                      // This will be handled by parent component
                      const event = new CustomEvent('exampleQuestion', { detail: question });
                      window.dispatchEvent(event);
                    }}
                    style={{
                      textAlign: 'left',
                      color: COLORS.danger,
                      fontSize: '12px',
                      padding: `${SPACING.xs} ${SPACING.sm}`,
                      height: 'auto',
                      border: `1px solid ${COLORS.danger}40`,
                      borderRadius: BORDER_RADIUS.sm,
                      transition: 'all 0.2s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = `${COLORS.danger}10`;
                      e.currentTarget.style.borderColor = COLORS.danger;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.borderColor = `${COLORS.danger}40`;
                    }}
                  >
                    {question}
                  </Button>
                ))}
              </>
            )}
          </div>
        )}
      </div>
      
      {/* Timestamp */}
      <span
        style={{
          fontSize: '12px',
          color: '#9CA3AF',
          paddingLeft: isUser ? 0 : SPACING.sm,
          paddingRight: isUser ? SPACING.sm : 0,
        }}
      >
        {dayjs(message.timestamp).format('h:mm A')}
      </span>
    </div>
  );
};

export default AIAnalyst;
