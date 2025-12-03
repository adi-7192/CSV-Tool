/**
 * Settings Page
 * 
 * Page for managing user settings, including LLM API keys.
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Modal,
  Input,
  message,
  Spin,
  Typography,
  Space,
  Alert,
} from 'antd';
import {
  KeyOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { apiKeyService, APIKeyInfo } from '@/services/apiKeyService';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS } from '@/styles/designTokens';

const { Title, Text, Paragraph } = Typography;
const { Password } = Input;

// TODO: Replace with actual user ID from authentication
const CURRENT_USER_ID = 'user-123'; // This should come from auth context

type Provider = 'openai' | 'anthropic';

interface APIKeyState {
  openai: APIKeyInfo | null;
  anthropic: APIKeyInfo | null;
  loading: boolean;
}

const Settings: React.FC = () => {
  const [apiKeys, setApiKeys] = useState<APIKeyState>({
    openai: null,
    anthropic: null,
    loading: true,
  });

  const [addModalVisible, setAddModalVisible] = useState(false);
  const [removeModalVisible, setRemoveModalVisible] = useState(false);
  const [currentProvider, setCurrentProvider] = useState<Provider | null>(null);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch API keys on mount
  useEffect(() => {
    loadAPIKeys();
  }, []);

  const loadAPIKeys = async () => {
    try {
      setApiKeys((prev) => ({ ...prev, loading: true }));
      
      const [openaiKey, anthropicKey] = await Promise.all([
        apiKeyService.getAPIKey('openai', CURRENT_USER_ID),
        apiKeyService.getAPIKey('anthropic', CURRENT_USER_ID),
      ]);

      setApiKeys({
        openai: openaiKey,
        anthropic: anthropicKey,
        loading: false,
      });
    } catch (error: any) {
      console.error('Failed to load API keys:', error);
      message.error('Failed to load API keys');
      setApiKeys((prev) => ({ ...prev, loading: false }));
    }
  };

  const handleAddKey = (provider: Provider) => {
    setCurrentProvider(provider);
    setApiKeyInput('');
    setError(null);
    setAddModalVisible(true);
  };

  const handleChangeKey = (provider: Provider) => {
    handleAddKey(provider);
  };

  const handleRemoveKey = (provider: Provider) => {
    setCurrentProvider(provider);
    setRemoveModalVisible(true);
  };

  const handleValidateAndSave = async () => {
    if (!apiKeyInput.trim()) {
      setError('Please enter an API key');
      return;
    }

    if (!currentProvider) return;

    // Validate format
    if (currentProvider === 'openai' && !apiKeyInput.trim().startsWith('sk-')) {
      setError('Invalid OpenAI API key format. Key must start with "sk-"');
      return;
    }

    if (currentProvider === 'anthropic' && !apiKeyInput.trim().startsWith('sk-ant-')) {
      setError('Invalid Anthropic API key format. Key must start with "sk-ant-"');
      return;
    }

    setValidating(true);
    setError(null);

    try {
      const savedKey = await apiKeyService.createAPIKey(
        currentProvider,
        apiKeyInput.trim(),
        CURRENT_USER_ID
      );

      // Update state
      setApiKeys((prev) => ({
        ...prev,
        [currentProvider]: savedKey,
      }));

      message.success(`${currentProvider === 'openai' ? 'OpenAI' : 'Anthropic'} API key saved successfully`);
      setAddModalVisible(false);
      setApiKeyInput('');
    } catch (error: any) {
      const errorMessage =
        error.response?.data?.detail ||
        error.message ||
        'Failed to validate or save API key';
      setError(errorMessage);
    } finally {
      setValidating(false);
    }
  };

  const handleConfirmRemove = async () => {
    if (!currentProvider) return;

    try {
      await apiKeyService.deleteAPIKey(currentProvider, CURRENT_USER_ID);

      // Update state
      setApiKeys((prev) => ({
        ...prev,
        [currentProvider]: null,
      }));

      message.success(`${currentProvider === 'openai' ? 'OpenAI' : 'Anthropic'} API key removed successfully`);
      setRemoveModalVisible(false);
    } catch (error: any) {
      message.error('Failed to remove API key');
      console.error('Failed to remove API key:', error);
    }
  };

  const getProviderName = (provider: Provider) => {
    return provider === 'openai' ? 'OpenAI' : 'Anthropic';
  };

  const getProviderLink = (provider: Provider) => {
    return provider === 'openai'
      ? 'https://platform.openai.com/api-keys'
      : 'https://console.anthropic.com/settings/keys';
  };

  const renderAPIKeySection = (provider: Provider) => {
    const key = apiKeys[provider];
    const hasKey = key !== null;

    return (
      <Card
        style={{
          marginBottom: SPACING.md,
          borderRadius: BORDER_RADIUS.md,
          boxShadow: SHADOWS.card,
        }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Title level={4} style={{ margin: 0 }}>
              {getProviderName(provider)}
            </Title>
            {hasKey && (
              <Space>
                <Button
                  icon={<EditOutlined />}
                  onClick={() => handleChangeKey(provider)}
                  style={{
                    borderRadius: BORDER_RADIUS.md,
                    transition: 'all 0.2s ease',
                  }}
                >
                  Change
                </Button>
                <Button
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleRemoveKey(provider)}
                  style={{
                    borderRadius: BORDER_RADIUS.md,
                    transition: 'all 0.2s ease',
                  }}
                >
                  Remove
                </Button>
              </Space>
            )}
          </div>

          {hasKey ? (
            <div>
              <Text type="secondary" style={{ fontSize: '14px', display: 'block', marginBottom: SPACING.xs }}>
                API Key:
              </Text>
              <Text
                code
                style={{
                  fontSize: '14px',
                  padding: `${SPACING.xs} ${SPACING.sm}`,
                  backgroundColor: COLORS.neutralLight,
                  borderRadius: BORDER_RADIUS.sm,
                }}
              >
                {key.masked_key}
              </Text>
              <div style={{ marginTop: SPACING.xs }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Last updated: {new Date(key.updated_at).toLocaleDateString()}
                </Text>
              </div>
            </div>
          ) : (
            <div>
              <Text type="secondary" style={{ fontSize: '14px', display: 'block', marginBottom: SPACING.sm }}>
                No API key set
              </Text>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => handleAddKey(provider)}
                style={{
                  borderRadius: BORDER_RADIUS.md,
                  transition: 'all 0.2s ease',
                }}
              >
                Add API Key
              </Button>
            </div>
          )}
        </Space>
      </Card>
    );
  };

  return (
    <div
      style={{
        maxWidth: '900px',
        margin: '0 auto',
        padding: SPACING.lg,
      }}
    >
      <Title level={2} style={{ marginBottom: SPACING.md }}>
        Settings
      </Title>

      {/* API Keys Section */}
      <Card
        style={{
          marginBottom: SPACING.lg,
          borderRadius: BORDER_RADIUS.md,
          boxShadow: SHADOWS.card,
        }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div>
            <Title level={3} style={{ marginBottom: SPACING.xs }}>
              <KeyOutlined style={{ marginRight: SPACING.xs, color: COLORS.primary }} />
              API Keys
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Manage your external LLM API keys for better AI responses
            </Paragraph>
          </div>

          {apiKeys.loading ? (
            <div style={{ textAlign: 'center', padding: SPACING.xl }}>
              <Spin size="large" />
            </div>
          ) : (
            <>
              {renderAPIKeySection('openai')}
              {renderAPIKeySection('anthropic')}
            </>
          )}
        </Space>
      </Card>

      {/* Add API Key Modal */}
      <Modal
        title={
          <Space>
            <KeyOutlined />
            <span>Add {currentProvider ? getProviderName(currentProvider) : ''} API Key</span>
          </Space>
        }
        open={addModalVisible}
        onCancel={() => {
          setAddModalVisible(false);
          setApiKeyInput('');
          setError(null);
        }}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setAddModalVisible(false);
              setApiKeyInput('');
              setError(null);
            }}
            disabled={validating}
            style={{ borderRadius: BORDER_RADIUS.md }}
          >
            Cancel
          </Button>,
          <Button
            key="save"
            type="primary"
            icon={validating ? <Spin size="small" /> : <CheckCircleOutlined />}
            onClick={handleValidateAndSave}
            loading={validating}
            style={{ borderRadius: BORDER_RADIUS.md }}
          >
            Validate & Save
          </Button>,
        ]}
        style={{ borderRadius: BORDER_RADIUS.md }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <div>
            <Text strong style={{ display: 'block', marginBottom: SPACING.xs }}>
              API Key
            </Text>
            <Password
              placeholder={`Enter your ${currentProvider ? getProviderName(currentProvider) : ''} API key`}
              value={apiKeyInput}
              onChange={(e) => {
                setApiKeyInput(e.target.value);
                setError(null);
              }}
              disabled={validating}
              style={{ width: '100%', borderRadius: BORDER_RADIUS.md }}
            />
          </div>

          <Alert
            message="Where to find your API key"
            description={
              <span>
                Get your API key from{' '}
                <a
                  href={currentProvider ? getProviderLink(currentProvider) : '#'}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: COLORS.primary }}
                >
                  {currentProvider ? getProviderName(currentProvider) : ''} platform
                </a>
                . Your key will be encrypted and stored securely.
              </span>
            }
            type="info"
            showIcon
            style={{ borderRadius: BORDER_RADIUS.md }}
          />

          {error && (
            <Alert
              message="Error"
              description={error}
              type="error"
              showIcon
              style={{ borderRadius: BORDER_RADIUS.md }}
            />
          )}

          {validating && (
            <div style={{ textAlign: 'center', padding: SPACING.md }}>
              <Spin size="large" />
              <div style={{ marginTop: SPACING.sm }}>
                <Text type="secondary">Validating API key...</Text>
              </div>
            </div>
          )}
        </Space>
      </Modal>

      {/* Remove Confirmation Modal */}
      <Modal
        title={
          <Space>
            <ExclamationCircleOutlined style={{ color: COLORS.danger }} />
            <span>⚠️ Remove API Key?</span>
          </Space>
        }
        open={removeModalVisible}
        onCancel={() => setRemoveModalVisible(false)}
        footer={[
          <Button
            key="cancel"
            onClick={() => setRemoveModalVisible(false)}
            style={{ borderRadius: BORDER_RADIUS.md }}
          >
            Cancel
          </Button>,
          <Button
            key="remove"
            danger
            icon={<DeleteOutlined />}
            onClick={handleConfirmRemove}
            style={{ borderRadius: BORDER_RADIUS.md }}
          >
            Remove Key
          </Button>,
        ]}
        style={{ borderRadius: BORDER_RADIUS.md }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Paragraph>
            Are you sure you want to remove your{' '}
            <Text strong>{currentProvider ? getProviderName(currentProvider) : ''}</Text> API key?
          </Paragraph>
          <Alert
            message="This action cannot be undone"
            description="You will need to add your API key again if you want to use this provider in the future."
            type="warning"
            showIcon
            style={{ borderRadius: BORDER_RADIUS.md }}
          />
        </Space>
      </Modal>
    </div>
  );
};

export default Settings;

