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
  Switch,
} from 'antd';
import {
  KeyOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LockOutlined,
} from '@ant-design/icons';
import { apiKeyService, APIKeyInfo } from '@/services/apiKeyService';
import { usersService } from '@/services/api';
import { useAuthStore } from '@/store/authStore';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS } from '@/styles/designTokens';
import { CheckCircleOutlined as CheckIcon, RocketOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Password } = Input;

type Provider = 'openai' | 'anthropic' | 'gemini';

interface APIKeyState {
  openai: APIKeyInfo | null;
  anthropic: APIKeyInfo | null;
  gemini: APIKeyInfo | null;
  loading: boolean;
}

const Settings: React.FC = () => {
  const { user } = useAuthStore();
  const [apiKeys, setApiKeys] = useState<APIKeyState>({
    openai: null,
    anthropic: null,
    gemini: null,
    loading: true,
  });

  const [addModalVisible, setAddModalVisible] = useState(false);
  const [removeModalVisible, setRemoveModalVisible] = useState(false);
  const [currentProvider, setCurrentProvider] = useState<Provider | null>(null);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Change password state
  const [changePasswordVisible, setChangePasswordVisible] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [changingPassword, setChangingPassword] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  // Fetch API keys on mount
  useEffect(() => {
    loadAPIKeys();
  }, []);

  const loadAPIKeys = async () => {
    if (!user?.id) {
      console.warn('User not authenticated, cannot load API keys');
      setApiKeys((prev) => ({ ...prev, loading: false }));
      return;
    }

    try {
      setApiKeys((prev) => ({ ...prev, loading: true }));
      
      // Only load Gemini key for now (OpenAI/Anthropic disabled)
      let geminiKey = null;
      try {
        geminiKey = await apiKeyService.getAPIKey('gemini', String(user.id));
      } catch (err: any) {
        // 404 is normal when no key exists
        if (err.response?.status !== 404) {
          console.warn('Failed to load Gemini key:', err);
        }
      }

      setApiKeys({
        openai: null,  // Disabled
        anthropic: null,  // Disabled
        gemini: geminiKey,
        loading: false,
      });
    } catch (error: any) {
      console.error('Failed to load API keys:', error);
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

    if (currentProvider === 'gemini' && apiKeyInput.trim().length < 20) {
      setError('Invalid Gemini API key format. Key should be at least 20 characters long.');
      return;
    }

    setValidating(true);
    setError(null);

    if (!user?.id) {
      setError('You must be logged in to save API keys');
      return;
    }

    try {
      const savedKey = await apiKeyService.createAPIKey(
        currentProvider,
        apiKeyInput.trim(),
        String(user.id)
      );

      // Update state
      setApiKeys((prev) => ({
        ...prev,
        [currentProvider]: savedKey,
      }));

      const providerName = currentProvider === 'openai' ? 'OpenAI' : currentProvider === 'anthropic' ? 'Anthropic' : 'Gemini';
      message.success(`${providerName} API key saved successfully`);
      setAddModalVisible(false);
      setApiKeyInput('');
      
      // Reload all API keys to ensure consistency
      await loadAPIKeys();
    } catch (error: any) {
      console.error('API key save error:', error);
      console.error('Error response data:', error.response?.data);
      console.error('Error response status:', error.response?.status);
      
      let errorMessage = 'Failed to validate or save API key';
      
      if (error.response) {
        // Backend returned an error
        const responseData = error.response.data;
        const detail = responseData?.detail || responseData?.message || responseData?.error;
        
        if (detail) {
          // Handle string or object detail
          if (typeof detail === 'string') {
            errorMessage = detail;
            
            // Special handling for Gemini API errors - preserve multi-line formatting
            if (currentProvider === 'gemini' && detail.includes('\n')) {
              // For Gemini, show the full error message with troubleshooting steps
              errorMessage = detail;
            } else if (currentProvider === 'gemini') {
              // Enhance Gemini error messages with troubleshooting
              if (detail.toLowerCase().includes('api key')) {
                errorMessage = `${detail}\n\nTroubleshooting:\n• Verify the API key is correct (no extra spaces)\n• Check that 'Generative Language API' is enabled in Google Cloud Console\n• Ensure API key has no IP/HTTP referrer restrictions\n• Verify the API key is active (not deleted or expired)`;
              } else if (detail.toLowerCase().includes('rate limit')) {
                errorMessage = `${detail}\n\nPlease wait a moment and try again, or check your API quota in Google Cloud Console.`;
              } else if (detail.toLowerCase().includes('timeout')) {
                errorMessage = `${detail}\n\nPlease check your internet connection and try again.`;
              }
            }
          } else if (typeof detail === 'object') {
            // If detail is an object, try to extract message
            errorMessage = detail.message || detail.error || JSON.stringify(detail);
          } else {
            errorMessage = String(detail);
          }
        } else if (error.response.status === 404) {
          errorMessage = 'API endpoint not found. Please check your API key format and try again.';
        } else if (error.response.status === 400) {
          // Try to get more specific error message
          if (responseData) {
            const detailStr = typeof responseData === 'string' ? responseData : JSON.stringify(responseData);
            errorMessage = detailStr;
            
            // Add Gemini-specific troubleshooting for 400 errors
            if (currentProvider === 'gemini') {
              errorMessage += '\n\nTroubleshooting:\n• Verify the API key is correct\n• Check that Generative Language API is enabled in Google Cloud Console\n• Ensure API key has no restrictions';
            }
          } else {
            errorMessage = 'Invalid API key or request. Please check your key and try again.';
          }
        } else if (error.response.status === 401) {
          errorMessage = 'Authentication required. Please refresh the page and try again.';
        } else if (error.response.status === 500) {
          errorMessage = 'Server error. Please try again later.';
        } else {
          errorMessage = `Error ${error.response.status}: ${error.response.statusText || 'Unknown error'}`;
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      // For Gemini errors, preserve newlines in the error display
      setError(errorMessage);
      
      // Show error message (truncate very long messages for toast)
      const toastMessage = errorMessage.length > 200 
        ? errorMessage.substring(0, 200) + '...' 
        : errorMessage;
      message.error(toastMessage, 10); // Show for 10 seconds for Gemini errors
    } finally {
      setValidating(false);
    }
  };

  const handleConfirmRemove = async () => {
    if (!currentProvider || !user?.id) return;

    try {
      await apiKeyService.deleteAPIKey(currentProvider, String(user.id));

      // Update state
      setApiKeys((prev) => ({
        ...prev,
        [currentProvider]: null,
      }));

      const providerName = currentProvider === 'openai' ? 'OpenAI' : currentProvider === 'anthropic' ? 'Anthropic' : 'Gemini';
      message.success(`${providerName} API key removed successfully`);
      setRemoveModalVisible(false);
    } catch (error: any) {
      message.error('Failed to remove API key');
      console.error('Failed to remove API key:', error);
    }
  };

  const handleToggleEnabled = async (provider: Provider, enabled: boolean) => {
    if (!user?.id) {
      message.error('You must be logged in to update API keys');
      return;
    }

    try {
      const updatedKey = await apiKeyService.updateEnabledStatus(provider, enabled, String(user.id));
      
      // Update state
      setApiKeys((prev) => ({
        ...prev,
        [provider]: updatedKey,
      }));

      const providerName = getProviderName(provider);
      message.success(`${providerName} API key ${enabled ? 'enabled' : 'disabled'} successfully`);
    } catch (error: any) {
      message.error(`Failed to ${enabled ? 'enable' : 'disable'} API key`);
      console.error('Failed to update enabled status:', error);
    }
  };

  const getProviderName = (provider: Provider) => {
    if (provider === 'openai') return 'OpenAI';
    if (provider === 'anthropic') return 'Anthropic';
    return 'Gemini';
  };

  const getProviderLink = (provider: Provider) => {
    if (provider === 'openai') return 'https://platform.openai.com/api-keys';
    if (provider === 'anthropic') return 'https://console.anthropic.com/settings/keys';
    return 'https://makersuite.google.com/app/apikey';
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
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: SPACING.sm }}>
                <Text strong style={{ fontSize: '14px' }}>
                  Enable for AI Chat:
                </Text>
                <Switch
                  checked={key.enabled}
                  onChange={(checked) => handleToggleEnabled(provider, checked)}
                  checkedChildren="Enabled"
                  unCheckedChildren="Disabled"
                />
              </div>
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
              {!key.enabled && (
                <Alert
                  message="This API key is disabled"
                  description="AI chat will not use this key. Enable it to use this provider."
                  type="info"
                  showIcon
                  style={{ marginTop: SPACING.sm, borderRadius: BORDER_RADIUS.md }}
                />
              )}
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

      {/* Plan & Billing Section */}
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
              <RocketOutlined style={{ marginRight: SPACING.xs, color: COLORS.primary }} />
              Plan & Billing
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Manage your subscription and view plan details
            </Paragraph>
          </div>

          {/* Current Plan */}
          <div>
            <Text strong style={{ fontSize: '16px', display: 'block', marginBottom: SPACING.sm }}>
              Current Plan: <span style={{ color: COLORS.primary, textTransform: 'capitalize' }}>{user?.plan || 'Free'}</span>
            </Text>
            {user?.plan === 'free' && (
              <Alert
                message="Free Plan Limits"
                description={
                  <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
                    <li>Up to 10,000 records</li>
                    <li>Basic analytics</li>
                    <li>AI chat support</li>
                    <li>Email support</li>
                  </ul>
                }
                type="info"
                showIcon
                style={{ borderRadius: BORDER_RADIUS.md }}
              />
            )}
          </div>

          {/* Upgrade Options */}
          <div>
            <Text strong style={{ fontSize: '14px', display: 'block', marginBottom: SPACING.md }}>
              Upgrade Options
            </Text>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: SPACING.md,
              }}
            >
              {/* Pro Plan Card */}
              <Card
                style={{
                  border: '1px solid #E2E8F0',
                  borderRadius: BORDER_RADIUS.md,
                }}
                bodyStyle={{ padding: SPACING.md }}
              >
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Title level={4} style={{ margin: 0 }}>
                    Pro
                  </Title>
                  <div>
                    <Text style={{ fontSize: '24px', fontWeight: '700' }}>$29</Text>
                    <Text type="secondary">/month</Text>
                  </div>
                  <Space direction="vertical" size="small" style={{ width: '100%', marginTop: SPACING.sm }}>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Unlimited records</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Advanced analytics</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Priority AI support</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Custom reports</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>API access</Text>
                    </div>
                  </Space>
                  <Button block disabled style={{ marginTop: SPACING.sm }}>
                    Coming Soon
                  </Button>
                </Space>
              </Card>

              {/* Enterprise Plan Card */}
              <Card
                style={{
                  border: '1px solid #E2E8F0',
                  borderRadius: BORDER_RADIUS.md,
                }}
                bodyStyle={{ padding: SPACING.md }}
              >
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Title level={4} style={{ margin: 0 }}>
                    Enterprise
                  </Title>
                  <div>
                    <Text style={{ fontSize: '24px', fontWeight: '700' }}>Custom</Text>
                  </div>
                  <Space direction="vertical" size="small" style={{ width: '100%', marginTop: SPACING.sm }}>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Everything in Pro</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Dedicated support</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>Custom integrations</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>SLA guarantee</Text>
                    </div>
                    <div>
                      <CheckIcon style={{ color: '#10B981', marginRight: '8px' }} />
                      <Text style={{ fontSize: '12px' }}>On-premise deployment</Text>
                    </div>
                  </Space>
                  <Button block disabled style={{ marginTop: SPACING.sm }}>
                    Coming Soon
                  </Button>
                </Space>
              </Card>
            </div>
          </div>
        </Space>
      </Card>

      {/* Security Section */}
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
              <LockOutlined style={{ marginRight: SPACING.xs, color: COLORS.primary }} />
              Security
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Manage your account security and password
            </Paragraph>
          </div>

          <div>
            <Button
              type="primary"
              onClick={() => setChangePasswordVisible(true)}
              style={{
                borderRadius: BORDER_RADIUS.md,
              }}
            >
              Change Password
            </Button>
          </div>
        </Space>
      </Card>

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
              {/* Gemini is the primary provider */}
              {renderAPIKeySection('gemini')}
              
              {/* OpenAI and Anthropic temporarily disabled */}
              {/* {renderAPIKeySection('openai')} */}
              {/* {renderAPIKeySection('anthropic')} */}
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
              description={
                <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {error.split('\n').map((line, index) => (
                    <div key={index} style={{ marginBottom: index < error.split('\n').length - 1 ? '4px' : 0 }}>
                      {line.startsWith('•') || line.startsWith('-') ? (
                        <span style={{ marginLeft: '8px' }}>{line}</span>
                      ) : (
                        <span>{line}</span>
                      )}
                    </div>
                  ))}
                </div>
              }
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

      {/* Change Password Modal */}
      <Modal
        title={
          <Space>
            <LockOutlined />
            <span>Change Password</span>
          </Space>
        }
        open={changePasswordVisible}
        onCancel={() => {
          setChangePasswordVisible(false);
          setCurrentPassword('');
          setNewPassword('');
          setConfirmPassword('');
          setPasswordError(null);
        }}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setChangePasswordVisible(false);
              setCurrentPassword('');
              setNewPassword('');
              setConfirmPassword('');
              setPasswordError(null);
            }}
            disabled={changingPassword}
            style={{ borderRadius: BORDER_RADIUS.md }}
          >
            Cancel
          </Button>,
          <Button
            key="change"
            type="primary"
            icon={changingPassword ? <Spin size="small" /> : <CheckCircleOutlined />}
            onClick={async () => {
              setPasswordError(null);

              // Validation
              if (!currentPassword) {
                setPasswordError('Please enter your current password');
                return;
              }

              if (!newPassword) {
                setPasswordError('Please enter a new password');
                return;
              }

              if (newPassword.length < 8) {
                setPasswordError('New password must be at least 8 characters long');
                return;
              }

              if (newPassword !== confirmPassword) {
                setPasswordError('New passwords do not match');
                return;
              }

              if (currentPassword === newPassword) {
                setPasswordError('New password must be different from current password');
                return;
              }

              setChangingPassword(true);
              try {
                await usersService.changePassword(currentPassword, newPassword);
                message.success('Password changed successfully. Please log in again.');
                setChangePasswordVisible(false);
                setCurrentPassword('');
                setNewPassword('');
                setConfirmPassword('');
                
                // Logout user after password change (sessions are invalidated)
                setTimeout(() => {
                  window.location.href = '/login';
                }, 2000);
              } catch (error: any) {
                const errorMessage = error.response?.data?.detail || error.message || 'Failed to change password';
                setPasswordError(errorMessage);
                message.error(errorMessage);
              } finally {
                setChangingPassword(false);
              }
            }}
            loading={changingPassword}
            style={{ borderRadius: BORDER_RADIUS.md }}
          >
            Change Password
          </Button>,
        ]}
        style={{ borderRadius: BORDER_RADIUS.md }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <div>
            <Text strong style={{ display: 'block', marginBottom: SPACING.xs }}>
              Current Password
            </Text>
            <Password
              placeholder="Enter your current password"
              value={currentPassword}
              onChange={(e) => {
                setCurrentPassword(e.target.value);
                setPasswordError(null);
              }}
              disabled={changingPassword}
              style={{ width: '100%', borderRadius: BORDER_RADIUS.md }}
            />
          </div>

          <div>
            <Text strong style={{ display: 'block', marginBottom: SPACING.xs }}>
              New Password
            </Text>
            <Password
              placeholder="Enter your new password (min 8 characters)"
              value={newPassword}
              onChange={(e) => {
                setNewPassword(e.target.value);
                setPasswordError(null);
              }}
              disabled={changingPassword}
              style={{ width: '100%', borderRadius: BORDER_RADIUS.md }}
            />
          </div>

          <div>
            <Text strong style={{ display: 'block', marginBottom: SPACING.xs }}>
              Confirm New Password
            </Text>
            <Password
              placeholder="Confirm your new password"
              value={confirmPassword}
              onChange={(e) => {
                setConfirmPassword(e.target.value);
                setPasswordError(null);
              }}
              disabled={changingPassword}
              style={{ width: '100%', borderRadius: BORDER_RADIUS.md }}
            />
          </div>

          {passwordError && (
            <Alert
              message="Error"
              description={passwordError}
              type="error"
              showIcon
              style={{ borderRadius: BORDER_RADIUS.md }}
            />
          )}

          <Alert
            message="Security Note"
            description="After changing your password, all active sessions will be invalidated and you'll need to log in again."
            type="info"
            showIcon
            style={{ borderRadius: BORDER_RADIUS.md }}
          />
        </Space>
      </Modal>
    </div>
  );
};

export default Settings;

