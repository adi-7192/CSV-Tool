import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { apiClient } from '@/services/api';

export interface User {
  id: string;
  email: string;
  role: 'user' | 'admin';
  plan: 'free' | 'pro' | 'enterprise';
  onboarded: boolean;
  tenant_id?: string;
  created_at?: string;
  name?: string;
  is_active?: boolean;
  last_login_at?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

interface AuthActions {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => void;
  setUserFromToken: (token: string) => Promise<void>;
  initFromStorage: () => Promise<void>;
  markOnboarded: () => Promise<void>;
  clearError: () => void;
}

type AuthStore = AuthState & AuthActions;

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      loading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ loading: true, error: null });
        try {
          const response = await apiClient.post('/api/auth/login', {
            email,
            password,
          });

          // Backend returns access_token, not token
          const { access_token, user } = response.data;

          set({
            token: access_token,
            user,
            loading: false,
            error: null,
          });

          // Store token in localStorage for axios interceptor
          localStorage.setItem('auth_token', access_token);
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Login failed';
          set({
            loading: false,
            error: errorMessage,
            token: null,
            user: null,
          });
          throw new Error(errorMessage);
        }
      },

      register: async (email: string, password: string, name?: string) => {
        set({ loading: true, error: null });
        try {
          const response = await apiClient.post('/api/auth/register', {
            email,
            password,
            name,
          });

          // Backend returns access_token, not token
          const { access_token, user } = response.data;

          set({
            token: access_token,
            user,
            loading: false,
            error: null,
          });

          // Store token in localStorage for axios interceptor
          localStorage.setItem('auth_token', access_token);
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Registration failed';
          set({
            loading: false,
            error: errorMessage,
            token: null,
            user: null,
          });
          throw new Error(errorMessage);
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          error: null,
        });
        localStorage.removeItem('auth_token');
      },

      setUserFromToken: async (token: string) => {
        set({ loading: true });
        try {
          // Verify token and get user info
          const response = await apiClient.get('/api/auth/me', {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });

          const user = response.data;

          set({
            token,
            user,
            loading: false,
            error: null,
          });

          localStorage.setItem('auth_token', token);
        } catch (error: any) {
          // Token invalid, clear auth
          set({
            token: null,
            user: null,
            loading: false,
            error: null,
          });
          localStorage.removeItem('auth_token');
        }
      },

      initFromStorage: async () => {
        const token = localStorage.getItem('auth_token');
        if (!token) {
          return;
        }

        set({ loading: true });
        try {
          const response = await apiClient.get('/api/auth/me', {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });

          const user = response.data;

          set({
            token,
            user,
            loading: false,
            error: null,
          });
        } catch (error: any) {
          // Token invalid, clear auth
          set({
            token: null,
            user: null,
            loading: false,
            error: null,
          });
          localStorage.removeItem('auth_token');
        }
      },

      markOnboarded: async () => {
        try {
          const response = await apiClient.post('/api/users/onboarded');
          const updatedUser = response.data;

          set((state) => ({
            user: updatedUser ? { ...state.user, ...updatedUser } : state.user,
          }));

          return updatedUser;
        } catch (error: any) {
          console.error('Failed to mark user as onboarded:', error);
          throw error;
        }
      },

      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'auth-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        token: state.token,
        user: state.user,
      }),
      onRehydrateStorage: () => (state) => {
        // On rehydrate, verify token is still valid
        if (state?.token) {
          state.setUserFromToken(state.token);
        }
      },
    }
  )
);

