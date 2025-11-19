import { create } from 'zustand';

interface UIStore {
  // Navigation state
  currentView: 'dashboard' | 'workspace' | 'analyst';
  // UI state
  sidebarCollapsed: boolean;
  selectedSKU: string | null;
  // Modal/Drawer state
  uploadDrawerOpen: boolean;
  // Methods
  setView: (view: 'dashboard' | 'workspace' | 'analyst') => void;
  setSelectedSKU: (sku: string | null) => void;
  toggleSidebar: () => void;
  setUploadDrawerOpen: (open: boolean) => void;
}

export const useUIStore = create<UIStore>((set) => ({
  currentView: 'dashboard',
  sidebarCollapsed: false,
  selectedSKU: null,
  uploadDrawerOpen: false,
  setView: (view) => set({ currentView: view }),
  setSelectedSKU: (sku) => set({ selectedSKU: sku }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setUploadDrawerOpen: (open) => set({ uploadDrawerOpen: open }),
}));
