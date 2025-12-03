/**
 * Comprehensive Design System Tokens
 * 
 * Centralized design constants for consistent styling across the application.
 * These tokens define colors, spacing, typography, shadows, and border radius.
 * 
 * All values are mapped to Ant Design theme configuration for global application.
 */

// ============================================================================
// BRAND PALETTE
// ============================================================================

export const COLORS = {
  // Primary colors
  primary: '#2563EB',        // Blue-600 - for main actions and primary UI elements
  primaryDark: '#1D4ED8',    // Blue-700 - for hover states and emphasis

  // Secondary colors
  secondary: '#4B5563',      // Gray-600 - for secondary actions
  secondaryDark: '#374151',  // Gray-700 - for secondary hover states

  // Semantic colors
  success: '#16A34A',        // Green-600 - for positive actions/metrics
  warning: '#EA580C',        // Orange-600 - for warnings
  danger: '#DC2626',         // Red-600 - for errors/negative metrics

  // Background colors
  background: '#FFFFFF',      // White - primary background
  backgroundSecondary: '#F9FAFB', // Gray-50 - secondary background

  // Neutral colors (for compatibility)
  neutralLight: '#F9FAFB',   // Gray-50 - light backgrounds
  neutralDark: '#1F2937',     // Dark text
  info: '#2563EB',            // Same as primary (blue) for Ant Design compatibility
} as const;

// ============================================================================
// TYPOGRAPHY
// ============================================================================

export const TYPOGRAPHY = {
  heading1: {
    fontSize: '32px',
    fontWeight: 700, // bold
    lineHeight: '40px',
    letterSpacing: '-0.02em',
  },
  heading2: {
    fontSize: '24px',
    fontWeight: 600, // semibold
    lineHeight: '32px',
    letterSpacing: '-0.01em',
  },
  body: {
    fontSize: '14px',
    fontWeight: 400, // regular
    lineHeight: '20px',
    letterSpacing: '0',
  },
  caption: {
    fontSize: '12px',
    fontWeight: 400, // regular
    lineHeight: '16px',
    letterSpacing: '0',
  },
  headingLarge: {
    fontSize: '28px',
    fontWeight: 700,
    lineHeight: '36px',
    letterSpacing: '-0.02em',
  },
  headingMedium: {
    fontSize: '20px',
    fontWeight: 600,
    lineHeight: '28px',
    letterSpacing: '-0.01em',
  },
  headingSmall: {
    fontSize: '16px',
    fontWeight: 600,
    lineHeight: '24px',
    letterSpacing: '0',
  },
  bodySmall: {
    fontSize: '13px',
    fontWeight: 400,
    lineHeight: '18px',
    letterSpacing: '0',
  },
} as const;

// ============================================================================
// SPACING
// ============================================================================

export const SPACING = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  '2xl': '48px',
} as const;

// ============================================================================
// SHADOWS
// ============================================================================

export const SHADOWS = {
  sm: '0 1px 2px rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px rgba(0, 0, 0, 0.1)',
  lg: '0 10px 15px rgba(0, 0, 0, 0.1)',
  card: '0 2px 8px rgba(0, 0, 0, 0.08)',
  cardHover: '0 4px 12px rgba(0, 0, 0, 0.12)',
} as const;

// ============================================================================
// BORDER RADIUS
// ============================================================================

export const BORDER_RADIUS = {
  sm: '4px',
  md: '8px',
  lg: '12px',
  full: '9999px',
} as const;

// ============================================================================
// TYPE EXPORTS (for TypeScript)
// ============================================================================

export type ColorKey = keyof typeof COLORS;
export type SpacingKey = keyof typeof SPACING;
export type BorderRadiusKey = keyof typeof BORDER_RADIUS;
export type ShadowKey = keyof typeof SHADOWS;
export type TypographyKey = keyof typeof TYPOGRAPHY;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get a color value by key
 */
export const getColor = (key: ColorKey): string => COLORS[key];

/**
 * Get a spacing value by key
 */
export const getSpacing = (key: SpacingKey): string => SPACING[key];

/**
 * Get a border radius value by key
 */
export const getBorderRadius = (key: BorderRadiusKey): string => BORDER_RADIUS[key];

/**
 * Get a shadow value by key
 */
export const getShadow = (key: ShadowKey): string => SHADOWS[key];

/**
 * Get typography styles by key
 */
export const getTypography = (key: TypographyKey) => TYPOGRAPHY[key];

/**
 * Convert pixel string to number (for Ant Design theme)
 */
export const pxToNumber = (px: string): number => parseInt(px.replace('px', ''), 10);

// ============================================================================
// ANT DESIGN THEME CONFIGURATION
// ============================================================================

import { ThemeConfig } from 'antd';

/**
 * Ant Design theme configuration mapped from design tokens
 * 
 * This configuration is compatible with Ant Design's ConfigProvider
 * and applies the design system globally across all Ant Design components.
 */
export const antdThemeConfig: ThemeConfig = {
  token: {
    // Primary colors from brand palette
    colorPrimary: COLORS.primary,              // Blue-600 (#2563EB)
    colorPrimaryHover: COLORS.primaryDark,     // Blue-700 (#1D4ED8)
    colorPrimaryActive: COLORS.primaryDark,    // Blue-700 (#1D4ED8)

    // Semantic colors
    colorSuccess: COLORS.success,              // Green-600 (#16A34A)
    colorWarning: COLORS.warning,              // Orange-600 (#EA580C)
    colorError: COLORS.danger,                 // Red-600 (#DC2626)
    colorInfo: COLORS.info,                    // Blue-600 (#2563EB)

    // Background colors
    colorBgBase: COLORS.background,            // White (#FFFFFF)
    colorBgContainer: COLORS.backgroundSecondary, // Gray-50 (#F9FAFB)
    colorBgElevated: COLORS.background,         // White (#FFFFFF)

    // Text colors
    colorTextBase: COLORS.neutralDark,         // Dark text (#1F2937)
    colorTextSecondary: COLORS.secondary,      // Gray-600 (#4B5563)

    // Border colors
    colorBorder: '#E5E7EB',                    // Gray-200
    colorBorderSecondary: '#F3F4F6',           // Gray-100

    // Border radius from design tokens
    borderRadius: pxToNumber(BORDER_RADIUS.md), // 8px
    borderRadiusSM: pxToNumber(BORDER_RADIUS.sm), // 4px
    borderRadiusLG: pxToNumber(BORDER_RADIUS.lg), // 12px

    // Typography from design tokens
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
    fontSize: pxToNumber(TYPOGRAPHY.body.fontSize), // 14px
    fontSizeLG: pxToNumber(TYPOGRAPHY.heading2.fontSize), // 24px
    fontSizeSM: pxToNumber(TYPOGRAPHY.caption.fontSize), // 12px
    lineHeight: pxToNumber(TYPOGRAPHY.body.lineHeight) / pxToNumber(TYPOGRAPHY.body.fontSize), // 20/14 ≈ 1.43

    // Font weights
    fontWeightStrong: TYPOGRAPHY.heading1.fontWeight, // 700 (bold)

    // Spacing from design tokens
    padding: pxToNumber(SPACING.md),          // 16px
    paddingXS: pxToNumber(SPACING.xs),        // 4px
    paddingSM: pxToNumber(SPACING.sm),        // 8px
    paddingMD: pxToNumber(SPACING.md),         // 16px
    paddingLG: pxToNumber(SPACING.lg),        // 24px
    paddingXL: pxToNumber(SPACING.xl),        // 32px

    // Component heights
    controlHeight: 40,
    controlHeightSM: 32,
    controlHeightLG: 48,

    // Box shadows (mapped from design tokens)
    boxShadow: SHADOWS.md,                     // Default shadow
    boxShadowSecondary: SHADOWS.sm,           // Light shadow
  },
  components: {
    // Button component customization
    Button: {
      borderRadius: pxToNumber(BORDER_RADIUS.md), // 8px
      fontWeight: 500,
      primaryShadow: SHADOWS.sm,
    },
    // Input component customization
    Input: {
      borderRadius: pxToNumber(BORDER_RADIUS.md), // 8px
      paddingBlock: pxToNumber(SPACING.sm),    // 8px
      paddingInline: pxToNumber(SPACING.md),   // 16px
    },
    // Card component customization
    Card: {
      borderRadius: pxToNumber(BORDER_RADIUS.lg), // 12px
      paddingLG: pxToNumber(SPACING.lg),       // 24px
    },
    // Modal component customization
    Modal: {
      borderRadius: pxToNumber(BORDER_RADIUS.lg), // 12px
      paddingContentHorizontal: pxToNumber(SPACING.lg), // 24px
    },
    // Table component customization
    Table: {
      borderRadius: pxToNumber(BORDER_RADIUS.md), // 8px
    },
  },
};

