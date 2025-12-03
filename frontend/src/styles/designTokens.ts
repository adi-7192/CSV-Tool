/**
 * Design Tokens (Legacy Export)
 * 
 * This file re-exports from design-tokens.ts for backward compatibility.
 * All existing imports from '@/styles/designTokens' will continue to work.
 * 
 * New code should import directly from '@/styles/design-tokens'.
 */

// Re-export everything from the new design-tokens.ts file
export {
  COLORS,
  TYPOGRAPHY,
  SPACING,
  SHADOWS,
  BORDER_RADIUS,
  getColor,
  getSpacing,
  getBorderRadius,
  getShadow,
  getTypography,
  pxToNumber,
  antdThemeConfig,
  type ColorKey,
  type SpacingKey,
  type BorderRadiusKey,
  type ShadowKey,
  type TypographyKey,
} from './design-tokens';
