/**
 * Animation Utilities
 * 
 * Reusable CSS keyframe animations and animation utilities.
 * All animations respect prefers-reduced-motion for accessibility.
 */

export const animations = {
  // Fade In
  fadeIn: `
    @keyframes fadeIn {
      from {
        opacity: 0;
      }
      to {
        opacity: 1;
      }
    }
  `,

  // Fade Out
  fadeOut: `
    @keyframes fadeOut {
      from {
        opacity: 1;
      }
      to {
        opacity: 0;
      }
    }
  `,

  // Slide In From Bottom
  slideInUp: `
    @keyframes slideInUp {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  `,

  // Slide In From Top
  slideInDown: `
    @keyframes slideInDown {
      from {
        opacity: 0;
        transform: translateY(-20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  `,

  // Slide In From Left
  slideInLeft: `
    @keyframes slideInLeft {
      from {
        opacity: 0;
        transform: translateX(-20px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }
  `,

  // Slide In From Right
  slideInRight: `
    @keyframes slideInRight {
      from {
        opacity: 0;
        transform: translateX(20px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }
  `,

  // Scale In
  scaleIn: `
    @keyframes scaleIn {
      from {
        opacity: 0;
        transform: scale(0.95);
      }
      to {
        opacity: 1;
        transform: scale(1);
      }
    }
  `,

  // Scale Out
  scaleOut: `
    @keyframes scaleOut {
      from {
        opacity: 1;
        transform: scale(1);
      }
      to {
        opacity: 0;
        transform: scale(0.95);
      }
    }
  `,

  // Shimmer Effect (for skeleton loaders)
  shimmer: `
    @keyframes shimmer {
      0% {
        background-position: -1000px 0;
      }
      100% {
        background-position: 1000px 0;
      }
    }
  `,

  // Pulse (subtle breathing effect)
  pulse: `
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.7;
      }
    }
  `,

  // Bounce (for icons/empty states)
  bounce: `
    @keyframes bounce {
      0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
      }
      40% {
        transform: translateY(-10px);
      }
      60% {
        transform: translateY(-5px);
      }
    }
  `,

  // Spin (for loading spinners)
  spin: `
    @keyframes spin {
      from {
        transform: rotate(0deg);
      }
      to {
        transform: rotate(360deg);
      }
    }
  `,
};

/**
 * Animation utility classes
 */
export const animationClasses = {
  fadeIn: {
    animation: 'fadeIn 0.3s ease-in forwards',
  },
  fadeOut: {
    animation: 'fadeOut 0.3s ease-out forwards',
  },
  slideInUp: {
    animation: 'slideInUp 0.4s ease-out forwards',
  },
  slideInDown: {
    animation: 'slideInDown 0.4s ease-out forwards',
  },
  slideInLeft: {
    animation: 'slideInLeft 0.4s ease-out forwards',
  },
  slideInRight: {
    animation: 'slideInRight 0.4s ease-out forwards',
  },
  scaleIn: {
    animation: 'scaleIn 0.3s ease-out forwards',
  },
  scaleOut: {
    animation: 'scaleOut 0.3s ease-out forwards',
  },
  shimmer: {
    background: 'linear-gradient(90deg, #F3F4F6 0%, #E5E7EB 50%, #F3F4F6 100%)',
    backgroundSize: '1000px 100%',
    animation: 'shimmer 1.5s infinite',
  },
  pulse: {
    animation: 'pulse 2s ease-in-out infinite',
  },
  bounce: {
    animation: 'bounce 1s ease-in-out',
  },
};

/**
 * Stagger animation delay helper
 * @param index - Item index
 * @param delay - Delay per item in seconds (default: 0.1)
 * @param maxDelay - Maximum total delay in seconds (default: 0.5)
 */
export const getStaggerDelay = (index: number, delay: number = 0.1, maxDelay: number = 0.5): number => {
  const calculatedDelay = index * delay;
  return Math.min(calculatedDelay, maxDelay);
};

/**
 * CSS string with all animations
 * Include this in your global CSS or inject it
 */
export const allAnimationsCSS = `
  ${animations.fadeIn}
  ${animations.fadeOut}
  ${animations.slideInUp}
  ${animations.slideInDown}
  ${animations.slideInLeft}
  ${animations.slideInRight}
  ${animations.scaleIn}
  ${animations.scaleOut}
  ${animations.shimmer}
  ${animations.pulse}
  ${animations.bounce}
  ${animations.spin}

  /* Respect reduced motion preference */
  @media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }
`;

