/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: 'var(--surface)',
          1: 'var(--surface-1)',
          2: 'var(--surface-2)',
          3: 'var(--surface-3)',
        },
        line: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        primary: {
          DEFAULT: 'var(--primary)',
          dim: 'var(--primary-dim)',
        },
        secondary: {
          DEFAULT: 'var(--secondary)',
          dim: 'var(--secondary-dim)',
        },
        success: {
          DEFAULT: 'var(--success)',
          dim: 'var(--success-dim)',
        },
        danger: {
          DEFAULT: 'var(--danger)',
          dim: 'var(--danger-dim)',
        },
        warning: {
          DEFAULT: 'var(--warning)',
          dim: 'var(--warning-dim)',
        },
        // Legacy aliases kept for backwards compatibility
        ink: {
          DEFAULT: 'var(--surface)',
          1: 'var(--surface-1)',
          2: 'var(--surface-2)',
          3: 'var(--surface-3)',
        },
        accent: {
          DEFAULT: 'var(--primary)',
          dim: 'var(--primary-dim)',
        },
        long: {
          DEFAULT: 'var(--success)',
          dim: 'var(--success-dim)',
        },
        short: {
          DEFAULT: 'var(--danger)',
          dim: 'var(--danger-dim)',
        },
        warn: {
          DEFAULT: 'var(--warning)',
          dim: 'var(--warning-dim)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', '1rem'],
      },
      boxShadow: {
        panel: '0 4px 24px rgba(2, 8, 23, 0.45)',
      },
      animation: {
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};
