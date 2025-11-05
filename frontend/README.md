# Sales Analytics Dashboard

A modern React + TypeScript dashboard for sales analytics.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

3. Start development server:
```bash
npm run dev
```

4. Open http://localhost:5173

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/          # Main views (Dashboard, DataWorkspace, AIAnalyst)
├── services/       # API services
├── store/          # Zustand state management
├── types/          # TypeScript interfaces
├── utils/          # Helper functions
├── styles/         # Global styles and theme
└── App.tsx         # Main app component
```

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Ant Design** - UI components
- **Zustand** - State management
- **React Router** - Routing
- **Recharts** - Data visualization
- **Axios** - HTTP client

