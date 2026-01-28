# InvestLLM Trading Platform

Production-ready algorithmic trading platform with React 19 + TypeScript frontend and Python FastAPI backend.

## Features

- **Real-time Dashboard**: Portfolio overview, positions, P&L tracking
- **Paper/Live Trading**: Switch between modes seamlessly
- **Signal Management**: View and execute AI-generated trading signals
- **Risk Monitoring**: Real-time risk metrics with circuit breaker protection
- **WebSocket Updates**: Live portfolio and signal updates

## Tech Stack

### Frontend
- React 19 with TypeScript
- Vite for fast development
- Tailwind CSS for styling
- Zustand for state management
- Recharts for data visualization

### Backend
- FastAPI with async support
- WebSocket for real-time updates
- Pydantic for data validation

### Infrastructure
- Docker multi-stage build
- Nginx reverse proxy
- Google Cloud Run deployment

## Quick Start

### Local Development

1. **Start Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

2. **Start Frontend**
```bash
cd frontend
npm install
npm run dev
```

3. **Access the app**: http://localhost:3000

### Docker Development

```bash
cd web
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Build

```bash
cd web
docker-compose up --build
```

Access at: http://localhost:8080

## Google Cloud Deployment

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed

### Deploy

1. **Set up secrets**
```bash
cd deploy
./setup-secrets.sh
```

2. **Deploy to Cloud Run**
```bash
./deploy.sh YOUR_PROJECT_ID asia-south1
```

### CI/CD with Cloud Build

The `cloudbuild.yaml` can be used for automated deployments:

```bash
gcloud builds submit --config=deploy/cloudbuild.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRADING_MODE` | `paper` or `live` | `paper` |
| `INITIAL_CAPITAL` | Starting capital (INR) | `1000000` |
| `ZERODHA_API_KEY` | Kite Connect API key | - |
| `ZERODHA_API_SECRET` | Kite Connect secret | - |
| `FIRECRAWL_API_KEY` | Firecrawl API key | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |

### Risk Limits

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_POSITION_PCT` | Max % per position | `0.05` (5%) |
| `MAX_SECTOR_PCT` | Max % per sector | `0.20` (20%) |
| `MAX_DRAWDOWN_PCT` | Circuit breaker trigger | `0.15` (15%) |
| `DAILY_LOSS_LIMIT_PCT` | Daily loss limit | `0.03` (3%) |

## API Endpoints

### Portfolio
- `GET /api/portfolio/` - Portfolio summary
- `GET /api/portfolio/positions` - Open positions
- `GET /api/portfolio/performance` - Performance history

### Trading
- `POST /api/trading/order` - Place order
- `POST /api/trading/close/{symbol}` - Close position
- `POST /api/trading/start` - Start trading
- `POST /api/trading/stop` - Stop trading

### Signals
- `GET /api/signals/` - Get signals
- `POST /api/signals/{id}/execute` - Execute signal
- `POST /api/signals/{id}/dismiss` - Dismiss signal

### Risk
- `GET /api/risk/` - Risk metrics
- `GET /api/risk/alerts` - Risk alerts

### Settings
- `GET /api/settings/` - Get settings
- `POST /api/settings/mode` - Switch mode
- `POST /api/settings/reset` - Reset paper account

### WebSocket
- `WS /ws` - Real-time updates (portfolio, signals, risk)

## Project Structure

```
web/
├── frontend/                 # React 19 + TypeScript
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── pages/           # Page components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API services
│   │   ├── store/           # Zustand store
│   │   ├── types/           # TypeScript types
│   │   └── utils/           # Utility functions
│   └── package.json
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── routes/          # API routes
│   │   ├── services/        # Business logic
│   │   └── core/            # Config, WebSocket
│   └── requirements.txt
├── deploy/                   # Deployment scripts
│   ├── deploy.sh            # Cloud Run deploy
│   ├── setup-secrets.sh     # Secret manager setup
│   └── cloudbuild.yaml      # Cloud Build config
├── Dockerfile               # Production image
├── docker-compose.yml       # Production compose
├── docker-compose.dev.yml   # Development compose
├── nginx.conf               # Nginx configuration
└── supervisord.conf         # Process manager
```

## License

MIT
