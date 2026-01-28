import { useEffect, useCallback } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useTradingStore } from './store';
import { useWebSocket } from './hooks/useWebSocket';
import type { WSMessage } from './types';

// Layout
import Layout from './components/Layout';

// Pages
import Dashboard from './pages/Dashboard';
import Positions from './pages/Positions';
import Signals from './pages/Signals';
import Risk from './pages/Risk';
import Settings from './pages/Settings';

function App() {
  const {
    fetchAll,
    updatePortfolio,
    updatePositions,
    addSignal,
    updateRisk,
    addAlert,
    addOrder,
  } = useTradingStore();

  const handleWSMessage = useCallback(
    (message: WSMessage) => {
      switch (message.type) {
        case 'portfolio':
          updatePortfolio(message.data as Parameters<typeof updatePortfolio>[0]);
          break;
        case 'positions':
          updatePositions(message.data as Parameters<typeof updatePositions>[0]);
          break;
        case 'signal':
          addSignal(message.data as Parameters<typeof addSignal>[0]);
          break;
        case 'risk':
          updateRisk(message.data as Parameters<typeof updateRisk>[0]);
          break;
        case 'alert':
          addAlert(message.data as Parameters<typeof addAlert>[0]);
          break;
        case 'trade':
          addOrder(message.data as Parameters<typeof addOrder>[0]);
          break;
        default:
          console.log('Unknown message type:', message.type);
      }
    },
    [updatePortfolio, updatePositions, addSignal, updateRisk, addAlert, addOrder]
  );

  const { isConnected } = useWebSocket({
    onMessage: handleWSMessage,
    onOpen: () => console.log('WebSocket connected'),
    onClose: () => console.log('WebSocket disconnected'),
  });

  useEffect(() => {
    fetchAll();
    // Refresh every 30 seconds
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  return (
    <Layout isConnected={isConnected}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/positions" element={<Positions />} />
        <Route path="/signals" element={<Signals />} />
        <Route path="/risk" element={<Risk />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default App;
