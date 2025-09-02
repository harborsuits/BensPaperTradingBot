/**
 * Adds keepalive ping/pong functionality to a WebSocket
 * to detect disconnections and handle them gracefully
 */
export function withKeepalive(ws: WebSocket, intervalMs = 15000) {
  let timerId: number | null = null;
  let alive = true;
  
  // Handle pong responses from server
  const onPong = (evt: MessageEvent) => {
    try {
      const data = JSON.parse(evt.data);
      if (data?.type === "pong") {
        alive = true;
      }
    } catch (err) {
      // Ignore parsing errors for non-JSON messages
    }
  };
  
  // Send ping with timestamp
  const ping = () => {
    // Don't ping if tab is hidden or socket is closed
    if (document.hidden || ws.readyState !== WebSocket.OPEN) return;
    
    // Mark as not alive until we get a pong
    alive = false;
    
    // Send ping
    ws.send(JSON.stringify({ type: "ping", ts: Date.now() }));
    
    // Set timeout to check for pong response
    window.setTimeout(() => {
      if (!alive && ws.readyState === WebSocket.OPEN) {
        console.warn('WebSocket keepalive timeout - reconnecting');
        ws.close(4000, "keepalive timeout");
      }
    }, Math.max(1000, intervalMs - 1000)); // Leave at least 1s for pong
  };
  
  // Setup event listeners and timer
  ws.addEventListener("message", onPong);
  
  // Calculate interval based on visibility
  const getInterval = () => document.hidden ? intervalMs * 3 : intervalMs;
  timerId = window.setInterval(ping, getInterval()) as unknown as number;
  
  // Handle tab visibility changes
  const visibilityHandler = () => {
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = window.setInterval(ping, getInterval()) as unknown as number;
    }
  };
  
  document.addEventListener("visibilitychange", visibilityHandler);
  
  // Return cleanup function
  const cleanup = () => {
    ws.removeEventListener("message", onPong);
    document.removeEventListener("visibilitychange", visibilityHandler);
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
  };
  
  return cleanup;
}
