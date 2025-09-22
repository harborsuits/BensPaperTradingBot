/**
 * Singleton SSE Manager to prevent multiple connections to the same endpoint
 */
import { ImprovedSSEService } from './improvedSSE';

class SSEManager {
  private connections: Map<string, ImprovedSSEService> = new Map();
  
  getConnection(url: string): ImprovedSSEService {
    if (!this.connections.has(url)) {
      const service = new ImprovedSSEService(url);
      this.connections.set(url, service);
      service.connect();
    }
    return this.connections.get(url)!;
  }
  
  destroyConnection(url: string): void {
    const service = this.connections.get(url);
    if (service) {
      service.destroy();
      this.connections.delete(url);
    }
  }
  
  destroyAll(): void {
    this.connections.forEach(service => service.destroy());
    this.connections.clear();
  }
}

export const sseManager = new SSEManager();
