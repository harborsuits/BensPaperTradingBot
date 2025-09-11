import axios from "axios";
import { useSafetyStatus } from "@/hooks/useSafetyStatus";
import { Card } from "@/components/ui/Card";

export default function SafetyPage(){
  const { data, refetch, isFetching } = useSafetyStatus();
  const act = async (f:()=>Promise<any>) => { await f(); await refetch(); };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'ready': return 'text-green-400';
      case 'active': return 'text-yellow-400';
      case 'normal': return 'text-blue-400';
      case 'triggered': return 'text-red-400';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <div className="w-full min-h-screen bg-background">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-foreground">Safety Controls</h1>
            <div className={`text-sm px-3 py-1 rounded-full ${isFetching ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400'}`}>
              {isFetching ? 'Updating...' : 'Live'}
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Trading safety systems and emergency controls
          </p>
        </div>

        {/* Main Dashboard Content - Vertical Flow */}
        <div className="dashboard-container">
          {/* Safety Status Section */}
          <div className="dashboard-section">
            <Card className="card">
              <div className="card-header">
                <h3 className="card-title">System Status</h3>
                <div className="card-subtle">Current safety system state</div>
              </div>
              <div className="card-content">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-card/30 rounded-lg border border-border/50">
                    <p className="text-sm text-muted-foreground mb-2">Trading Mode</p>
                    <p className={`text-xl font-bold ${getStatusColor(data?.mode)}`}>
                      {data?.mode ?? "PAPER"}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-card/30 rounded-lg border border-border/50">
                    <p className="text-sm text-muted-foreground mb-2">Kill Switch</p>
                    <p className={`text-xl font-bold ${getStatusColor(data?.killSwitch)}`}>
                      {data?.killSwitch ?? "READY"}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-card/30 rounded-lg border border-border/50">
                    <p className="text-sm text-muted-foreground mb-2">Circuit Breaker</p>
                    <p className={`text-xl font-bold ${getStatusColor(data?.circuitBreaker)}`}>
                      {data?.circuitBreaker ?? "NORMAL"}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-card/30 rounded-lg border border-border/50">
                    <p className="text-sm text-muted-foreground mb-2">Cooldown</p>
                    <p className={`text-xl font-bold ${getStatusColor(data?.cooldown)}`}>
                      {data?.cooldown ?? "READY"}
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Controls Section */}
          <div className="dashboard-section">
            <Card className="card">
              <div className="card-header">
                <h3 className="card-title">Emergency Controls</h3>
                <div className="card-subtle">Critical safety actions</div>
              </div>
              <div className="card-content">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  <button
                    className="flex items-center justify-center gap-3 p-4 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg transition-colors text-red-400 font-medium"
                    onClick={()=>act(()=>axios.post("/api/safety/emergency-stop"))}
                  >
                    <div className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center">
                      <span className="text-red-400 font-bold text-sm">!</span>
                    </div>
                    <span>Emergency Stop</span>
                  </button>

                  <button
                    className="flex items-center justify-center gap-3 p-4 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-lg transition-colors text-blue-400 font-medium"
                    onClick={()=>act(()=>axios.post("/api/safety/trading-mode",{mode:"paper"}))}
                  >
                    <div className="w-8 h-8 bg-blue-500/20 rounded-full flex items-center justify-center">
                      <span className="text-blue-400 font-bold text-sm">P</span>
                    </div>
                    <span>Set PAPER Mode</span>
                  </button>

                  <button
                    className="flex items-center justify-center gap-3 p-4 bg-green-500/10 hover:bg-green-500/20 border border-green-500/30 rounded-lg transition-colors text-green-400 font-medium"
                    onClick={()=>act(()=>axios.post("/api/safety/trading-mode",{mode:"live"}))}
                  >
                    <div className="w-8 h-8 bg-green-500/20 rounded-full flex items-center justify-center">
                      <span className="text-green-400 font-bold text-sm">L</span>
                    </div>
                    <span>Set LIVE Mode</span>
                  </button>
                </div>

                <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-yellow-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-yellow-400 text-xs">⚠</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-yellow-400 mb-1">Important Safety Notice</h4>
                      <p className="text-sm text-muted-foreground">
                        Emergency Stop will immediately halt all trading operations and freeze the system.
                        Use this only in critical situations. Paper mode enables simulated trading without real capital.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Additional Safety Information */}
          <div className="dashboard-section">
            <Card className="card">
              <div className="card-header">
                <h3 className="card-title">Safety Guidelines</h3>
                <div className="card-subtle">Best practices for safe trading</div>
              </div>
              <div className="card-content">
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="p-4 bg-card/30 rounded-lg border border-border/50">
                      <h4 className="font-medium text-foreground mb-2">🛡️ Risk Management</h4>
                      <p className="text-sm text-muted-foreground">
                        Always monitor position sizes and maintain proper risk limits.
                        The circuit breaker automatically pauses trading if losses exceed thresholds.
                      </p>
                    </div>
                    <div className="p-4 bg-card/30 rounded-lg border border-border/50">
                      <h4 className="font-medium text-foreground mb-2">🔄 System Monitoring</h4>
                      <p className="text-sm text-muted-foreground">
                        Regularly check system status and be prepared to use emergency controls.
                        The kill switch provides an immediate system-wide halt.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
