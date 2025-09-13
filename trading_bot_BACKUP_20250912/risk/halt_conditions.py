# trading_bot/risk/halt_conditions.py
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timedelta

@dataclass
class HaltConfig:
    daily_pnl_limit_pct: float = -0.03  # Daily PnL ≤ -3%
    max_gate_breaches_24h: int = 3      # Max gate breaches in 24h
    max_slippage_over_model_bps: float = 10.0  # Realized slippage > model + 10 bps

class HaltManager:
    """
    Manages halt conditions to prevent strategy from running when conditions are poor.
    """

    def __init__(self, config: HaltConfig = None):
        self.config = config or HaltConfig()
        self.daily_pnl = 0.0
        self.daily_start_capital = 0.0
        self.gate_breaches_24h: List[datetime] = []
        self.slippage_alerts: List[Dict[str, Any]] = []
        self.halted = False
        self.halt_reason = ""

    def update_daily_pnl(self, current_capital: float):
        """Update daily PnL tracking"""
        if self.daily_start_capital == 0:
            self.daily_start_capital = current_capital
        else:
            self.daily_pnl = (current_capital - self.daily_start_capital) / self.daily_start_capital

    def check_daily_pnl_halt(self) -> bool:
        """Check if daily PnL limit is exceeded"""
        return self.daily_pnl <= self.config.daily_pnl_limit_pct

    def record_gate_breach(self):
        """Record a gate breach event"""
        now = datetime.now()
        self.gate_breaches_24h.append(now)
        # Keep only breaches from last 24 hours
        cutoff = now - timedelta(hours=24)
        self.gate_breaches_24h = [b for b in self.gate_breaches_24h if b > cutoff]

    def check_gate_breach_halt(self) -> bool:
        """Check if too many gate breaches in 24h"""
        return len(self.gate_breaches_24h) >= self.config.max_gate_breaches_24h

    def record_slippage_alert(self, modeled_bps: float, realized_bps: float, trade_details: Dict[str, Any]):
        """Record slippage that exceeds model by too much"""
        if realized_bps > modeled_bps + self.config.max_slippage_over_model_bps:
            self.slippage_alerts.append({
                'timestamp': datetime.now(),
                'modeled_bps': modeled_bps,
                'realized_bps': realized_bps,
                'excess_bps': realized_bps - modeled_bps,
                'trade_details': trade_details
            })

    def check_slippage_halt(self) -> bool:
        """Check if slippage is consistently too high"""
        # Halt if 2 consecutive slippage alerts
        recent_alerts = [a for a in self.slippage_alerts if (datetime.now() - a['timestamp']).seconds < 3600]  # Last hour
        return len(recent_alerts) >= 2

    def should_halt(self) -> tuple[bool, str]:
        """Check all halt conditions"""
        if self.check_daily_pnl_halt():
            return True, f"Daily PnL limit exceeded: {self.daily_pnl:.1%}"

        if self.check_gate_breach_halt():
            return True, f"Too many gate breaches: {len(self.gate_breaches_24h)} in 24h"

        if self.check_slippage_halt():
            recent = [a for a in self.slippage_alerts if (datetime.now() - a['timestamp']).seconds < 3600]
            return True, f"Excessive slippage: {len(recent)} alerts in last hour"

        return False, ""

    def reset_daily_tracking(self):
        """Reset daily tracking at start of new day"""
        self.daily_pnl = 0.0
        self.daily_start_capital = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current halt status and metrics"""
        return {
            'halted': self.halted,
            'halt_reason': self.halt_reason,
            'daily_pnl_pct': self.daily_pnl,
            'gate_breaches_24h': len(self.gate_breaches_24h),
            'recent_slippage_alerts': len([a for a in self.slippage_alerts if (datetime.now() - a['timestamp']).seconds < 3600])
        }
