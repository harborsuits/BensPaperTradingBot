#!/usr/bin/env python3
"""
Edge Check Script - Analyzes trading performance for statistical edge
Computes win rate, profit factor, and EV after costs
"""

import json
import sys
import argparse
from pathlib import Path

def load_trades_from_files(trade_files):
    """Load trades from multiple JSON files"""
    all_trades = []

    for filepath in trade_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

                # Handle different response formats
                if isinstance(data, dict):
                    if 'items' in data:
                        all_trades.extend(data['items'])
                    elif 'trades' in data:
                        all_trades.extend(data['trades'])
                    else:
                        # Single trade object
                        all_trades.append(data)
                elif isinstance(data, list):
                    all_trades.extend(data)

        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            continue

    return all_trades

def filter_trades_by_strategy(trades, strategy_name=None):
    """Filter trades by strategy if specified"""
    if not strategy_name:
        return trades

    filtered = []
    for trade in trades:
        # Check various fields where strategy might be stored
        trade_strategy = (trade.get('strategy_id') or
                         trade.get('strategy') or
                         trade.get('strategy_name') or '')

        if strategy_name.lower() in trade_strategy.lower():
            filtered.append(trade)

    return filtered

def calculate_edge_metrics(trades, costs_per_trade=0.0004):
    """Calculate key edge metrics from trades"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'ev_per_trade': 0,
            'has_edge': False,
            'confidence': 'insufficient_data'
        }

    wins = []
    losses = []

    for trade in trades:
        # Extract P&L from various possible fields
        pnl = (trade.get('pnl') or
               trade.get('realized_pnl') or
               trade.get('profit_loss') or
               trade.get('profit') or 0)

        # Convert to float if needed
        try:
            pnl = float(pnl)
        except (ValueError, TypeError):
            continue

        # Apply costs
        pnl_after_costs = pnl - costs_per_trade

        if pnl_after_costs > 0:
            wins.append(pnl_after_costs)
        elif pnl_after_costs < 0:
            losses.append(abs(pnl_after_costs))  # Store as positive for averaging

    total_trades = len(wins) + len(losses)
    if total_trades == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'ev_per_trade': 0,
            'has_edge': False,
            'confidence': 'no_valid_trades'
        }

    win_rate = len(wins) / total_trades
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Profit Factor = Gross Wins / Gross Losses
    gross_wins = sum(wins)
    gross_losses = sum(losses)
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Expected Value per trade
    ev_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - costs_per_trade

    # Edge assessment
    has_edge = (profit_factor > 1.10 and
                win_rate > 0.55 and
                ev_per_trade > 0 and
                total_trades >= 50)

    # Confidence assessment
    if total_trades >= 200:
        confidence = 'high'
    elif total_trades >= 100:
        confidence = 'moderate'
    elif total_trades >= 30:
        confidence = 'low'
    else:
        confidence = 'insufficient_data'

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'ev_per_trade': ev_per_trade,
        'gross_wins': gross_wins,
        'gross_losses': gross_losses,
        'costs_per_trade': costs_per_trade,
        'has_edge': has_edge,
        'confidence': confidence
    }

def format_currency(amount):
    """Format amount as currency"""
    return "8.4f"

def print_edge_report(metrics, strategy_name=None):
    """Print formatted edge report"""
    title = f"🎯 EDGE ANALYSIS{f' - {strategy_name}' if strategy_name else ''}"
    print(f"\n{title}")
    print("=" * 60)

    if metrics['total_trades'] == 0:
        print("❌ No trades found to analyze")
        return

    print("📊 PERFORMANCE METRICS:")
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  Win Rate:         {metrics['win_rate']:.1%}")
    print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"  Avg Win:          {format_currency(metrics['avg_win'])}")
    print(f"  Avg Loss:         {format_currency(metrics['avg_loss'])}")
    print(f"  EV per Trade:     {format_currency(metrics['ev_per_trade'])}")
    print(f"  Gross Wins:       {format_currency(metrics['gross_wins'])}")
    print(f"  Gross Losses:     {format_currency(metrics['gross_losses'])}")
    print(f"  Costs per Trade:  {format_currency(metrics['costs_per_trade'])}")

    print("\n🎯 EDGE ASSESSMENT:")
    print(f"  Confidence Level: {metrics['confidence'].upper()}")

    if metrics['has_edge']:
        print("  ✅ STATISTICAL EDGE DETECTED")
        print("     • Profit Factor > 1.10")
        print("     • Win Rate > 55%")
        print("     • Positive Expected Value")
        print("     • Sufficient Sample Size")
    else:
        print("  ❌ NO STATISTICAL EDGE DETECTED")
        reasons = []
        if metrics['profit_factor'] <= 1.10:
            reasons.append("Profit Factor too low")
        if metrics['win_rate'] <= 0.55:
            reasons.append("Win Rate too low")
        if metrics['ev_per_trade'] <= 0:
            reasons.append("Negative Expected Value")
        if metrics['total_trades'] < 50:
            reasons.append("Insufficient sample size")

        for reason in reasons:
            print(f"     • {reason}")

    print("\n💡 RECOMMENDATIONS:")
    if metrics['has_edge'] and metrics['confidence'] in ['high', 'moderate']:
        print("  ✅ Strategy shows robust edge - consider scaling up")
    elif metrics['has_edge'] and metrics['confidence'] == 'low':
        print("  ⚠️  Edge detected but sample size small - continue monitoring")
    elif not metrics['has_edge'] and metrics['total_trades'] >= 100:
        print("  ❌ No edge detected after sufficient testing - consider strategy changes")
    elif metrics['total_trades'] < 100:
        print("  📊 More data needed for reliable edge assessment")
    else:
        print("  🔄 Continue testing and monitoring performance")

def main():
    parser = argparse.ArgumentParser(description='Analyze trading performance for statistical edge')
    parser.add_argument('--trades', nargs='+', help='Trade data JSON files')
    parser.add_argument('--strategy', help='Filter by strategy name')
    parser.add_argument('--costs', type=float, default=0.0004, help='Costs per trade (default: 0.04%)')

    args = parser.parse_args()

    # Auto-discover trade files if none specified
    if not args.trades:
        evidence_dir = Path('evidence')
        if evidence_dir.exists():
            trade_files = list(evidence_dir.glob('*trades*.json')) + \
                         list(evidence_dir.glob('*fills*.json')) + \
                         list(evidence_dir.glob('*orders*.json'))
            args.trades = [str(f) for f in trade_files]
        else:
            print("❌ No trade files specified and no evidence/ directory found")
            print("Usage: python3 edge_check.py --trades file1.json file2.json")
            sys.exit(1)

    print(f"🔍 Analyzing edge from {len(args.trades)} files:")
    for f in args.trades:
        print(f"  • {f}")

    # Load and filter trades
    all_trades = load_trades_from_files(args.trades)
    print(f"\n📊 Loaded {len(all_trades)} total trades")

    if args.strategy:
        filtered_trades = filter_trades_by_strategy(all_trades, args.strategy)
        print(f"📊 Filtered to {len(filtered_trades)} trades for strategy '{args.strategy}'")
        trades_to_analyze = filtered_trades
    else:
        trades_to_analyze = all_trades

    # Calculate edge metrics
    metrics = calculate_edge_metrics(trades_to_analyze, args.costs)

    # Print report
    print_edge_report(metrics, args.strategy)

if __name__ == '__main__':
    main()
