"""Fixed performance report generator for EvoTrader."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def generate_performance_report(results: Dict[int, Any], detailed_results_path: str, output_path: str) -> None:
    """
    Generate a simplified HTML performance report.
    
    Args:
        results: Dictionary of results per generation
        detailed_results_path: Path to the detailed results directory
        output_path: Path to save the HTML report
    """
    generations = sorted(results.keys())
    final_gen = generations[-1]
    
    # Extract final generation stats
    if hasattr(results[final_gen], 'to_dict'):
        final_stats = results[final_gen].to_dict()
    else:
        final_stats = results[final_gen]
    
    # Load detailed bot results from final generation
    try:
        with open(os.path.join(detailed_results_path, f"gen_{final_gen}_bots.json"), "r") as f:
            bot_details = json.load(f)
    except:
        bot_details = {}
    
    # Sort bots by performance
    sorted_bots = []
    for bot_id, bot_data in bot_details.items():
        sorted_bots.append({
            'bot_id': bot_id,
            'equity': bot_data.get('equity', 0),
            'trades': bot_data.get('total_trades', 0),
            'win_rate': bot_data.get('win_rate', 0),
            'strategy': bot_data.get('strategy', 'Unknown'),
            'avg_profit': bot_data.get('avg_profit_per_trade', 0),
            'max_drawdown': bot_data.get('max_drawdown', 0)
        })
    
    sorted_bots = sorted(sorted_bots, key=lambda x: x['equity'], reverse=True)
    
    # Create a simple HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EvoTrader Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .stat-box {{ 
                display: inline-block;
                margin: 10px;
                padding: 15px;
                background-color: #f7f7f7;
                border-radius: 5px;
                min-width: 150px;
                text-align: center;
            }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        </style>
    </head>
    <body>
        <h1>EvoTrader Performance Report</h1>
        
        <h2>Summary Statistics</h2>
        <div>
            <div class="stat-box">
                <div>Generations</div>
                <div class="stat-value">{len(generations)}</div>
            </div>
            <div class="stat-box">
                <div>Population Size</div>
                <div class="stat-value">{len(bot_details)}</div>
            </div>
            <div class="stat-box">
                <div>Avg Equity</div>
                <div class="stat-value">${final_stats.get('avg_equity', 0):.2f}</div>
            </div>
            <div class="stat-box">
                <div>Max Equity</div>
                <div class="stat-value">${final_stats.get('max_equity', 0):.2f}</div>
            </div>
            <div class="stat-box">
                <div>Win Rate</div>
                <div class="stat-value">{final_stats.get('win_rate', 0) * 100:.1f}%</div>
            </div>
        </div>
        
        <h2>Strategy Distribution</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
    """
    
    # Add strategy distribution
    strat_dist = final_stats.get('strategy_distribution', {})
    total_bots = sum(strat_dist.values())
    
    for strat, count in sorted(strat_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_bots * 100) if total_bots > 0 else 0
        html += f"<tr><td>{strat}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>\n"
        
    html += """
        </table>
        
        <h2>Top Performing Bots</h2>
        <table>
            <tr>
                <th>Bot ID</th>
                <th>Strategy</th>
                <th>Equity</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Avg Profit</th>
                <th>Max Drawdown</th>
            </tr>
    """
    
    # Add top bots
    for bot in sorted_bots[:10]:  # Top 10 bots
        html += f"""
            <tr>
                <td>{bot['bot_id']}</td>
                <td>{bot['strategy']}</td>
                <td>${bot['equity']:.2f}</td>
                <td>{bot['trades']}</td>
                <td>{bot['win_rate']*100:.1f}%</td>
                <td>${bot['avg_profit']:.4f}</td>
                <td>{bot['max_drawdown']*100:.1f}%</td>
            </tr>
        """
        
    html += """
        </table>
    </body>
    </html>
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the report to file
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Performance report saved to: {output_path}")
