"""Visualization utilities for EvoTrader performance analysis."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.colors import Normalize


def visualize_results(results: Dict[int, Any], output_path: str) -> None:
    """
    Visualize evolution results, focusing on strategy distribution.
    
    Args:
        results: Dictionary of results per generation
        output_path: Path to save the visualization
    """
    generations = sorted(results.keys())
    
    # Extract strategy distribution data
    strategy_names = []
    strategy_data = []
    
    for gen in generations:
        stats = results[gen]
        
        if hasattr(stats, 'to_dict'):
            stats_dict = stats.to_dict()
        else:
            stats_dict = stats
            
        distribution = stats_dict.get('strategy_distribution', {})
        
        # First generation - collect strategy names
        if not strategy_names:
            strategy_names = list(distribution.keys())
            # Initialize data structure
            for _ in strategy_names:
                strategy_data.append([])
                
        # Collect percentages for each strategy
        for i, name in enumerate(strategy_names):
            strategy_data[i].append(distribution.get(name, 0) * 100)  # Convert to percentage
            
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    # Create colormap
    cmap = cm.get_cmap('tab20', len(strategy_names))
    norm = Normalize(vmin=0, vmax=len(strategy_names))
    
    # Plot stacked area chart
    x = generations
    y = np.row_stack(strategy_data)
    ax.stackplot(x, y, labels=strategy_names, colors=[cmap(norm(i)) for i in range(len(strategy_names))])
    
    # Add labels and title
    plt.title('Strategy Distribution Across Generations', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Percentage of Population', fontsize=14)
    
    # Add legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_equity_curves(results: Dict[int, Any], output_path: str) -> None:
    """
    Plot equity curves showing the evolution of bot performance over generations.
    
    Args:
        results: Dictionary of results per generation
        output_path: Path to save the visualization
    """
    generations = sorted(results.keys())
    
    # Extract equity data
    avg_equity = []
    max_equity = []
    min_equity = []
    perc_25 = []
    perc_75 = []
    
    for gen in generations:
        stats = results[gen]
        
        if hasattr(stats, 'to_dict'):
            stats_dict = stats.to_dict()
        else:
            stats_dict = stats
            
        avg_equity.append(stats_dict.get('avg_equity', 0))
        max_equity.append(stats_dict.get('max_equity', 0))
        min_equity.append(stats_dict.get('min_equity', 0))
        perc_25.append(stats_dict.get('perc_25_equity', min_equity[-1]))
        perc_75.append(stats_dict.get('perc_75_equity', max_equity[-1]))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    plt.plot(generations, avg_equity, 'b-', linewidth=2, label='Average Equity')
    plt.plot(generations, max_equity, 'g-', linewidth=2, label='Max Equity')
    plt.fill_between(generations, perc_25, perc_75, alpha=0.2, color='blue', label='25-75 Percentile')
    
    # Add labels and title
    plt.title('Equity Evolution Across Generations', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Equity', fontsize=14)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_performance_report(results: Dict[int, Any], detailed_results_path: str, output_path: str) -> None:
    """
    Generate a detailed HTML performance report.
    
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
    
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EvoTrader Performance Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #0066cc;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .summary-stats {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 15px;
                min-width: 200px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #0066cc;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #0066cc;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #ddd;
            }
            .strategy-distribution {
                margin-bottom: 30px;
            }
            .bar-container {
                height: 25px;
                background-color: #e0e0e0;
                border-radius: 5px;
                margin-top: 10px;
            }
            .generation-summary {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>EvoTrader Performance Report</h1>
            
            <h2>Simulation Summary</h2>
            <div class="summary-stats">
                <div class="stat-card">
                    <div>Generations</div>
                    <div class="stat-value">{num_generations}</div>
                </div>
                <div class="stat-card">
                    <div>Population Size</div>
                    <div class="stat-value">{population_size}</div>
                </div>
                <div class="stat-card">
                    <div>Final Avg Equity</div>
                    <div class="stat-value">${avg_equity:.2f}</div>
                </div>
                <div class="stat-card">
                    <div>Final Max Equity</div>
                    <div class="stat-value">${max_equity:.2f}</div>
                </div>
                <div class="stat-card">
                    <div>Overall Win Rate</div>
                    <div class="stat-value">{win_rate:.1f}%</div>
                </div>
            </div>
            
            <h2>Strategy Distribution</h2>
            <div class="strategy-distribution">
    """.format(
        num_generations=len(generations),
        population_size=len(bot_details),
        avg_equity=final_stats.get('avg_equity', 0),
        max_equity=final_stats.get('max_equity', 0),
        win_rate=final_stats.get('win_rate', 0) * 100
    )
    
    # Add strategy distribution
    strategy_dist = final_stats.get('strategy_distribution', {})
    colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#8F00FF', '#FF5733', '#00BFFF', '#FF00BF']
    color_index = 0
    
    for strategy, percentage in strategy_dist.items():
        color = colors[color_index % len(colors)]
        color_index += 1
        
        html += f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span>{strategy}</span>
                <span>{percentage*100:.1f}%</span>
            </div>
            <div class="bar-container">
                <div style="width: {percentage*100}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
            </div>
        </div>
        """
    
    html += """
            </div>
            
            <h2>Generation Summary</h2>
            <div class="generation-summary">
                <table>
                    <tr>
                        <th>Generation</th>
                        <th>Avg Equity</th>
                        <th>Max Equity</th>
                        <th>Win Rate</th>
                        <th>Avg Trades</th>
                    </tr>
    """
    
    # Add generation summary
    for gen in generations:
        if hasattr(results[gen], 'to_dict'):
            gen_stats = results[gen].to_dict()
        else:
            gen_stats = results[gen]
            
        html += f"""
        <tr>
            <td>{gen}</td>
            <td>${gen_stats.get('avg_equity', 0):.2f}</td>
            <td>${gen_stats.get('max_equity', 0):.2f}</td>
            <td>{gen_stats.get('win_rate', 0)*100:.1f}%</td>
            <td>{gen_stats.get('avg_trades', 0):.1f}</td>
        </tr>
        """
    
    html += """
                </table>
            </div>
            
            <h2>Top Performing Bots</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Bot ID</th>
                    <th>Strategy</th>
                    <th>Equity</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                    <th>Avg Profit/Trade</th>
                    <th>Max Drawdown</th>
                </tr>
    """
    
    # Add top bots
    for i, bot in enumerate(sorted_bots[:20]):  # Top 20 bots
        html += f"""
        <tr>
            <td>{i+1}</td>
            <td>{bot['bot_id']}</td>
            <td>{bot['strategy']}</td>
            <td>${bot['equity']:.2f}</td>
            <td>{bot['win_rate']*100:.1f}%</td>
            <td>{bot['trades']}</td>
            <td>${bot['avg_profit']:.4f}</td>
            <td>{bot['max_drawdown']*100:.1f}%</td>
        </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(output_path, "w") as f:
        f.write(html)


def plot_strategy_performance(results_path: str, output_dir: str) -> None:
    """
    Create individual performance charts for each strategy type.
    
    Args:
        results_path: Path to the detailed results directory
        output_dir: Directory to save the output charts
    """
    try:
        # Load detailed bot results from final generation
        files = os.listdir(results_path)
        gen_files = [f for f in files if f.startswith("gen_") and f.endswith("_bots.json")]
        
        if not gen_files:
            return
            
        # Sort by generation
        gen_files.sort(key=lambda x: int(x.split("_")[1]))
        final_gen_file = gen_files[-1]
        
        with open(os.path.join(results_path, final_gen_file), "r") as f:
            bot_details = json.load(f)
            
        # Group by strategy
        strategy_data = {}
        
        for bot_id, bot_data in bot_details.items():
            strategy = bot_data.get('strategy', 'Unknown')
            
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    'equities': [],
                    'win_rates': [],
                    'avg_profits': [],
                    'drawdowns': [],
                    'trade_counts': []
                }
                
            strategy_data[strategy]['equities'].append(bot_data.get('equity', 0))
            strategy_data[strategy]['win_rates'].append(bot_data.get('win_rate', 0) * 100)
            strategy_data[strategy]['avg_profits'].append(bot_data.get('avg_profit_per_trade', 0))
            strategy_data[strategy]['drawdowns'].append(bot_data.get('max_drawdown', 0) * 100)
            strategy_data[strategy]['trade_counts'].append(bot_data.get('total_trades', 0))
            
        # Create box plots
        for measure in ['equities', 'win_rates', 'avg_profits', 'drawdowns', 'trade_counts']:
            plt.figure(figsize=(14, 8))
            
            data = []
            labels = []
            
            for strategy, metrics in strategy_data.items():
                if metrics[measure]:  # Only add if has data
                    data.append(metrics[measure])
                    labels.append(strategy)
                    
            if not data:
                continue
                
            # Create boxplot
            plt.boxplot(data, labels=labels, patch_artist=True)
            
            # Add labels and title
            measure_name = {
                'equities': 'Final Equity',
                'win_rates': 'Win Rate (%)',
                'avg_profits': 'Average Profit Per Trade',
                'drawdowns': 'Maximum Drawdown (%)',
                'trade_counts': 'Number of Trades'
            }
            
            plt.title(f'Strategy Performance: {measure_name[measure]}', fontsize=16)
            plt.ylabel(measure_name[measure], fontsize=14)
            plt.xlabel('Strategy', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            output_path = os.path.join(output_dir, f"strategy_{measure}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Error generating strategy performance plots: {e}")


if __name__ == "__main__":
    # Simple test
    pass
