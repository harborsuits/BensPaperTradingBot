"""
EvoTrader Component for BensBot Dashboard
Displays genetic algorithm data including population, fitness trends, and controls
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Get genetic algorithm data from MongoDB
def get_genetic_data(db):
    """
    Retrieve genetic algorithm data from MongoDB
    Returns population data, fitness history, and configuration
    """
    if db is None:
        return generate_mock_genetic_data()
    
    try:
        # Get population data
        population_docs = list(db.genetic_population.find({}))
        
        # Get fitness history
        fitness_docs = list(db.genetic_fitness.find({}).sort("generation", 1))
        
        # Get configuration
        config_doc = db.genetic_config.find_one({})
        
        if population_docs and fitness_docs and config_doc:
            # Convert to DataFrames
            population_df = pd.DataFrame(population_docs)
            fitness_df = pd.DataFrame(fitness_docs)
            
            return {
                "population": population_df,
                "fitness": fitness_df,
                "config": config_doc
            }
        else:
            # If any data is missing, generate mock data
            return generate_mock_genetic_data()
    except Exception as e:
        st.error(f"Error retrieving genetic data: {e}")
        return generate_mock_genetic_data()

# Generate mock genetic algorithm data
def generate_mock_genetic_data():
    """Generate synthetic genetic algorithm data for development and testing"""
    # Current generation
    current_gen = random.randint(5, 20)
    
    # Generate population data
    population_size = 30
    population_data = []
    
    # Strategy templates with parameters ranges
    param_templates = [
        {
            "name": "SMA Period",
            "min": 5,
            "max": 50,
            "type": "int",
            "distribution": "uniform"
        },
        {
            "name": "RSI Period",
            "min": 2,
            "max": 30,
            "type": "int",
            "distribution": "uniform"
        },
        {
            "name": "RSI Overbought",
            "min": 65,
            "max": 85,
            "type": "int",
            "distribution": "uniform"
        },
        {
            "name": "RSI Oversold",
            "min": 15,
            "max": 35,
            "type": "int",
            "distribution": "uniform"
        },
        {
            "name": "Stop Loss",
            "min": 0.5,
            "max": 5.0,
            "type": "float",
            "distribution": "uniform"
        },
        {
            "name": "Take Profit",
            "min": 1.0,
            "max": 10.0,
            "type": "float",
            "distribution": "uniform"
        },
        {
            "name": "Position Size",
            "min": 5.0,
            "max": 25.0,
            "type": "float",
            "distribution": "uniform"
        }
    ]
    
    # Generate individual parameter values
    for i in range(population_size):
        # Base fitness diminishes slightly for lower-ranked individuals
        base_fitness = 100 - (i * 100 / population_size)
        
        # Random variation in fitness
        fitness = base_fitness + random.uniform(-5, 5)
        
        # Generate parameters
        params = {}
        for param in param_templates:
            if param["type"] == "int":
                value = random.randint(param["min"], param["max"])
            else:  # float
                value = round(random.uniform(param["min"], param["max"]), 2)
            
            params[param["name"]] = value
        
        # Create individual
        individual = {
            "id": f"gen{current_gen}_ind{i+1:02d}",
            "generation": current_gen,
            "rank": i+1,
            "fitness": round(fitness, 2),
            "win_rate": round(random.uniform(45, 75), 1),
            "profit_factor": round(random.uniform(1.0, 2.5), 2),
            "sharpe_ratio": round(random.uniform(0.8, 2.2), 2),
            "max_drawdown": round(random.uniform(-3, -15), 1),
            "trades": random.randint(50, 300),
            "parameters": params,
            "parent_ids": [] if i < 5 else [f"gen{current_gen-1}_ind{random.randint(1, 10):02d}", 
                                          f"gen{current_gen-1}_ind{random.randint(1, 10):02d}"],
            "mutation_rate": 0.0 if i < 3 else round(random.uniform(0.05, 0.3), 2),
            "created_at": (datetime.datetime.now() - 
                          datetime.timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        population_data.append(individual)
    
    # Generate fitness history
    fitness_history = []
    
    # Start with a base fitness level
    best_fitness = 70.0
    avg_fitness = 45.0
    worst_fitness = 20.0
    
    # Increasing trend with some noise
    for gen in range(1, current_gen + 1):
        # Add some randomness to the trend
        best_delta = random.uniform(0, 2)
        avg_delta = random.uniform(-0.5, 1.5)
        worst_delta = random.uniform(-1, 1)
        
        # Update fitness values
        best_fitness = min(100, best_fitness + best_delta)
        avg_fitness = min(95, avg_fitness + avg_delta)
        worst_fitness = min(90, max(0, worst_fitness + worst_delta))
        
        # Ensure the order best > avg > worst is maintained
        avg_fitness = min(avg_fitness, best_fitness - 5)
        worst_fitness = min(worst_fitness, avg_fitness - 5)
        
        # Add entry
        fitness_history.append({
            "generation": gen,
            "best_fitness": round(best_fitness, 2),
            "average_fitness": round(avg_fitness, 2),
            "worst_fitness": round(worst_fitness, 2),
            "population_size": population_size,
            "timestamp": (datetime.datetime.now() - 
                         datetime.timedelta(days=current_gen - gen)).strftime("%Y-%m-%d")
        })
    
    # Create configuration
    config = {
        "population_size": population_size,
        "generations": current_gen,
        "mutation_rate": 0.2,
        "crossover_rate": 0.7,
        "elitism_count": 3,
        "tournament_size": 5,
        "fitness_metric": "composite",
        "parameter_constraints": [p for p in param_templates],
        "status": "running" if random.random() > 0.3 else "paused",
        "start_time": (datetime.datetime.now() - 
                      datetime.timedelta(days=current_gen)).strftime("%Y-%m-%d %H:%M:%S"),
        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to pandas DataFrame for population
    population_df = pd.DataFrame(population_data)
    
    # Convert to pandas DataFrame for fitness history
    fitness_df = pd.DataFrame(fitness_history)
    
    return {
        "population": population_df,
        "fitness": fitness_df,
        "config": config
    }

# Render genetic algorithm controls
def render_genetic_controls(genetic_data):
    """Render controls for the genetic algorithm operation"""
    config = genetic_data.get("config", {})
    
    # Status indicator and controls
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Status indicator
        status = config.get("status", "unknown")
        
        if status == "running":
            status_color = "green"
            status_text = "Running"
        elif status == "paused":
            status_color = "orange"
            status_text = "Paused"
        else:
            status_color = "gray"
            status_text = status.title()
        
        # Generations info
        current_gen = config.get("generations", 0)
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
            <h3 style="margin-top: 0;">Genetic Algorithm Status</h3>
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Current Status</p>
            <p style="font-size: 1.5rem; font-weight: 700; color: {status_color}; margin-bottom: 15px;">{status_text}</p>
            
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Current Generation</p>
            <p style="font-size: 1.5rem; font-weight: 700; margin-bottom: 15px;">{current_gen}</p>
            
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Population Size</p>
            <p style="font-size: 1.5rem; font-weight: 700; margin-bottom: 15px;">{config.get("population_size", 0)}</p>
            
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Last Updated</p>
            <p style="font-size: 1.0rem; margin-bottom: 0;">{config.get("last_update", "Unknown")}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Control panel
        st.markdown(f"""<h3>Algorithm Controls</h3>""", unsafe_allow_html=True)
        
        # Generation slider
        generation_slider = st.slider(
            "Maximum Generations", 
            min_value=current_gen, 
            max_value=current_gen + 50,
            value=current_gen + 10,
            step=5
        )
        
        # Parameter controls
        col_a, col_b = st.columns(2)
        
        with col_a:
            mutation_rate = st.slider(
                "Mutation Rate", 
                min_value=0.0, 
                max_value=1.0,
                value=config.get("mutation_rate", 0.2),
                step=0.05,
                format="%.2f"
            )
            
            tournament_size = st.slider(
                "Tournament Size", 
                min_value=2, 
                max_value=10,
                value=config.get("tournament_size", 5)
            )
        
        with col_b:
            crossover_rate = st.slider(
                "Crossover Rate", 
                min_value=0.0, 
                max_value=1.0,
                value=config.get("crossover_rate", 0.7),
                step=0.05,
                format="%.2f"
            )
            
            elitism_count = st.slider(
                "Elitism Count", 
                min_value=0, 
                max_value=10,
                value=config.get("elitism_count", 3)
            )
        
        # Action buttons
        if status == "running":
            if st.button("Pause Evolution", key="pause_button"):
                st.info("This would pause the genetic algorithm in a real implementation.")
        else:
            if st.button("Resume Evolution", key="resume_button"):
                st.success("This would resume the genetic algorithm in a real implementation.")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("Apply Settings", key="apply_settings"):
                st.success("This would apply the new settings in a real implementation.")
        
        with col_d:
            if st.button("Reset Population", key="reset_population"):
                st.warning("This would reset the population in a real implementation.")

# Render fitness trend chart
def render_fitness_trend(fitness_df):
    """Render chart showing fitness trends over generations"""
    st.subheader("Fitness Evolution")
    
    # Create the plot
    fig = go.Figure()
    
    # Add traces for best, average, and worst fitness
    fig.add_trace(
        go.Scatter(
            x=fitness_df["generation"],
            y=fitness_df["best_fitness"],
            mode="lines+markers",
            name="Best Fitness",
            line=dict(width=3, color="#2ca02c")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=fitness_df["generation"],
            y=fitness_df["average_fitness"],
            mode="lines+markers",
            name="Average Fitness",
            line=dict(width=2, color="#1f77b4")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=fitness_df["generation"],
            y=fitness_df["worst_fitness"],
            mode="lines+markers",
            name="Worst Fitness",
            line=dict(width=2, color="#d62728")
        )
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Render population table
def render_population_table(population_df):
    """Render table of current population individuals with metrics"""
    st.subheader("Population Table")
    
    # Filter out unnecessary columns for display
    display_cols = [
        "id", "rank", "fitness", "win_rate", "profit_factor", 
        "sharpe_ratio", "max_drawdown", "trades", "mutation_rate"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in population_df.columns:
            population_df[col] = "N/A"
    
    # Create display DataFrame
    display_df = population_df[display_cols].copy()
    
    # Rename columns for display
    display_df.columns = [
        "ID", "Rank", "Fitness", "Win Rate", "Profit Factor", 
        "Sharpe", "Max DD", "Trades", "Mutation Rate"
    ]
    
    # Format columns
    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)
    display_df["Max DD"] = display_df["Max DD"].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)
    display_df["Mutation Rate"] = display_df["Mutation Rate"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # Add coloring to highlight top individuals
    def highlight_top_performers(s):
        is_top = s.name < 5  # Top 5 individuals
        return ['background-color: rgba(0,200,0,0.2)' if is_top else '' for _ in s]
    
    # Display table
    st.dataframe(
        display_df.style.apply(highlight_top_performers, axis=1),
        use_container_width=True
    )
    
    # Individual selection for detailed view
    st.subheader("Individual Details")
    selected_id = st.selectbox(
        "Select an individual to view details",
        [""] + population_df["id"].tolist()
    )
    
    # Show detailed view if an individual is selected
    if selected_id:
        selected_individual = population_df[population_df["id"] == selected_id].iloc[0]
        render_individual_details(selected_individual)

# Render individual strategy details
def render_individual_details(individual):
    """Render detailed view for a selected individual strategy"""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Individual info
        st.markdown(f"""
        <h3 style="margin-bottom: 0;">{individual['id']}</h3>
        <p style="color: #666; margin-top: 0;">Rank: {individual['rank']} | Fitness: {individual['fitness']}</p>
        """, unsafe_allow_html=True)
        
        # Parameter table
        st.subheader("Strategy Parameters")
        
        params = individual.get('parameters', {})
        if isinstance(params, dict) and params:
            # Convert to DataFrame for display
            params_df = pd.DataFrame({
                "Parameter": list(params.keys()),
                "Value": list(params.values())
            })
            
            st.dataframe(params_df, use_container_width=True)
        else:
            st.info("No parameter data available.")
        
        # Parentage info
        parent_ids = individual.get('parent_ids', [])
        if parent_ids:
            st.subheader("Parent Strategies")
            for parent_id in parent_ids:
                st.markdown(f"- {parent_id}")
        else:
            st.subheader("Parent Strategies")
            st.markdown("Elite strategy (no parents)")
    
    with col2:
        # Performance metrics
        st.subheader("Performance Metrics")
        
        metrics = [
            {"key": "win_rate", "name": "Win Rate", "format": "{}%"},
            {"key": "profit_factor", "name": "Profit Factor", "format": "{:.2f}"},
            {"key": "sharpe_ratio", "name": "Sharpe Ratio", "format": "{:.2f}"},
            {"key": "max_drawdown", "name": "Max Drawdown", "format": "{}%"},
            {"key": "trades", "name": "Total Trades", "format": "{}"}
        ]
        
        metrics_html = ""
        for metric in metrics:
            value = individual.get(metric['key'], 'N/A')
            if isinstance(value, (int, float)):
                formatted_value = metric['format'].format(value)
            else:
                formatted_value = value
            
            metrics_html += f"""
            <div style="border-bottom: 1px solid #eee; padding: 8px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1.0rem; color: #666;">{metric['name']}</span>
                    <span style="font-size: 1.0rem; font-weight: 600;">{formatted_value}</span>
                </div>
            </div>
            """
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
            {metrics_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        st.subheader("Actions")
        
        if st.button("Deploy to Paper Trading", key=f"deploy_paper_{individual['id']}"):
            st.success("This would deploy the strategy to paper trading in a real implementation.")
        
        if st.button("Deploy to Live Trading", key=f"deploy_live_{individual['id']}"):
            st.warning("This would deploy the strategy to live trading in a real implementation.")
        
        if st.button("Create Variant", key=f"create_variant_{individual['id']}"):
            st.info("This would create a variant of this strategy in a real implementation.")

# Main render function for this component
def render(db):
    """Main render function for the EvoTrader (Genetic Module) section"""
    
    # Get genetic algorithm data
    genetic_data = get_genetic_data(db)
    
    # Display top-level controls
    render_genetic_controls(genetic_data)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display fitness trend
    render_fitness_trend(genetic_data.get("fitness", pd.DataFrame()))
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display population table and individual details
    render_population_table(genetic_data.get("population", pd.DataFrame()))
