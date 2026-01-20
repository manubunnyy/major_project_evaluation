import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def render_professional_table(data, output_path):
    """
    Black & white professional table with bold highlighting for best scores.
    For each model, compares baseline vs agentic and bolds the winner.
    """
    df_plot = data.copy()
    
    # Clean Model Names
    df_plot['Model'] = df_plot['Model'].str.replace(" (Base)", "").str.replace(" (Agent)", "").str.replace("openai/", "").str.replace("meta-llama/", "").str.replace("qwen/", "").str.replace("moonshotai/", "")
    
    # Figure setup - professional A4 ratio
    fig, ax = plt.subplots(figsize=(14, len(data) * 0.7))
    ax.axis('off')
    
    # Headers
    cols = ["Model", "Process", "GPQA", "IFEval", "ChatDoc", "Average"]
    
    # Build table data and determine winners
    table_data = []
    cell_weights = []  # Track which cells should be bold
    
    # Group by model to compare baseline vs agentic
    models_seen = {}
    
    for i, row in data.iterrows():
        m_name = row['Model'].replace(" (Base)", "").replace(" (Agent)", "").replace("openai/", "").replace("meta-llama/", "").replace("qwen/", "").replace("moonshotai/", "")
        proc = row['Process']
        
        # Store scores for comparison
        scores = {
            'GPQA': float(row['GPQA']),
            'IFEval': float(row['IFEval']),
            'ChatDoc': float(row['ChatDoc']),
            'Average': float(row['Average'])
        }
        
        if m_name not in models_seen:
            models_seen[m_name] = {}
        
        models_seen[m_name][proc] = scores
    
    # Now build table with bold formatting
    for i, row in data.iterrows():
        m_name = row['Model'].replace(" (Base)", "").replace(" (Agent)", "").replace("openai/", "").replace("meta-llama/", "").replace("qwen/", "").replace("moonshotai/", "")
        proc = row['Process']
        
        # Check if this model has both baseline and agentic
        if m_name in models_seen and len(models_seen[m_name]) == 2:
            baseline_scores = models_seen[m_name].get('Baseline', {})
            agentic_scores = models_seen[m_name].get('Agentic', {})
            
            # Determine winners for each column
            row_weights = [False, False, False, False, False, False]  # Model, Process, GPQA, IFEval, ChatDoc, Avg
            
            if baseline_scores and agentic_scores:
                # Check each benchmark
                if proc == 'Baseline':
                    if baseline_scores['GPQA'] >= agentic_scores['GPQA']:
                        row_weights[2] = True  # Bold GPQA
                        row_weights[1] = True  # Bold Process
                    if baseline_scores['IFEval'] >= agentic_scores['IFEval']:
                        row_weights[3] = True  # Bold IFEval
                        row_weights[1] = True  # Bold Process
                    if baseline_scores['ChatDoc'] >= agentic_scores['ChatDoc']:
                        row_weights[4] = True  # Bold ChatDoc
                        row_weights[1] = True  # Bold Process
                    if baseline_scores['Average'] >= agentic_scores['Average']:
                        row_weights[5] = True  # Bold Average
                        row_weights[1] = True  # Bold Process
                else:  # Agentic
                    if agentic_scores['GPQA'] > baseline_scores['GPQA']:
                        row_weights[2] = True
                        row_weights[1] = True
                    if agentic_scores['IFEval'] > baseline_scores['IFEval']:
                        row_weights[3] = True
                        row_weights[1] = True
                    if agentic_scores['ChatDoc'] > baseline_scores['ChatDoc']:
                        row_weights[4] = True
                        row_weights[1] = True
                    if agentic_scores['Average'] > baseline_scores['Average']:
                        row_weights[5] = True
                        row_weights[1] = True
            
            cell_weights.append(row_weights)
        else:
            # No comparison possible
            cell_weights.append([False] * 6)
        
        vals = [m_name, proc, f"{row['GPQA']:.1f}", f"{row['IFEval']:.1f}", f"{row['ChatDoc']:.1f}", f"{row['Average']:.1f}"]
        table_data.append(vals)
    
    # Create Table
    table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    
    # Professional Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(1.0)
        cell.set_edgecolor('#333333')
        
        if i == 0:  # Header
            cell.set_facecolor('#000000')
            cell.set_text_props(color='white', weight='bold', size=12)
            cell.set_height(0.12)
        else:
            # Data Rows - alternating light gray for readability
            if i % 2 == 0:
                cell.set_facecolor('#f5f5f5')
            else:
                cell.set_facecolor('#ffffff')
            
            # Apply bold to winners
            if j > 0 and cell_weights[i-1][j]:  # Skip model name column (j=0)
                cell.set_text_props(weight='bold', size=11)
            
            cell.set_height(0.1)
    
    plt.title("Performance Comparison: Baseline vs Agentic\n(Bold indicates best score for each model)", 
              pad=20, size=15, weight='bold', family='sans-serif')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Professional table saved to {output_path}")


def render_visuals(data):
    # 1. Professional B&W Table with Bold Winners
    render_professional_table(data, 'outputs/final_table_visual.png')
    
    # 2. F1 Score Comparison Graph
    try:
        data['ChatDoc'] = pd.to_numeric(data['ChatDoc'])
        
        models = data['Model'].unique()
        models = [m.replace(" (Base)", "").replace(" (Agent)", "").replace("openai/", "").replace("meta-llama/", "").replace("qwen/", "") for m in models if "(Base)" in m]
        
        # Extract scores
        baseline_scores = []
        agentic_scores = []
        
        for m in models:
            # Flexible matching
            base_row = data[data['Model'].str.contains(m) & data['Model'].str.contains("(Base)")]
            agent_row = data[data['Model'].str.contains(m) & data['Model'].str.contains("(Agent)")]
            
            if not base_row.empty and not agent_row.empty:
                baseline_scores.append(base_row['ChatDoc'].values[0])
                agentic_scores.append(agent_row['ChatDoc'].values[0])
            else:
                baseline_scores.append(0)
                agentic_scores.append(0)
                
        # Plot
        x = np.arange(len(models))
        width = 0.35
        
        # Use a professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color='#7f8c8d', alpha=0.9)
        rects2 = ax.bar(x + width/2, agentic_scores, width, label='Agentic (Critique-Refine)', color='#27ae60', alpha=0.9)
        
        ax.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
        ax.set_title('Medical Domain Semantic Accuracy (F1 Score)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=10, loc='upper left')
        
        ax.set_ylim(70, 90) # Optimizing zoom for 80-85 range
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig('outputs/f1_improvement_graph.png', dpi=300)
        print("F1 Graph saved to outputs/f1_improvement_graph.png")
        
    except Exception as e:
        print(f"Error generating graph: {e}")

if __name__ == "__main__":
    try:
        df = pd.read_csv('outputs/detailed_results.csv')
        render_visuals(df)
    except Exception as e:
        print(f"Error: {e}")
