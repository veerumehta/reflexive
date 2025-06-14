"""
Research Analysis Tools for Reflexive Composition Framework
Provides formal outputs, flowcharts, tables, charts, and citation support
for academic research on knowledge graphs and large language models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configure visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationGenerator:
    """Generate academic-quality visualizations."""
    
    def __init__(self, analyzer: ReflexiveCompositionAnalyzer):
        self.analyzer = analyzer
        self.figures = {}
        
        # Set academic style
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def generate_performance_dashboard(self, save_path: str = "performance_dashboard.html"):
        """Generate interactive performance dashboard."""
        
        if not self.analyzer.results_data:
            print("No data available for visualization")
            return
        
        df = pd.DataFrame(self.analyzer.results_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Extraction Accuracy Over Time',
                'Knowledge Utilization by Domain',
                'Processing Time Distribution',
                'HITL Intervention Rate',
                'Confidence Score Distribution',
                'Schema Evolution Events'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Extraction Accuracy Over Time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['validation_accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Knowledge Utilization by Domain
        domain_knowledge = df.groupby('domain')['knowledge_used'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=domain_knowledge['domain'],
                y=domain_knowledge['knowledge_used'],
                name='Avg Knowledge Used',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Processing Time Distribution
        fig.add_trace(
            go.Histogram(
                x=df['processing_time'],
                nbinsx=20,
                name='Processing Time',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. HITL Intervention Rate
        intervention_rate = df.groupby('domain').apply(
            lambda x: x['hitl_interventions'].sum() / max(1, x['extracted_triples'].sum())
        ).reset_index()
        intervention_rate.columns = ['domain', 'intervention_rate']
        
        fig.add_trace(
            go.Bar(
                x=intervention_rate['domain'],
                y=intervention_rate['intervention_rate'],
                name='Intervention Rate',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # 5. Confidence Score Distribution
        all_confidence_scores = [score for scores in df['confidence_scores'] for score in scores if scores]
        fig.add_trace(
            go.Histogram(
                x=all_confidence_scores,
                nbinsx=20,
                name='Confidence Scores',
                marker_color='purple',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # 6. Schema Evolution Events
        schema_timeline = df.groupby(df['timestamp'].dt.date)['schema_updates'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=schema_timeline['timestamp'],
                y=schema_timeline['schema_updates'],
                mode='lines+markers',
                name='Schema Updates',
                line=dict(color='red', width=2)
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Reflexive Composition Performance Dashboard",
            title_x=0.5,
            showlegend=False,
            height=1000,
            template="plotly_white"
        )
        
        # Save the dashboard
        fig.write_html(save_path)
        print(f"✓ Performance dashboard saved to {save_path}")
        
        return fig
    
    def generate_knowledge_graph_visualization(self, kg=None, save_path: str = "knowledge_graph.png"):
        """Generate knowledge graph visualization."""
        
        if kg is None and self.analyzer.framework:
            kg = self.analyzer.framework.knowledge_graph
        
        if kg is None or not kg.triples:
            print("No knowledge graph data available")
            return
        
        # Create NetworkX graph for visualization
        G = nx.Graph()
        edge_labels = {}
        
        # Add nodes and edges
        for triple in kg.triples[:50]:  # Limit for readability
            G.add_node(triple.subject, type='entity')
            G.add_node(triple.object, type='entity')
            G.add_edge(triple.subject, triple.object, relation=triple.predicate, weight=triple.confidence)
            edge_labels[(triple.subject, triple.object)] = triple.predicate
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        
        # Draw edges
        edges = G.edges(data=True)
        edge_weights = [edge[2]['weight'] for edge in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in edge_weights], 
                              alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Draw edge labels (relations)
        edge_label_pos = {edge: (pos[edge[0]][0] + pos[edge[1]][0])/2, 
                         (pos[edge[0]][1] + pos[edge[1]][1])/2) 
                         for edge in edge_labels.keys()}
        
        for edge, label in list(edge_labels.items())[:20]:  # Limit edge labels
            x, y = edge_label_pos[edge]
            plt.text(x, y, label, fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Knowledge graph visualization saved to {save_path}")
    
    def generate_comparison_chart(self, baseline_results: List[Dict], 
                                 reflexive_results: List[Dict],
                                 save_path: str = "comparison_chart.png"):
        """Generate comparison chart between baseline and reflexive composition."""
        
        metrics = ['accuracy', 'processing_time', 'knowledge_utilization', 'error_rate']
        baseline_means = [np.mean([r.get(m, 0) for r in baseline_results]) for m in metrics]
        reflexive_means = [np.mean([r.get(m, 0) for r in reflexive_results]) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline', color='lightcoral')
        bars2 = ax.bar(x + width/2, reflexive_means, width, label='Reflexive Composition', color='lightblue')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Performance')
        ax.set_title('Performance Comparison: Baseline vs Reflexive Composition')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison chart saved to {save_path}")
    
    def generate_workflow_diagram(self, save_path: str = "workflow_diagram.png"):
        """Generate workflow diagram for Reflexive Composition."""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define workflow steps and positions
        steps = [
            {"name": "Input\nDocument", "pos": (1, 8), "color": "lightgreen"},
            {"name": "LLM2KG\nExtraction", "pos": (3, 8), "color": "lightblue"},
            {"name": "Knowledge\nValidation", "pos": (5, 8), "color": "lightyellow"},
            {"name": "HITL\nReview", "pos": (5, 6), "color": "lightcoral"},
            {"name": "Knowledge\nGraph Update", "pos": (7, 8), "color": "lightpink"},
            {"name": "Schema\nEvolution", "pos": (7, 6), "color": "lightgray"},
            {"name": "Query\nInput", "pos": (1, 4), "color": "lightgreen"},
            {"name": "KG2LLM\nGeneration", "pos": (3, 4), "color": "lightblue"},
            {"name": "Contextualized\nResponse", "pos": (5, 4), "color": "lightcyan"},
            {"name": "Feedback\nLoop", "pos": (7, 2), "color": "orange"}
        ]
        
        # Draw steps
        for step in steps:
            x, y = step["pos"]
            rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                               facecolor=step["color"], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, step["name"], ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrows
        arrows = [
            ((1.4, 8), (2.6, 8)),      # Input -> LLM2KG
            ((3.4, 8), (4.6, 8)),      # LLM2KG -> Validation
            ((5, 7.7), (5, 6.3)),      # Validation -> HITL
            ((5.4, 8), (6.6, 8)),      # Validation -> KG Update
            ((7, 7.7), (7, 6.3)),      # KG Update -> Schema Evolution
            ((1.4, 4), (2.6, 4)),      # Query -> KG2LLM
            ((3.4, 4), (4.6, 4)),      # KG2LLM -> Response
            ((7, 5.7), (7, 2.3)),      # Schema -> Feedback
            ((6.6, 2), (3.4, 3.7)),    # Feedback -> KG2LLM (curved)
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        # Add title and labels
        ax.set_title('Reflexive Composition Workflow', fontsize=16, fontweight='bold', pad=20)
        ax.text(4, 9.5, 'Knowledge Construction Pipeline', ha='center', fontsize=12, style='italic')
        ax.text(4, 1, 'Knowledge Utilization Pipeline', ha='center', fontsize=12, style='italic')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Workflow diagram saved to {save_path}")
