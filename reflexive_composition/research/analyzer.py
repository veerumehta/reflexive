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

class ReflexiveCompositionAnalyzer:
    """Academic research analysis tools for Reflexive Composition."""
    
    def __init__(self, framework=None):
        self.framework = framework
        self.results_data = []
        self.experiment_metadata = {
            "start_time": datetime.utcnow(),
            "experiments_run": 0,
            "total_documents": 0,
            "total_extractions": 0
        }
    
    def record_experiment(self, 
                         experiment_type: str,
                         document_text: str,
                         extraction_result: Dict[str, Any],
                         validation_result: Dict[str, Any],
                         generation_result: Dict[str, Any],
                         metadata: Dict[str, Any] = None):
        """Record experimental data for analysis."""
        
        experiment_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_type": experiment_type,
            "document_length": len(document_text),
            "extracted_triples": len(extraction_result.get("triples", [])),
            "validated_triples": len(validation_result.get("validated_triples", [])),
            "validation_accuracy": (len(validation_result.get("validated_triples", [])) / 
                                  max(1, len(extraction_result.get("triples", [])))),
            "response_length": len(generation_result.get("response", "")),
            "knowledge_used": generation_result.get("knowledge_used", 0),
            "processing_time": extraction_result.get("processing_time", 0),
            "confidence_scores": [t.get("confidence", 0) for t in extraction_result.get("triples", [])],
            "domain": metadata.get("domain", "general") if metadata else "general",
            "hitl_interventions": validation_result.get("human_reviewed", 0),
            "schema_updates": metadata.get("schema_updates", 0) if metadata else 0,
            "error_count": len(extraction_result.get("errors", [])),
            "metadata": metadata or {}
        }
        
        self.results_data.append(experiment_data)
        self.experiment_metadata["experiments_run"] += 1
        self.experiment_metadata["total_documents"] += 1
        self.experiment_metadata["total_extractions"] += len(extraction_result.get("triples", []))
    
    def generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        
        if not self.results_data:
            return {"error": "No experimental data available"}
        
        df = pd.DataFrame(self.results_data)
        
        analysis = {
            "summary_statistics": self._calculate_summary_statistics(df),
            "extraction_performance": self._analyze_extraction_performance(df),
            "validation_efficiency": self._analyze_validation_efficiency(df),
            "knowledge_utilization": self._analyze_knowledge_utilization(df),
            "temporal_trends": self._analyze_temporal_trends(df),
            "domain_analysis": self._analyze_domain_performance(df),
            "error_analysis": self._analyze_error_patterns(df)
        }
        
        return analysis
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            "total_experiments": len(df),
            "avg_extraction_accuracy": df["validation_accuracy"].mean(),
            "avg_processing_time": df["processing_time"].mean(),
            "avg_triples_per_document": df["extracted_triples"].mean(),
            "avg_knowledge_utilization": df["knowledge_used"].mean(),
            "total_hitl_interventions": df["hitl_interventions"].sum(),
            "avg_confidence_score": np.mean([np.mean(scores) for scores in df["confidence_scores"] if scores]),
            "error_rate": df["error_count"].sum() / len(df),
            "unique_domains": df["domain"].nunique(),
            "schema_evolution_events": df["schema_updates"].sum()
        }
    
    def _analyze_extraction_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze knowledge extraction performance."""
        return {
            "accuracy_by_domain": df.groupby("domain")["validation_accuracy"].agg(["mean", "std"]).to_dict(),
            "extraction_volume_correlation": df[["document_length", "extracted_triples"]].corr().iloc[0, 1],
            "confidence_distribution": {
                "mean": np.mean([np.mean(scores) for scores in df["confidence_scores"] if scores]),
                "std": np.std([np.mean(scores) for scores in df["confidence_scores"] if scores]),
                "percentiles": np.percentile([score for scores in df["confidence_scores"] for score in scores], 
                                           [25, 50, 75, 90, 95])
            }
        }
    
    def _analyze_validation_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze HITL validation efficiency."""
        total_validations = df["extracted_triples"].sum()
        hitl_interventions = df["hitl_interventions"].sum()
        
        return {
            "automation_rate": 1 - (hitl_interventions / max(1, total_validations)),
            "intervention_rate_by_domain": df.groupby("domain").apply(
                lambda x: x["hitl_interventions"].sum() / max(1, x["extracted_triples"].sum())
            ).to_dict(),
            "validation_accuracy_trend": df["validation_accuracy"].rolling(window=5).mean().tolist()
        }
    
    def _analyze_knowledge_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how extracted knowledge is utilized in generation."""
        return {
            "utilization_rate": df["knowledge_used"].mean() / max(1, df["validated_triples"].mean()),
            "response_quality_correlation": df[["knowledge_used", "response_length"]].corr().iloc[0, 1],
            "utilization_by_domain": df.groupby("domain")["knowledge_used"].mean().to_dict()
        }
    
    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        return {
            "accuracy_trend": df["validation_accuracy"].rolling(window=10).mean().tolist(),
            "processing_time_trend": df["processing_time"].rolling(window=10).mean().tolist(),
            "extraction_volume_trend": df["extracted_triples"].rolling(window=10).mean().tolist()
        }
    
    def _analyze_domain_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across different domains."""
        domain_stats = df.groupby("domain").agg({
            "validation_accuracy": ["mean", "std"],
            "extracted_triples": ["mean", "std"],
            "processing_time": ["mean", "std"],
            "hitl_interventions": "sum",
            "knowledge_used": "mean"
        })
        
        return domain_stats.to_dict()
    
    def _analyze_error_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze error patterns and failure modes."""
        return {
            "error_rate_by_domain": df.groupby("domain")["error_count"].sum().to_dict(),
            "error_correlation_with_length": df[["document_length", "error_count"]].corr().iloc[0, 1],
            "high_error_experiments": df[df["error_count"] > df["error_count"].quantile(0.9)].index.tolist()
        }
