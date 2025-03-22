"""
Performance Insights Analysis using LangGraph.

This package provides a workflow for analyzing asset performance data
and generating insights reports using LangGraph and Vertex AI.
"""

from .workflow import run_workflow
from .state import WorkflowState, AssetInfo, AttributeInfo, TrendData
from .studio import graph

__version__ = "0.1.0"
__all__ = ["run_workflow", "WorkflowState", "AssetInfo", "AttributeInfo", "TrendData", "graph"] 