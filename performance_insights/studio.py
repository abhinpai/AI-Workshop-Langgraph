"""
Studio configuration for LangGraph development server.
"""

from performance_insights.workflow import create_workflow

# Create a sample workflow instance for development
graph, _ = create_workflow(
    tenant_id="gfsconnected",
    site_name="DC12",
    object_name="DC12Ammoniapump2",
    date_range=10,
    email_ids=["abhin.pai@honeywell.com"],
    kind="vibration"
)

# Export the graph for langgraph dev server
__all__ = ["graph"]
