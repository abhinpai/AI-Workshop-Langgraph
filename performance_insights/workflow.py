from typing import Dict, List, Any, Annotated, Tuple
from datetime import datetime, timedelta
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import VertexAI
from performance_insights.nodes import (
    analyze_trend_with_llm,
    analyze_trends_with_llm,
    get_token,
    get_assets,
    get_attributes,
    filter_attributes,
    get_trend_data,
    generate_report,
    send_email
)
from performance_insights.state import AgentState, OutputState, WorkflowState

def create_workflow(
    tenant_id: str,
    site_name: str,
    object_name: str,
    date_range: int,
    email_ids: List[str],
    kind: str = None
) -> Tuple[Graph, WorkflowState]:
    """
    Create the performance insights workflow graph.
    
    Args:
        tenant_id: The tenant ID
        site_name: The site name
        object_name: The object name to analyze
        date_range: Date range in days
        email_ids: List of email IDs to send the report
        kind: Optional kind to filter attributes
        
    Returns:
        Tuple[Graph, WorkflowState]: The configured workflow graph and initial state
    """
    
    # Initialize the workflow graph
    workflow = StateGraph(AgentState, output=OutputState)
    
    # Add nodes to the graph
    workflow.add_node("get_token", get_token)
    workflow.add_node("get_assets", get_assets)
    workflow.add_node("get_attributes", get_attributes)
    workflow.add_node("filter_attributes", filter_attributes)
    workflow.add_node("get_trend_data", get_trend_data)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("send_email", send_email)
    workflow.add_node("analyze_trend_with_llm", analyze_trend_with_llm)
    
    # # Define the workflow edges
    workflow.add_edge("get_token", "get_assets")
    workflow.add_edge("get_assets", "get_attributes")
    
    # # Conditional edge for attribute filtering
    def should_filter(state: AgentState) -> str:
        return "filter" if state.kind else "get_trend"
    
    workflow.add_conditional_edges(
        "get_attributes",
        should_filter,
        {
            "filter": "filter_attributes",
            "get_trend": "get_trend_data"
        }
    )
    
    workflow.add_edge("filter_attributes", "get_trend_data")
    workflow.add_conditional_edges("get_trend_data", analyze_trends_with_llm, "analyze_trend_with_llm")
    workflow.add_edge("analyze_trend_with_llm", "generate_report")
    workflow.add_edge("generate_report", "send_email")
    workflow.add_edge("send_email", END)
    
    # Set the entry point
    workflow.set_entry_point("get_token")
    
    # Compile the workflow
    app = workflow.compile()


    initial_state = AgentState(
        site_name=site_name,
        object_name=object_name,
        tenant_id=tenant_id,
        date_range=date_range,
        email_ids=email_ids,
        kind=kind
    )

    return app, initial_state

async def run_workflow(
    tenant_id: str,
    site_name: str,
    object_name: str,
    date_range: int,
    email_ids: List[str],
    kind: str = None
) -> Dict[str, Any]:
    """
    Run the performance insights workflow.
    
    Args:
        tenant_id: The tenant ID
        site_name: The site name
        object_name: The object name to analyze
        date_range: Date range in days
        email_ids: List of email IDs to send the report
        kind: Optional kind to filter attributes
        
    Returns:
        Dict[str, Any]: The final workflow state
    """
    app, initial_state = create_workflow(
        tenant_id=tenant_id,
        site_name=site_name,
        object_name=object_name,
        date_range=date_range,
        email_ids=email_ids,
        kind=kind
    )
    
    # Run the workflow
    final_state = await app.ainvoke(initial_state)
    return final_state 