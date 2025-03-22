from operator import add
from typing import Annotated, Dict, Any, List, TypedDict, Optional
from dataclasses import  dataclass, field
from pydantic import Field

from pydantic import BaseModel

class AssetInfo(BaseModel):
    id: str
    name: str
    display_name: str
    type: str
    asset_type_display_name: str
    path: Optional[str]
    parent: Optional[str]
    criticality: str | int | None
    category: str
    asset_class: str
    description: Optional[str]

class AttributeInfo(BaseModel):
    name: str
    write_tag_Name: str
    read_server_name: str
    read_tag_name: str
    history_server_name: str
    history_tagN_name: str
    equipment_name: str
    equipment_id: str
    display_name: str
    full_display_name: str
    tag_name: str
    uom: str
    timestamp: str | float | None
    actual_value: str | int | float | None
    expected_value: str | int | float | None
    high_limit: str | int | float | None
    last: str | int | float | None
    low_limit: str | int | float | None
    raw: str | int | float | None

class TrendData(TypedDict):
    timestamp: float | list[float] | None
    value: int | float | list[int] | list[float] | None
    quality: int | float | list[int] | list[float] | None

@dataclass
class WorkflowState:
    """State management for the performance insights workflow."""
    
    # Configuration passed to the workflow
    config: Dict[str, Any]
    
    # Private state (e.g., authentication tokens)
    private_state: Dict[str, Any]
    
    # Shared state between nodes
    shared_state: Dict[str, Any] = field(default_factory=lambda: {
        "assets": [],  # List[AssetInfo]
        "attributes": {},  # Dict[str, List[AttributeInfo]]  # asset_id -> attributes
        "filtered_attributes": {},  # Dict[str, List[AttributeInfo]]  # asset_id -> filtered attributes
        "trend_data": {},  # Dict[str, Dict[str, TrendData]]  # asset_id -> {attribute_name -> trend_data}
        "report_data": None,  # bytes of PDF report
        "error_log": []  # List of any errors or warnings during processing
    }) 

def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}


class AgentState(BaseModel):
    site_name: str = Field(..., description="Name of the site")
    object_name: str = Field(..., description="Name of any Object name")
    tenant_id: str = Field(..., description="Tenant ID")
    date_range: Optional[int] = Field(default=7, description="Date range in days", ge=0)
    email_ids: Optional[list[str]] = Field(
        default=None, description="List of email IDs"
    )
    kind: Optional[str] = Field(
        default="",
        description="Kind to determine what type of attribute report you are interested in",
    )

class OutputState(BaseModel):
    assets: Annotated[list[AssetInfo], add]
    attributes: Annotated[dict[str, list[AttributeInfo]], merge_dicts]
    filtered_attributes: Annotated[dict[str, list[AttributeInfo]], merge_dicts]
    trends: Annotated[dict[str, dict[str, TrendData]], merge_dicts]
    start_date: Optional[str]
    end_date: Optional[str]
    report_data: Optional[bytes]

class PrivateAgentState(AgentState, OutputState):
    token: str = Field(..., description="Access token")