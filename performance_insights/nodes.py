import os
import json
import asyncio
import ssl
import certifi
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
import aiohttp
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from .state import AgentState, OutputState, AssetInfo, AttributeInfo, TrendData, PrivateAgentState
from .utils import Logger
import requests 
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

# Create SSL context with certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

async def get_token(state: AgentState) -> PrivateAgentState:
    """Get authentication token and store in private state."""
    Logger.step(1, 6, "Getting authentication token")
    
    payload = "grant_type=client_credentials&scope=builder.access"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": os.getenv("IAM_CREDENTIALS"),
    }
    
    Logger.api("POST", os.getenv("IAM_TOKEN_BASE_URL"))
    response = requests.request(
        "POST", os.getenv("IAM_TOKEN_BASE_URL"), headers=headers, data=payload
    )
    response_json = json.loads(response.text)
    access_token = response_json["access_token"]
    Logger.success("Successfully obtained access token")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=state.date_range)

    formatted_start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    formatted_end_date = end_date.strftime("%Y-%m-%dT%H:%M:%S")
    Logger.info(f"Analysis period: {formatted_start_date} to {formatted_end_date}")

    return {
        "token": access_token,
        "start_date": formatted_start_date,
        "end_date": formatted_end_date,
    }

async def get_assets(state: PrivateAgentState) -> OutputState:
    """Query and get all assets for the given object name."""
    Logger.step(2, 6, f"Fetching assets for {state.object_name}")
    
    query = f"get info for all assets in {state.object_name} including subsystem"
    url = f"{API_BASE_URL}/query-templates/query"

    headers = {
        "Content-Type": "application/json",
        "Customerid": state.tenant_id,
        "Sitename": state.site_name,
        "content-encoding": "gzip",
        "Authorization": f"Bearer {state.token}"
    }
    
    Logger.api("GET", url)
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(
            url,
            params={"query": query},
            headers=headers
        ) as response:
            data = await response.json()
            assets = [
                AssetInfo(
                    name=asset[0],
                    display_name=asset[1],
                    type=asset[2],
                    asset_type_display_name=asset[3],
                    id=asset[4],
                    path=asset[5],
                    parent=asset[6],
                    criticality=asset[7],
                    category=asset[8],
                    asset_class=asset[9],
                    description=asset[10],
                )
                for asset in data["ResultSet"]["Data"]
            ]
    
    Logger.success(f"Found {len(assets)} assets")
    return {
        "assets": assets
    }

async def get_attributes(state: PrivateAgentState) -> OutputState:
    """Get attributes for all assets in parallel."""
    Logger.step(3, 6, "Fetching attributes for all assets")
    
    headers = {
        "Content-Type": "application/json",
        "Customerid": state.tenant_id,
        "Sitename": state.site_name,
        "content-encoding": "gzip",
        "Authorization": f"Bearer {state.token}"
    }
    
    async def fetch_asset_attributes(asset: AssetInfo):
        Logger.processing(f"Fetching attributes for asset: {asset.name}")
        query = f"GET VALUES OF ATTRIBUTES WITH PROPERTIES FOR ALL ASSETS IN {asset.name} including subsystem"
        url = f"{API_BASE_URL}/query-templates/query"
        
        Logger.api("GET", url)
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                url,
                params={"query": query},
                headers=headers
            ) as response:
                data = await response.json()
                attributes = [
                    AttributeInfo(
                        name=attribute[0],
                        write_tag_Name=attribute[1],
                        read_server_name=attribute[2],
                        read_tag_name=attribute[3],
                        history_server_name=attribute[4],
                        history_tagN_name=attribute[5],
                        equipment_name=attribute[6],
                        equipment_id=attribute[7],
                        display_name=attribute[8],
                        full_display_name=attribute[9],
                        tag_name=attribute[10],
                        uom=attribute[11],
                        timestamp=attribute[12],
                        actual_value=attribute[13],
                        expected_value=attribute[14],
                        high_limit=attribute[15],
                        last=attribute[16],
                        low_limit=attribute[17],
                        raw=attribute[18],
                    )
                    for attribute in data["ResultSet"]["Data"]
                ]
                Logger.data(f"Found {len(attributes)} attributes for {asset.name}")
                return asset.id, attributes
    
    tasks = [fetch_asset_attributes(asset) for asset in state.assets]
    results = await asyncio.gather(*tasks)
    
    Logger.success("Successfully fetched all attributes")
    return {
        "attributes": dict(results)
    }

async def filter_attributes(state: PrivateAgentState) -> OutputState:
    """Filter attributes based on kind using Vertex AI."""
    Logger.step(4, 6, f"Filtering attributes by kind: {state.kind}")
    
    if not state.kind:
        Logger.info("No kind specified, skipping filtering")
        return state
    
    llm = VertexAI(
        model_name="gemini-2.0-flash-001",
        temperature=0
    )
    
    filtered_attributes = {}
    for asset_id, attributes in state.attributes.items():
        Logger.processing(f"Filtering attributes for asset: {asset_id}")
        schema = list[AttributeInfo.model_json_schema()]
        prompt = f"""Identify and filter attributes that are related to {state.kind} from the provided list of attributes:

        Attributes:
        {json.dumps([attribute.model_dump() for attribute in attributes], indent=2)}
        
        RESPONSE REQUIREMENTS:
        1. Return ONLY a JSON array
        2. Do NOT include ```json``` markers
        3. Do NOT include any explanations
        4. Do NOT include any additional text
        5. If no attributes match {state.kind}, return []
        6. Response must exactly match this schema:
        {schema}
        """
        
        response = await llm.ainvoke(prompt)
        try:
            cleaned_response = response
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1]
            if "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[0]
            
            cleaned_response = cleaned_response.strip()
            response = json.loads(cleaned_response)
            Logger.data(f"Found {len(response)} matching attributes")
        except Exception as e:
            Logger.error(f"Error parsing response: {e}")
            response = []

        filtered_attributes[asset_id] = [AttributeInfo(**attribute) for attribute in response]
    
    Logger.success("Successfully filtered attributes")
    return {
        "filtered_attributes": filtered_attributes
    }

async def get_trend_data(state: PrivateAgentState) -> OutputState:
    """Get trend data for attributes in parallel."""
    Logger.step(5, 6, "Fetching trend data")
    
    attributes_map = (
        state.filtered_attributes
        if state.filtered_attributes
        else state.attributes
    )
    
    async def fetch_trend_data(asset: AssetInfo, attribute: AttributeInfo):
        Logger.processing(f"Fetching trend data for {attribute.name} in {asset.name}")
        start = datetime.strptime(state.start_date, "%Y-%m-%dT%H:%M:%S")
        end = datetime.strptime(state.end_date, "%Y-%m-%dT%H:%M:%S")

        headers = {
            "Content-Type": "application/json",
            "Customerid": state.tenant_id,
            "Sitename": state.site_name,
            "content-encoding": "gzip",
            "Authorization": f"Bearer {state.token}"
        }
        
        all_data = []
        while start < end:
            chunk_end = min(start + timedelta(days=30), end)
            
            query = f"GET TREND OF {attribute.name} WITH PROPERTIES FOR ASSET {asset.name} DURING PERIOD {start.strftime('%Y-%m-%dT%H:%M:%S')} AND {chunk_end.strftime('%Y-%m-%dT%H:%M:%S')}"
            url = f"{API_BASE_URL}/query-templates/query"
            
            Logger.api("GET", url)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    url,
                    params={"query": query},
                    headers=headers
                ) as response:
                    data = await response.json()
                    all_data.extend(data["ResultSet"]["Data"])
            
            start = chunk_end
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            Logger.data(f"Processed {len(df)} trend points for {attribute.name}")
            return asset.id, attribute.name, TrendData(
                timestamp=df[0].tolist(),
                value=df[3].tolist(),
                quality=df[4].tolist(),
            )
        else:
            Logger.warning(f"No trend data found for {attribute.name}")
            return asset.id, attribute.name, None
    
    tasks = []
    for asset in state.assets:
        if asset.id in attributes_map:
            for attribute in attributes_map[asset.id]:
                tasks.append(fetch_trend_data(asset, attribute))
    
    results = await asyncio.gather(*tasks)
    
    trend_data = {}
    for asset_id, attr_name, data in results:
        if asset_id not in trend_data:
            trend_data[asset_id] = {}
        if data is not None:
            trend_data[asset_id][attr_name] = data
    
    Logger.success("Successfully fetched all trend data")
    return {
        "trends": trend_data
    }

def format_datetime(date_str: str) -> str:
    """Convert ISO datetime to human readable format."""
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    return dt.strftime("%B %d, %Y %I:%M %p")  # Example: March 22, 2024 02:30 PM

async def generate_report(state: PrivateAgentState) -> OutputState:
    """Generate PDF report with stats and graphs."""
    Logger.step(6, 6, "Generating PDF report")
    
    # Create PDF document
    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title and summary
    elements.append(Paragraph("Performance Insights Report", styles["Title"]))
    elements.append(Spacer(1, 24))
    
    # Add configuration summary with better formatting and human-readable dates
    elements.append(Paragraph(f"Analysis Period: {format_datetime(state.start_date)} to {format_datetime(state.end_date)}", styles["Normal"]))
    elements.append(Paragraph(f"Object Name: {state.object_name if state.object_name else 'All'}", styles["Normal"]))
    elements.append(Paragraph(f"Site Name: {state.site_name}", styles["Normal"]))
    elements.append(Spacer(1, 24))
    
    # Add tables and graphs for each asset
    for asset in state.assets:
        Logger.processing(f"Processing report for asset: {asset.name}")
        asset_id = asset.id
        elements.append(Paragraph(f"Asset: {asset.name}", styles["Heading2"]))
        elements.append(Spacer(1, 24))
        
        # Create attribute table
        attributes = state.attributes[asset.id]
        trend_data = state.trends.get(asset_id, {})
        
        table_data = [
            ["Attribute", "High Limit", "Low Limit", "Expected Value", "Avg Trend", "High Trend", "Low Trend"]
        ]
        
        for attr in attributes:
            trend = trend_data.get(attr.name)
            if trend:
                values = np.array(trend['value'])
                avg_trend = np.mean(values) if len(values) > 0 else None
                high_trend = np.max(values) if len(values) > 0 else None
                low_trend = np.min(values) if len(values) > 0 else None

                row = [
                    attr.name,
                    str(attr.high_limit if attr.high_limit is not None else 'N/A'),
                    str(attr.low_limit if attr.low_limit is not None else 'N/A'),
                    str(attr.expected_value if attr.expected_value is not None else 'N/A'),
                    str(round(avg_trend, 2)) if avg_trend is not None else "No Data",
                    str(round(high_trend, 2)) if high_trend is not None else "No Data",
                    str(round(low_trend, 2)) if low_trend is not None else "No Data"
                ]
                table_data.append(row)
           
            Logger.data(f"Processed trend data for {attr.name}")
        
        # Add left and right margins by adjusting page width
        available_width = letter[0] - 72  # 72 points = 1 inch margin on each side
        col_widths = [
            available_width * 0.20,  # Attribute name gets more space
            available_width * 0.13,  # High Limit
            available_width * 0.13,  # Low Limit
            available_width * 0.15,  # Expected Value
            available_width * 0.13,  # Avg Trend
            available_width * 0.13,  # High Trend
            available_width * 0.13,  # Low Trend
        ]
        table = Table(table_data, colWidths=col_widths)
        
        # Apply enhanced table styling
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),  # Darker blue header
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            
            # Cell padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            
            # Alternating row colors
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9F9')),  # Light gray
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            
            # Grid styling
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),  # Light gray grid
            ('LINEBEFORE', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('LINEAFTER', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('LINEABOVE', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            
            # First column styling (attribute names)
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 36))  # Increased spacing after table
        
        # Generate trend graphs with enhanced styling and better clarity
        for attr_name, trend in trend_data.items():
            Logger.processing(f"Generating trend graph for {attr_name}")
            
            # Create figure with improved size and DPI but slightly reduced height
            plt.figure(figsize=(12, 6), dpi=300)  # Reduced height for better page fit
            
            # Add more padding to the plot
            plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
            
            # Plot with enhanced styling
            plt.plot(trend['timestamp'], trend['value'], linewidth=2.5, color='#2C3E50')
            
            # Customize the plot
            plt.title(f"{attr_name} Trend", pad=20, fontsize=16, fontweight='bold')
            plt.xlabel('Timestamp', labelpad=12, fontsize=12)
            plt.ylabel('Value', labelpad=12, fontsize=12)
            
            # Improve tick labels
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            # Customize grid
            plt.grid(True, linestyle='--', alpha=0.6, color='#95a5a6')
            
            # Rotate and align the tick labels so they look better
            plt.xticks(rotation=45, ha='right')
            
            # Add padding around the plot
            plt.margins(x=0.05)
            plt.tight_layout()
            
            # Save plot with high quality but reduced padding
            plt.savefig(
                f"temp_{asset_id}_{attr_name}.png",
                bbox_inches='tight',
                pad_inches=0.5,  # Reduced padding
                dpi=300
            )
            
            # Calculate available space and add page break if needed
            available_height = letter[1]  # Total page height
            current_height = 0
            
            # Add graph title with minimal spacing
            elements.append(Paragraph(f"{attr_name} Trend", styles["Heading3"]))
            elements.append(Spacer(1, 12))  # Reduced spacing before graph
            
            # Add image with adjusted dimensions
            img = Image(f"temp_{asset_id}_{attr_name}.png", width=500, height=250)  # Reduced height
            elements.append(img)
            elements.append(Spacer(1, 24))  # Spacing after graph
            
            plt.close()
            
            # Add a page break after every second graph
            if len(elements) % 4 == 0:  # Assuming each graph section has 4 elements
                elements.append(PageBreak())
    
    # Build PDF
    Logger.processing("Building final PDF report")
    doc.build(elements)
    
    # Read generated PDF
    with open("report.pdf", "rb") as f:
        report_data = f.read()
        
    # Cleanup temporary files
    # Logger.processing("Cleaning up temporary files")
    # for asset in state.assets:
    #     for attr_name in state.trends.get(asset.id, {}):
    #         os.remove(f"temp_{asset.id}_{attr_name}.png")
    # os.remove("report.pdf")
    
    Logger.success("Report generation completed")
    return {
        "report_data": report_data
    }
    

async def send_email(state: PrivateAgentState) -> OutputState:
    """Send email with report to specified recipients."""
    email_sender = os.getenv("EMAIL_SENDER")
    email_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    
    # Create message
    msg = MIMEMultipart()
    msg["From"] = email_sender
    msg["To"] = ", ".join(state.email_ids)
    msg["Subject"] = f"Performance Insights Report - {state.site_name}"
    
    # Format dates for email
    start_date_formatted = format_datetime(state.start_date)
    end_date_formatted = format_datetime(state.end_date)
    
    # Add HTML body with better formatting
    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2C3E50; text-align: center; border-bottom: 2px solid #BDC3C7; padding-bottom: 10px;">
                    Performance Insights Report
                </h1>
                
                <div style="background-color: #F8F9F9; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong style="color: #2C3E50;">📍 Site Name:</strong> {state.site_name}</p>
                    <p><strong style="color: #2C3E50;">📅 Analysis of {state.date_range} Days Trends</strong><br>
                       <span style="margin-left: 25px">From: {start_date_formatted}</span><br>
                       <span style="margin-left: 25px">To: {end_date_formatted}</span>
                    </p>
                    <p><strong style="color: #2C3E50;">🎯 Focus Area:</strong> {state.kind if state.kind else "All Attributes"}</p>
                </div>
                
                <div style="background-color: #EBF5FB; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p style="color: #2980B9;">
                        <strong>📊 Report Summary:</strong><br>
                        This report provides detailed analysis of attributes{f' related to "{state.kind}"' if state.kind else ''} 
                        for the specified time period. It includes trend analysis, statistical data, and performance metrics.
                    </p>
                </div>
                
                <p style="color: #7F8C8D; font-style: italic;">
                    Please find the detailed report attached to this email.
                </p>
                
                <div style="border-top: 1px solid #BDC3C7; margin-top: 20px; padding-top: 20px; font-size: 12px; color: #95A5A6;">
                    <p>This is an automated report generated by Performance Insights System.</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    # Attach HTML body
    msg.attach(MIMEText(body, "html"))
    
    # Attach PDF
    pdf_attachment = MIMEApplication(state.report_data, _subtype="pdf")
    pdf_attachment.add_header(
        "Content-Disposition",
        "attachment",
        filename="performance_insights_report.pdf"
    )
    msg.attach(pdf_attachment)
    
    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_sender, email_password)
        server.send_message(msg)
    
    return {} 