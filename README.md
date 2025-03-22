# Performance Insights Analysis

A LangGraph-based workflow for analyzing asset performance data and generating insights reports.

## Features

- Fetches and analyzes asset performance data
- Parallel processing of attribute data
- Intelligent attribute filtering using Vertex AI
- Automatic trend data analysis with date range splitting
- PDF report generation with tables and graphs
- Email delivery of reports

## Requirements

- Python 3.8+
- Google Cloud Platform account with Vertex AI enabled
- SMTP server access for sending emails

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd performance-insights
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following configuration:
```
API_BASE_URL=https://emp.app.honeywellforge.com/dataapi/api
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
EMAIL_SENDER=your-email@example.com
EMAIL_PASSWORD=your-app-specific-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

## Usage

Run the analysis with the following command:

```bash
python -m performance_insights \
    --tenant-id <tenant-id> \
    --site-name <site-name> \
    --object-name <object-name> \
    --date-range <days> \
    --email-ids abhin.pai@honeywell.com \
    --kind Vibration  # Optional: Filter attributes by kind
```

### Arguments

- `--tenant-id`: Your tenant ID
- `--site-name`: Name of the site to analyze
- `--object-name`: Name of the object to analyze
- `--date-range`: Number of days to analyze
- `--email-ids`: One or more email addresses to send the report to
- `--kind`: (Optional) Filter attributes by kind (e.g., "Vibration")

## Report Contents

The generated PDF report includes:

1. Analysis configuration summary
2. For each asset:
   - Attribute tables with limits and trends
   - Trend graphs for each attribute
   - Statistical analysis

## Development

The project uses a LangGraph workflow with the following nodes:

1. Token acquisition
2. Asset data fetching
3. Parallel attribute processing
4. Intelligent attribute filtering (optional)
5. Parallel trend data analysis
6. Report generation
7. Email delivery

## License

MIT License 