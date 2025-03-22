import asyncio
import argparse
from typing import List
from .workflow import run_workflow

def parse_args():
    parser = argparse.ArgumentParser(description="Performance Insights Analysis")
    parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    parser.add_argument("--site-name", required=True, help="Site name")
    parser.add_argument("--object-name", required=True, help="Object name")
    parser.add_argument("--date-range", type=int, required=True, help="Date range in days")
    parser.add_argument("--email-ids", required=True, nargs="+", help="Email IDs to send report")
    parser.add_argument("--kind", help="Optional kind to filter attributes")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    try:
        final_state = await run_workflow(
            tenant_id=args.tenant_id,
            site_name=args.site_name,
            object_name=args.object_name,
            date_range=args.date_range,
            email_ids=args.email_ids,
            kind=args.kind
        )
        print("Performance insights analysis completed successfully!")
        print(f"Report sent to: {', '.join(args.email_ids)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 