# Utility Scripts

This directory contains utility scripts for the Agentic Web Summarizer application.

## Database Scripts

### `init_db.py`

Initializes the database tables required by the application.

```bash
python scripts/init_db.py
```

Run this script before starting the application for the first time or after making changes to the database models.

### `view_agent_interactions.py`

Displays recent agent interactions from the database.

```bash
python scripts/view_agent_interactions.py [hours] [limit]
```

Parameters:
- `hours`: Number of hours to look back (default: 24)
- `limit`: Maximum number of records to display (default: 20)

Example:
```bash
# View interactions from the last 48 hours, limited to 30 records
python scripts/view_agent_interactions.py 48 30
```

## Environment Setup

Before running any scripts, make sure your environment is properly configured:

1. Create a virtual environment and activate it
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set up your `.env` file with the necessary configuration
4. Run the database initialization script: `python scripts/init_db.py`