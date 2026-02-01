"""
Analytics and event logging module.

Provides:
- Event logging for user behavior tracking
- Events table schema and initialization
- Helper functions for logging from backend and receiving from frontend
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request

# Database path - same directory as other DBs
DB_PATH = Path(__file__).parent.parent / "data" / "analytics.db"


def get_db():
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_events_db():
    """Initialize the events table."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            session_id TEXT,
            source TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            ip TEXT,
            user_agent TEXT
        )
    """)

    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")

    conn.commit()
    conn.close()
    print(f"Analytics database initialized at {DB_PATH}")


def log_event(
    *,
    event_type: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source: str = "backend",
    payload: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> int:
    """
    Log a user behavior event.

    Args:
        event_type: Semantic event type (e.g., 'train_submitted', 'select_tab')
        user_id: Username of the user performing the action
        session_id: Browser session ID for grouping events
        source: 'backend' or 'frontend'
        payload: Additional event data (context, ui, data)
        request: FastAPI request object for IP/user-agent

    Returns:
        The ID of the inserted event
    """
    conn = get_db()
    cursor = conn.cursor()

    # Extract request info if available
    ip = None
    user_agent = None
    if request:
        ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

    cursor.execute(
        """
        INSERT INTO events (timestamp, user_id, session_id, source, event_type, payload, ip, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            user_id,
            session_id,
            source,
            event_type,
            json.dumps(payload or {}),
            ip,
            user_agent,
        ),
    )

    event_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return event_id


def get_events(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Query events with optional filters.

    Args:
        user_id: Filter by user
        session_id: Filter by session
        event_type: Filter by event type
        since: Filter events after this timestamp (ISO format)
        until: Filter events before this timestamp (ISO format)
        limit: Maximum number of events to return

    Returns:
        List of event dictionaries
    """
    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM events WHERE 1=1"
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)

    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)

    if since:
        query += " AND timestamp >= ?"
        params.append(since)

    if until:
        query += " AND timestamp <= ?"
        params.append(until)

    query += " ORDER BY timestamp ASC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    events = []
    for row in rows:
        event = dict(row)
        event["payload"] = json.loads(event["payload"])
        events.append(event)

    return events


def list_sessions(
    user_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List distinct sessions with summary info.

    Args:
        user_id: Filter by user
        limit: Maximum number of sessions to return

    Returns:
        List of session summaries
    """
    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT
            session_id,
            user_id,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            COUNT(*) as event_count
        FROM events
        WHERE session_id IS NOT NULL
    """
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    query += " GROUP BY session_id, user_id ORDER BY start_time DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_session_events(session_id: str) -> List[Dict[str, Any]]:
    """Get all events for a specific session, ordered by time."""
    return get_events(session_id=session_id, limit=10000)


# Event type constants for consistency
class EventTypes:
    """Standard event types for the dashboard."""

    # Navigation
    PAGE_VIEW = "page_view"
    SELECT_TAB = "select_tab"

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"

    # Dataset
    SELECT_DATASET = "select_dataset"
    APPLY_FILTER = "apply_filter"

    # Training
    TRAIN_SUBMITTED = "train_submitted"
    TRAIN_STARTED = "train_started"
    TRAIN_COMPLETED = "train_completed"
    TRAIN_FAILED = "train_failed"

    # Model inspection
    VIEW_MODEL = "view_model"
    VIEW_METRICS = "view_metrics"
    VIEW_FEATURE_IMPORTANCE = "view_feature_importance"
    VIEW_TRAINING_CURVES = "view_training_curves"
    COMPARE_MODELS = "compare_models"

    # Analysis
    RUN_ANALYSIS = "run_analysis"
    ASK_QUESTION = "ask_question"
    VIEW_EXPLANATION = "view_explanation"

    # Samples
    VIEW_SAMPLE = "view_sample"
    NAVIGATE_SAMPLES = "navigate_samples"


# Initialize DB on module load
init_events_db()
