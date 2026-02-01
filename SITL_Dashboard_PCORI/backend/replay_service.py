"""
Event Replay Service

A separate web service for replaying and analyzing user behavior sessions.
Run on port 8767 with: uvicorn backend.replay_service:app --port 8767

Features:
- List user sessions
- View chronological event timeline
- Group events into chapters/workflows
- Playback controls
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .analytics import get_events, get_session_events, list_sessions

app = FastAPI(title="SITL Event Replay Service", version="1.0.0")

# Serve static files from frontend directory
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")


# ============== API Endpoints ==============


@app.get("/api/sessions")
async def get_sessions(
    user_id: Optional[str] = None,
    limit: int = Query(default=100, le=500),
):
    """List all sessions with summary info."""
    sessions = list_sessions(user_id=user_id, limit=limit)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get all events for a specific session."""
    events = get_session_events(session_id)
    return {"session_id": session_id, "events": events, "count": len(events)}


@app.get("/api/sessions/{session_id}/chapters")
async def get_session_chapters(session_id: str):
    """Get events grouped into semantic chapters."""
    events = get_session_events(session_id)
    chapters = group_into_chapters(events)
    return {"session_id": session_id, "chapters": chapters}


@app.get("/api/events")
async def query_events(
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = Query(default=1000, le=5000),
):
    """Query events with filters."""
    events = get_events(
        user_id=user_id,
        event_type=event_type,
        since=since,
        until=until,
        limit=limit,
    )
    return {"events": events, "count": len(events)}


@app.get("/api/users")
async def list_users():
    """List all users who have events."""
    events = get_events(limit=10000)
    users = {}
    for event in events:
        user_id = event.get("user_id")
        if user_id and user_id not in users:
            users[user_id] = {"user_id": user_id, "first_seen": event["timestamp"]}
        if user_id:
            users[user_id]["last_seen"] = event["timestamp"]
    return {"users": list(users.values())}


@app.get("/api/event-types")
async def list_event_types():
    """List all distinct event types."""
    events = get_events(limit=10000)
    types = {}
    for event in events:
        etype = event.get("event_type")
        if etype:
            types[etype] = types.get(etype, 0) + 1
    return {"event_types": [{"type": k, "count": v} for k, v in sorted(types.items())]}


# ============== Chapter Grouping ==============


IDLE_THRESHOLD = timedelta(minutes=5)

CHAPTER_DEFINITIONS = {
    "login": ["login"],
    "navigation": ["page_view", "select_tab", "select_dataset"],
    "training": ["train_submitted", "train_started", "train_completed", "train_failed", "train_config_changed"],
    "evaluation": ["view_model", "view_metrics", "view_feature_importance", "view_training_curves", "compare_models"],
    "analysis": ["run_analysis", "ask_question", "view_explanation"],
    "samples": ["view_sample", "navigate_samples"],
}


def get_chapter_type(event_type: str) -> str:
    """Determine chapter type for an event type."""
    for chapter, types in CHAPTER_DEFINITIONS.items():
        if event_type in types:
            return chapter
    return "misc"


def group_into_chapters(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group events into semantic chapters."""
    if not events:
        return []

    chapters = []
    current = None

    def start_chapter(chapter_type: str, event: Dict):
        return {
            "type": chapter_type,
            "start_time": event["timestamp"],
            "end_time": event["timestamp"],
            "events": [event],
        }

    def close_current():
        nonlocal current
        if current and current["events"]:
            current["end_time"] = current["events"][-1]["timestamp"]
            current["event_count"] = len(current["events"])
            chapters.append(current)
        current = None

    last_time = None

    for event in events:
        event_type = event.get("event_type", "unknown")
        chapter_type = get_chapter_type(event_type)

        # Parse timestamp
        try:
            t = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except:
            t = datetime.now()

        # Check for idle break
        if last_time and (t - last_time) > IDLE_THRESHOLD:
            close_current()

        # Start new chapter if type changes
        if current is None:
            current = start_chapter(chapter_type, event)
        elif current["type"] != chapter_type:
            close_current()
            current = start_chapter(chapter_type, event)
        else:
            current["events"].append(event)

        last_time = t

    close_current()
    return chapters


# ============== Replay UI ==============


REPLAY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SITL Event Replay</title>
    <style>
        :root {
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --border-color: #30363d;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { color: var(--accent-blue); font-size: 1.5rem; }
        .container {
            display: grid;
            grid-template-columns: 280px 1fr 350px;
            height: calc(100vh - 60px);
        }
        .panel {
            border-right: 1px solid var(--border-color);
            overflow-y: auto;
            padding: 1rem;
        }
        .panel:last-child { border-right: none; }
        .panel h2 {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .session-item {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .session-item:hover { border-color: var(--accent-blue); }
        .session-item.active { border-color: var(--accent-green); background: var(--bg-hover); }
        .session-item .user { color: var(--accent-green); font-weight: 500; }
        .session-item .time { color: var(--text-secondary); font-size: 0.8rem; }
        .session-item .count { color: var(--text-secondary); font-size: 0.75rem; }
        .chapters-bar {
            display: flex;
            gap: 4px;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .chapter-block {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            color: white;
        }
        .chapter-block.login { background: #3fb950; }
        .chapter-block.navigation { background: #58a6ff; }
        .chapter-block.training { background: #f5a623; }
        .chapter-block.evaluation { background: #a371f7; }
        .chapter-block.analysis { background: #f85149; }
        .chapter-block.samples { background: #8b949e; }
        .chapter-block.misc { background: #6e7681; }
        .controls {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            align-items: center;
        }
        .controls button {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
        }
        .controls button:hover { background: var(--bg-hover); }
        .controls select {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem;
            border-radius: 4px;
        }
        .timeline { }
        .event-row {
            display: flex;
            gap: 1rem;
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 2px;
            cursor: pointer;
        }
        .event-row:hover { background: var(--bg-hover); }
        .event-row.active { background: var(--bg-card); border: 1px solid var(--accent-blue); }
        .event-row .time {
            color: var(--text-secondary);
            font-size: 0.8rem;
            min-width: 70px;
        }
        .event-row .icon { font-size: 1rem; }
        .event-row .label { flex: 1; }
        .event-row .source {
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 3px;
            background: var(--bg-hover);
            color: var(--text-secondary);
        }
        .event-detail {
            background: var(--bg-card);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .event-detail h3 {
            color: var(--accent-blue);
            margin-bottom: 0.5rem;
        }
        .event-detail .field {
            margin-bottom: 0.5rem;
        }
        .event-detail .field-label {
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        .event-detail .field-value {
            font-family: monospace;
            font-size: 0.85rem;
        }
        .event-detail pre {
            background: var(--bg-dark);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.8rem;
        }
        .loading { text-align: center; padding: 2rem; color: var(--text-secondary); }
        .filter-row {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .filter-row input, .filter-row select {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem;
            border-radius: 4px;
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SITL Event Replay</h1>
        <div id="stats"></div>
    </div>

    <div class="container">
        <!-- Sessions Panel -->
        <div class="panel" id="sessionsPanel">
            <h2>Sessions</h2>
            <div class="filter-row">
                <input type="text" id="userFilter" placeholder="Filter by user...">
            </div>
            <div id="sessionsList" class="loading">Loading sessions...</div>
        </div>

        <!-- Timeline Panel -->
        <div class="panel" id="timelinePanel">
            <h2>Timeline</h2>
            <div id="chaptersBar" class="chapters-bar"></div>
            <div class="controls">
                <button id="btnPrev" title="Previous">&#9664; Prev</button>
                <button id="btnPlay" title="Play/Pause">&#9654; Play</button>
                <button id="btnNext" title="Next">Next &#9654;</button>
                <select id="speedSelect">
                    <option value="1">1x</option>
                    <option value="2">2x</option>
                    <option value="4">4x</option>
                </select>
            </div>
            <div id="timeline" class="timeline">
                <div class="loading">Select a session to view timeline</div>
            </div>
        </div>

        <!-- Details Panel -->
        <div class="panel" id="detailsPanel">
            <h2>Event Details</h2>
            <div id="eventDetails">
                <div class="loading">Select an event to view details</div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        let sessions = [];
        let currentSession = null;
        let events = [];
        let chapters = [];
        let currentIndex = -1;
        let playTimer = null;

        // Icons for event types
        const EVENT_ICONS = {
            'page_view': '📄',
            'select_tab': '📑',
            'select_dataset': '📊',
            'login': '🔐',
            'logout': '🚪',
            'train_submitted': '🚀',
            'train_started': '⚙️',
            'train_completed': '✅',
            'train_failed': '❌',
            'view_model': '🔍',
            'view_metrics': '📈',
            'view_feature_importance': '📊',
            'ask_question': '❓',
            'view_sample': '👤',
        };

        async function loadSessions() {
            try {
                const res = await fetch(`${API_BASE}/api/sessions`);
                const data = await res.json();
                sessions = data.sessions || [];
                renderSessions();
            } catch (e) {
                document.getElementById('sessionsList').innerHTML =
                    '<div style="color: var(--accent-red);">Failed to load sessions</div>';
            }
        }

        function renderSessions() {
            const filter = document.getElementById('userFilter').value.toLowerCase();
            const filtered = sessions.filter(s =>
                !filter || (s.user_id && s.user_id.toLowerCase().includes(filter))
            );

            const container = document.getElementById('sessionsList');
            if (filtered.length === 0) {
                container.innerHTML = '<div class="loading">No sessions found</div>';
                return;
            }

            container.innerHTML = filtered.map(s => `
                <div class="session-item ${currentSession?.session_id === s.session_id ? 'active' : ''}"
                     onclick="selectSession('${s.session_id}')">
                    <div class="user">${s.user_id || 'Unknown'}</div>
                    <div class="time">${formatTime(s.start_time)}</div>
                    <div class="count">${s.event_count} events</div>
                </div>
            `).join('');
        }

        async function selectSession(sessionId) {
            currentSession = sessions.find(s => s.session_id === sessionId);
            renderSessions();

            // Load events
            document.getElementById('timeline').innerHTML = '<div class="loading">Loading events...</div>';
            try {
                const [eventsRes, chaptersRes] = await Promise.all([
                    fetch(`${API_BASE}/api/sessions/${sessionId}`),
                    fetch(`${API_BASE}/api/sessions/${sessionId}/chapters`)
                ]);
                const eventsData = await eventsRes.json();
                const chaptersData = await chaptersRes.json();

                events = eventsData.events || [];
                chapters = chaptersData.chapters || [];
                currentIndex = -1;

                renderChapters();
                renderTimeline();
            } catch (e) {
                document.getElementById('timeline').innerHTML =
                    '<div style="color: var(--accent-red);">Failed to load events</div>';
            }
        }

        function renderChapters() {
            const container = document.getElementById('chaptersBar');
            container.innerHTML = chapters.map((ch, idx) => `
                <div class="chapter-block ${ch.type}" onclick="jumpToChapter(${idx})"
                     title="${ch.event_count} events">
                    ${ch.type} (${ch.event_count})
                </div>
            `).join('');
        }

        function renderTimeline() {
            const container = document.getElementById('timeline');
            if (events.length === 0) {
                container.innerHTML = '<div class="loading">No events in this session</div>';
                return;
            }

            container.innerHTML = events.map((ev, idx) => `
                <div class="event-row ${idx === currentIndex ? 'active' : ''}"
                     onclick="selectEvent(${idx})">
                    <span class="time">${formatTimeShort(ev.timestamp)}</span>
                    <span class="icon">${EVENT_ICONS[ev.event_type] || '•'}</span>
                    <span class="label">${formatEventLabel(ev)}</span>
                    <span class="source">${ev.source}</span>
                </div>
            `).join('');

            // Scroll to active
            if (currentIndex >= 0) {
                const activeRow = container.children[currentIndex];
                if (activeRow) activeRow.scrollIntoView({ block: 'center', behavior: 'smooth' });
            }
        }

        function selectEvent(index) {
            currentIndex = index;
            renderTimeline();
            renderEventDetails(events[index]);
        }

        function renderEventDetails(event) {
            if (!event) {
                document.getElementById('eventDetails').innerHTML =
                    '<div class="loading">Select an event</div>';
                return;
            }

            const container = document.getElementById('eventDetails');
            container.innerHTML = `
                <div class="event-detail">
                    <h3>${EVENT_ICONS[event.event_type] || '•'} ${event.event_type}</h3>
                    <div class="field">
                        <div class="field-label">Time</div>
                        <div class="field-value">${event.timestamp}</div>
                    </div>
                    <div class="field">
                        <div class="field-label">User</div>
                        <div class="field-value">${event.user_id || 'Unknown'}</div>
                    </div>
                    <div class="field">
                        <div class="field-label">Source</div>
                        <div class="field-value">${event.source}</div>
                    </div>
                    <div class="field">
                        <div class="field-label">Session ID</div>
                        <div class="field-value" style="font-size: 0.7rem;">${event.session_id || 'N/A'}</div>
                    </div>
                </div>
                <div class="event-detail">
                    <h3>Payload</h3>
                    <pre>${JSON.stringify(event.payload, null, 2)}</pre>
                </div>
            `;
        }

        function jumpToChapter(chapterIndex) {
            if (chapterIndex < 0 || chapterIndex >= chapters.length) return;
            const chapter = chapters[chapterIndex];
            if (!chapter.events || chapter.events.length === 0) return;

            // Find index of first event in this chapter
            const firstEventId = chapter.events[0].id;
            const eventIndex = events.findIndex(e => e.id === firstEventId);
            if (eventIndex >= 0) {
                selectEvent(eventIndex);
            }
        }

        // Playback controls
        function nextEvent() {
            if (currentIndex < events.length - 1) {
                selectEvent(currentIndex + 1);
            } else {
                stopPlay();
            }
        }

        function prevEvent() {
            if (currentIndex > 0) {
                selectEvent(currentIndex - 1);
            }
        }

        function startPlay() {
            const speed = parseInt(document.getElementById('speedSelect').value) || 1;
            const intervalMs = 1000 / speed;
            if (playTimer) clearInterval(playTimer);
            playTimer = setInterval(nextEvent, intervalMs);
            document.getElementById('btnPlay').innerHTML = '⏸ Pause';
        }

        function stopPlay() {
            if (playTimer) clearInterval(playTimer);
            playTimer = null;
            document.getElementById('btnPlay').innerHTML = '▶ Play';
        }

        function togglePlay() {
            if (playTimer) {
                stopPlay();
            } else {
                if (currentIndex < 0) selectEvent(0);
                startPlay();
            }
        }

        // Formatting helpers
        function formatTime(ts) {
            if (!ts) return '';
            const d = new Date(ts);
            return d.toLocaleString();
        }

        function formatTimeShort(ts) {
            if (!ts) return '';
            const d = new Date(ts);
            return d.toLocaleTimeString();
        }

        function formatEventLabel(event) {
            const type = event.event_type;
            const payload = event.payload || {};

            if (type === 'select_tab') return `Tab: ${payload.ui?.tab_name || 'unknown'}`;
            if (type === 'select_dataset') return `Dataset: ${payload.data?.dataset_id || 'unknown'}`;
            if (type === 'train_submitted') return `Train: ${payload.data?.model_type || payload.model_type || 'model'}`;
            if (type === 'login') return payload.success ? 'Login success' : 'Login failed';
            if (type === 'page_view') return `Page: ${payload.context?.page || 'unknown'}`;

            return type.replace(/_/g, ' ');
        }

        // Event listeners
        document.getElementById('userFilter').addEventListener('input', renderSessions);
        document.getElementById('btnPrev').addEventListener('click', prevEvent);
        document.getElementById('btnPlay').addEventListener('click', togglePlay);
        document.getElementById('btnNext').addEventListener('click', nextEvent);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === 'n') nextEvent();
            if (e.key === 'ArrowLeft' || e.key === 'p') prevEvent();
            if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        });

        // Init
        loadSessions();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def replay_ui():
    """Serve the replay UI."""
    return HTMLResponse(content=REPLAY_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8767)
