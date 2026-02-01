/**
 * User Behavior Tracker
 *
 * Tracks semantic user actions (not raw clicks) for behavior analysis.
 * Events are sent to /api/analytics/log and stored in the backend.
 *
 * Usage:
 *   Tracker.init({ userId: 'username' });
 *   Tracker.track('select_tab', { tab_name: 'training' });
 */

const Tracker = (function () {
    let sessionId = null;
    let userId = null;
    let initialized = false;

    /**
     * Initialize the tracker.
     * @param {Object} config - Configuration object
     * @param {string} config.userId - Current user's username
     * @param {string} [config.sessionId] - Optional session ID (auto-generated if not provided)
     */
    function init(config = {}) {
        userId = config.userId || null;
        sessionId = config.sessionId || generateSessionId();
        initialized = true;

        // Track page view on init
        track('page_view', {
            context: {
                page: getPageName(),
                url: window.location.pathname
            }
        });

        console.log('[Tracker] Initialized', { userId, sessionId });
    }

    /**
     * Generate a unique session ID.
     */
    function generateSessionId() {
        // Use crypto.randomUUID if available, otherwise fallback
        if (crypto.randomUUID) {
            return crypto.randomUUID();
        }
        return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Get current page name from URL or document title.
     */
    function getPageName() {
        const path = window.location.pathname;
        if (path === '/' || path === '') return 'homepage';
        if (path.startsWith('/dataset/')) return 'dashboard';
        if (path === '/login') return 'login';
        return path.replace(/^\//, '').replace(/\//g, '_');
    }

    /**
     * Track a semantic event.
     * @param {string} eventType - Type of event (e.g., 'select_tab', 'train_submitted')
     * @param {Object} payload - Event data
     * @param {Object} [payload.context] - Page/view context
     * @param {Object} [payload.ui] - UI element info
     * @param {Object} [payload.data] - Domain data
     */
    function track(eventType, payload = {}) {
        if (!initialized) {
            console.warn('[Tracker] Not initialized. Call Tracker.init() first.');
            return;
        }

        const event = {
            event_type: eventType,
            session_id: sessionId,
            timestamp: new Date().toISOString(),
            payload: {
                ...payload,
                // Always include context if not provided
                context: payload.context || {
                    page: getPageName(),
                    url: window.location.pathname
                }
            }
        };

        // Send event to backend (fire-and-forget)
        sendEvent(event);
    }

    /**
     * Send event to backend.
     */
    function sendEvent(event) {
        // Use sendBeacon for reliability (survives page unload)
        // Fall back to fetch for older browsers
        const url = '/api/analytics/log';
        const data = JSON.stringify(event);

        if (navigator.sendBeacon) {
            const blob = new Blob([data], { type: 'application/json' });
            const sent = navigator.sendBeacon(url, blob);
            if (!sent) {
                // Fallback if sendBeacon fails
                fetchEvent(url, data);
            }
        } else {
            fetchEvent(url, data);
        }
    }

    /**
     * Send event via fetch (fallback).
     */
    function fetchEvent(url, data) {
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: data,
            credentials: 'same-origin'
        }).catch(err => {
            console.warn('[Tracker] Failed to send event:', err);
        });
    }

    /**
     * Get current session ID.
     */
    function getSessionId() {
        return sessionId;
    }

    /**
     * Get current user ID.
     */
    function getUserId() {
        return userId;
    }

    // Public API
    return {
        init,
        track,
        getSessionId,
        getUserId
    };
})();

// Standard event types for consistency
const EventTypes = {
    // Navigation
    PAGE_VIEW: 'page_view',
    SELECT_TAB: 'select_tab',

    // Authentication
    LOGIN: 'login',
    LOGOUT: 'logout',

    // Dataset
    SELECT_DATASET: 'select_dataset',
    APPLY_FILTER: 'apply_filter',

    // Training
    TRAIN_SUBMITTED: 'train_submitted',
    TRAIN_CONFIG_CHANGED: 'train_config_changed',

    // Model inspection
    VIEW_MODEL: 'view_model',
    VIEW_METRICS: 'view_metrics',
    VIEW_FEATURE_IMPORTANCE: 'view_feature_importance',
    VIEW_TRAINING_CURVES: 'view_training_curves',
    COMPARE_MODELS: 'compare_models',

    // Analysis
    RUN_ANALYSIS: 'run_analysis',
    ASK_QUESTION: 'ask_question',
    VIEW_EXPLANATION: 'view_explanation',

    // Samples
    VIEW_SAMPLE: 'view_sample',
    NAVIGATE_SAMPLES: 'navigate_samples'
};
