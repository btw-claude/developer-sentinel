/**
 * Centralized UI message constants for the Developer Sentinel dashboard.
 *
 * All client-facing warning, error, and informational strings are defined here
 * to support future i18n/localization and ensure consistency across templates.
 *
 * Usage:
 *   UI_MESSAGES.TOAST.TOGGLE_FAILED  // "Toggle operation failed"
 *   UI_MESSAGES.CSRF.REFRESH_FAILED  // "CSRF token refresh failed..."
 *
 * Categories:
 * - TOAST: Messages displayed via showToast() notifications
 * - CSRF: CSRF token related messages
 * - LOG_VIEWER: Log viewer page messages
 * - FORM_VALIDATION: Form input validation messages
 * - DEBUG: Developer-facing console.warn/error messages (not user-visible)
 *
 * @module ui-messages
 * @see DS-740
 */

/* eslint-disable no-unused-vars */
var UI_MESSAGES = Object.freeze({

    /** Toast notification messages shown to users via showToast(). */
    TOAST: Object.freeze({
        // Toggle operations (base.html - handleToggleResponse)
        TOGGLE_FAILED: 'Toggle operation failed',
        PARSE_RESPONSE_FAILED: 'Failed to process server response',
        ORCHESTRATION_ENABLED: 'Orchestration enabled successfully',
        ORCHESTRATION_DISABLED: 'Orchestration disabled successfully',

        // Delete operations (base.html - handleDeleteResponse)
        DELETE_SUCCESS: function(name) { return 'Orchestration \'' + name + '\' deleted successfully'; },
        DELETE_FAILED: function(name) { return 'Failed to delete orchestration \'' + name + '\''; },
        RATE_LIMIT_EXCEEDED: 'Rate limit exceeded. Please wait before trying again.',

        // Create operations (orchestration-forms.js)
        CREATE_SUCCESS: function(name) { return 'Orchestration \'' + name + '\' created successfully'; },
        CREATE_CSRF_REFRESH_FAILED: 'CSRF token refresh failed. Please reload the page.',
        CREATE_CSRF_VALIDATION_FAILED: 'CSRF token validation failed. Please reload the page.',
        VALIDATION_ERROR: 'Validation error',
        CREATE_FAILED: 'Failed to create orchestration',
        CREATE_NETWORK_ERROR: 'Network error while creating orchestration',

        // Edit operations (orchestration-forms.js)
        EDIT_SUCCESS: function(name) { return 'Orchestration \'' + name + '\' updated successfully'; },
        EDIT_FAILED: 'Failed to update orchestration',
        EDIT_NETWORK_ERROR: 'Network error while updating orchestration',

        // Log viewer invalid URL parameters (log_viewer.html)
        LOG_FILE_NOT_FOUND: function(file) { return 'Log file \'' + file + '\' not found in selected orchestration'; },
        ORCHESTRATION_NOT_FOUND: function(name) { return 'Orchestration \'' + name + '\' not found'; }
    }),

    /** CSRF token messages (orchestration_create_form.html). */
    CSRF: Object.freeze({
        REFRESH_FAILED_CONSOLE: 'CSRF token refresh failed on form load; form submissions may fail with 403. Try reloading the page or check browser console for network errors.'
    }),

    /** Form validation messages shown to users (orchestration-forms.js). */
    FORM_VALIDATION: Object.freeze({
        NAME_REQUIRED: 'Orchestration name is required',
        TARGET_FILE_REQUIRED: 'Target file must be selected',
        NEW_FILE_NAME_REQUIRED: 'New file name is required'
    }),

    /** Log viewer UI text (log_viewer.html). */
    LOG_VIEWER: Object.freeze({
        NO_FILE_SELECTED: 'No log file selected',
        SELECT_PROMPT: '\uD83D\uDCCB Select an orchestration and log file to view',
        SELECT_PROMPT_SUBTITLE: 'Log files are updated in real-time via SSE',
        LOADING: 'Loading...',
        CONNECTED: 'Connected',
        DISCONNECTED: 'Disconnected',
        PAUSE: 'Pause',
        RESUME: 'Resume',
        PAUSE_ICON: '\u23F8',
        RESUME_ICON: '\u25B6'
    }),

    /**
     * Developer-facing console messages for debugging.
     * These are not user-visible but centralized for consistency.
     */
    DEBUG: Object.freeze({
        // base.html - collapsible sections
        COLLAPSIBLE_NO_PARENT: 'toggleCollapsibleSection: Could not find parent .collapsible-section',
        COLLAPSIBLE_NO_CONTENT: 'toggleCollapsibleSection: Could not find .collapsible-content within section',
        COLLAPSIBLE_NO_ICON: 'toggleCollapsibleSection: Could not find .collapse-icon within header',
        NESTED_NO_PARENT: 'toggleNestedAccordion: Could not find parent .nested-accordion-item',
        DETAIL_ROW_NOT_FOUND: function(name) { return 'toggleStepDetail: Could not find detail row for: ' + name; },
        ELEMENT_NOT_FOUND: function(id) { return 'isExpanded: Element not found: ' + id; },
        INIT_NO_CONTAINER: 'initCollapsibleSections: Could not find main container for event delegation',
        TOGGLE_PARSE_ERROR: 'handleToggleResponse: Failed to parse JSON response:',
        DELETE_PARSE_ERROR: 'handleDeleteResponse: Failed to parse JSON response:',

        // orchestration_create_form.html
        FILES_LOAD_FAILED: 'Failed to load step files:',

        // orchestration-forms.js
        EDIT_FORM_NULL: 'submitOrchestrationEdit: formElement is null',
        EDIT_FETCH_ERROR: 'submitOrchestrationEdit: fetch error:',
        CREATE_FORM_NULL: 'submitOrchestrationCreate: formElement is null',
        CREATE_FETCH_ERROR: 'submitOrchestrationCreate: fetch error:',

        // log_viewer.html - URL parameter validation
        INVALID_FILE_PARAM: function(file, orch) { return 'Invalid URL parameter: file \'' + file + '\' not found in orchestration \'' + orch + '\''; },
        INVALID_STEP_PARAM: function(orch) { return 'Invalid URL parameter: step \'' + orch + '\' not found'; }
    })
});
