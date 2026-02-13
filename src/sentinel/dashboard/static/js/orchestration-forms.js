/**
 * Orchestration Forms - Form submission and validation logic
 *
 * This module provides JavaScript functions for submitting orchestration
 * create and edit forms via the REST API endpoints. Extracted from base.html
 * for better maintainability and code organization.
 *
 * Dependencies:
 * - htmx: For AJAX requests and DOM updates
 * - showToast: Global toast notification function (defined in base.html)
 * - UI_MESSAGES: Centralized message constants (defined in ui-messages.js)
 *
 * CSRF Token Auto-Refresh (DS-737):
 * When a page is refreshed, embedded CSRF tokens become invalid (single-use).
 * The refreshCsrfToken() function fetches a fresh token from the API and
 * updates the hidden input, preventing 403 errors on form submission after
 * page refresh.
 *
 * Shared CSRF Retry Helper (DS-1091):
 * The fetchWithCsrfRetry() function encapsulates the common pattern of
 * CSRF token pre-fetch, header injection, 403 retry logic, and standard
 * error handling (422, 429) used by all form submission functions.
 *
 * Defensive Hardening (DS-1092):
 * Added Content-Type check before parsing responses as JSON (handles
 * non-JSON 403 pages from reverse proxies), inline documentation for
 * the DOM hidden-input fallback, and a console.warn when both the API
 * refresh and DOM csrf_token element are unavailable.
 *
 * Content-Type Guard Narrowing (DS-1093):
 * Narrowed the Content-Type guard to only apply to non-2xx responses,
 * preventing successful responses with unexpected Content-Types from
 * silently resolving with empty data and bypassing the onSuccess callback.
 *
 * 2xx Non-JSON Content-Type Diagnostic Warning (DS-1094):
 * Added a console.warn when a successful 2xx response arrives with a
 * non-JSON Content-Type (e.g., text/html from a misconfigured proxy).
 * The response is still parsed via response.json() (which will throw a
 * SyntaxError caught by the .catch handler), but the warning helps
 * developers quickly diagnose proxy/CDN misconfiguration issues.
 *
 * @module orchestration-forms
 */

/**
 * Fetch a fresh CSRF token from the server and update the hidden input.
 *
 * This function is called on page load and after a 403 CSRF failure to
 * transparently refresh the token. If the hidden input element does not
 * exist, the function is a no-op.
 *
 * @returns {Promise<string|null>} The new CSRF token, or null on failure
 */
function refreshCsrfToken() {
    return fetch('/api/csrf-token')
        .then(function(response) {
            if (!response.ok) return null;
            return response.json();
        })
        .then(function(data) {
            if (data && data.csrf_token) {
                var el = document.getElementById('csrf_token');
                if (el) el.value = data.csrf_token;
                return data.csrf_token;
            }
            return null;
        })
        .catch(function() {
            return null;
        });
}

/**
 * Perform a fetch request with automatic CSRF token handling and retry.
 *
 * Encapsulates the shared pattern used by all state-changing form submissions:
 * 1. Pre-fetches a fresh CSRF token via refreshCsrfToken()
 * 2. Makes the fetch request with the X-CSRF-Token header
 * 3. On 403 + first attempt, refreshes the token and retries once
 * 4. On 403 + retry, shows a CSRF validation failed toast
 * 5. Handles 422 (validation), 429 (rate limit), and generic error cases
 *
 * Extracted from the identical inner helpers (doCreate, doEdit, doGitHubEdit,
 * doTriggerEdit) to reduce code duplication (DS-1091).
 *
 * @param {string} url - The API endpoint URL
 * @param {Object} fetchOptions - Options for the fetch call (method, body).
 *     The Content-Type and X-CSRF-Token headers are added automatically.
 * @param {Object} messageConfig - Toast message configuration
 * @param {string} messageConfig.csrfRefreshFailed - Message when CSRF refresh fails
 * @param {string} messageConfig.csrfValidationFailed - Message when CSRF validation fails after retry
 * @param {string} messageConfig.genericError - Default error message for non-specific failures
 * @param {string} messageConfig.networkError - Message for network/fetch errors
 * @param {string} messageConfig.debugFetchError - Console error prefix for fetch failures
 * @param {function} messageConfig.onSuccess - Callback invoked with the parsed response data on success
 */
function fetchWithCsrfRetry(url, fetchOptions, messageConfig) {
    // Read CSRF token from hidden input as fallback
    var csrfToken = document.getElementById('csrf_token');

    /**
     * Internal helper that performs the actual fetch with CSRF retry logic.
     *
     * @param {string} token - CSRF token to include in the request header
     * @param {boolean} isRetry - Whether this is a retry after token refresh
     */
    function doFetch(token, isRetry) {
        var headers = { 'Content-Type': 'application/json' };
        headers['X-CSRF-Token'] = token;

        var options = {
            method: fetchOptions.method,
            headers: headers,
            body: fetchOptions.body
        };

        fetch(url, options).then(function(response) {
            // Defensive Content-Type check: some reverse proxies or WAFs return
            // non-JSON 403 pages (e.g., HTML error pages).  Attempting to parse
            // these with response.json() would throw a SyntaxError that falls
            // through to the catch handler, masking the real 403 cause.  By
            // checking the Content-Type header first, we can surface a clear
            // CSRF / permission error toast instead (DS-1092).
            //
            // Narrowed to non-2xx responses only (DS-1093): if a successful 2xx
            // response arrives with a non-JSON Content-Type (e.g., text/html
            // from a misconfigured proxy), we should still attempt to parse it
            // rather than silently resolving with empty data and bypassing the
            // onSuccess callback.
            var contentType = response.headers.get('Content-Type') || '';
            if (contentType.indexOf('application/json') === -1 && !(response.status >= 200 && response.status < 300)) {
                return { status: response.status, data: {} };
            }
            // Diagnostic warning for 2xx responses with unexpected Content-Type
            // (DS-1094): helps developers quickly diagnose proxy/CDN
            // misconfiguration issues.  The response.json() call below will
            // throw a SyntaxError if the body is not valid JSON, which is
            // caught by the .catch handler.
            if (contentType.indexOf('application/json') === -1 && response.status >= 200 && response.status < 300) {
                console.warn('fetchWithCsrfRetry: 2xx response with unexpected Content-Type: ' + contentType);
            }
            return response.json().then(function(data) {
                return { status: response.status, data: data };
            });
        }).then(function(result) {
            if (result.status >= 200 && result.status < 300 && result.data.success) {
                messageConfig.onSuccess(result.data);
            } else if (result.status === 403 && !isRetry) {
                // CSRF token invalid (e.g., page refresh) - auto-refresh and retry
                return refreshCsrfToken().then(function(newToken) {
                    if (newToken) {
                        doFetch(newToken, true);
                    } else {
                        showToast('error', messageConfig.csrfRefreshFailed);
                    }
                });
            } else if (result.status === 403) {
                showToast('error', messageConfig.csrfValidationFailed);
            } else if (result.status === 422) {
                showToast('error', result.data.detail || UI_MESSAGES.TOAST.VALIDATION_ERROR);
            } else if (result.status === 429) {
                showToast('warning', UI_MESSAGES.TOAST.RATE_LIMIT_EXCEEDED);
            } else {
                showToast('error', result.data.detail || messageConfig.genericError);
            }
        }).catch(function(error) {
            console.error(messageConfig.debugFetchError, error);
            showToast('error', messageConfig.networkError);
        });
    }

    // Fetch a fresh CSRF token first, then submit
    refreshCsrfToken().then(function(freshToken) {
        // DOM fallback: when refreshCsrfToken() returns null (e.g., the CSRF
        // endpoint is unreachable or returns a non-OK status), fall back to the
        // value of the hidden <input id="csrf_token"> rendered by the server
        // template.  This token may be stale after a page refresh, but the 403
        // retry logic above will auto-refresh it on failure (DS-1092).
        var token = freshToken || (csrfToken ? csrfToken.value : '');
        if (!freshToken && (!csrfToken || !csrfToken.value)) {
            // Both the API refresh and the DOM hidden input are unavailable.
            // This typically indicates a template rendering issue where the
            // server did not inject the csrf_token hidden input (DS-1092).
            console.warn('fetchWithCsrfRetry: CSRF token unavailable from both API refresh and DOM hidden input. Requests will likely fail with 403.');
        }
        doFetch(token, false);
    });
}

/**
 * Encode a file path by encoding each segment individually.
 *
 * Unlike encodeURIComponent(filePath), which encodes slashes to %2F
 * (potentially causing routing issues behind reverse proxies), this
 * function splits on "/" and encodes each segment separately, then
 * re-joins with literal slashes (DS-1088).
 *
 * @param {string} filePath - The relative file path to encode
 * @returns {string} The path with each segment URI-encoded
 */
function encodeFilePath(filePath) {
    return filePath.split('/').map(encodeURIComponent).join('/');
}

/**
 * Split a comma-separated or newline-separated string into an array.
 *
 * Helper function to parse form inputs that accept multiple values.
 * Handles both comma-separated and newline-separated formats.
 * Empty or whitespace-only values are filtered out.
 *
 * @param {string} value - The string to split
 * @param {string} [separator=','] - The separator to use (',', '\n', etc.)
 * @returns {string[]} Array of trimmed non-empty strings
 */
function splitList(value, separator) {
    if (!value || !value.trim()) return [];
    var sep = separator || ',';
    return value.split(sep === '\n' ? /\r?\n/ : sep)
        .map(function(s) { return s.trim(); })
        .filter(function(s) { return s.length > 0; });
}

/**
 * Submit an orchestration edit form via the PUT API endpoint.
 *
 * Collects form fields, builds nested JSON matching OrchestrationEditRequest,
 * and submits via fetchWithCsrfRetry with automatic CSRF token handling.
 * Shows toast notification on result and reloads the detail view on success.
 *
 * @param {HTMLFormElement} formElement - The form element to collect data from
 * @param {string} name - The orchestration name to update
 */
function submitOrchestrationEdit(formElement, name) {
    if (!formElement) {
        console.error(UI_MESSAGES.DEBUG.EDIT_FORM_NULL);
        return;
    }

    // Build the trigger section
    var triggerSource = formElement.querySelector('input[name="trigger_source"]:checked');
    var trigger = {};
    if (triggerSource) {
        trigger.source = triggerSource.value;
        if (triggerSource.value === 'jira') {
            var project = formElement.querySelector('#jira_project');
            if (project && project.value.trim()) trigger.project = project.value.trim();
            var jqlFilter = formElement.querySelector('#jira_jql_filter');
            if (jqlFilter && jqlFilter.value.trim()) trigger.jql_filter = jqlFilter.value.trim();
            var tags = formElement.querySelector('#jira_tags');
            if (tags && tags.value.trim()) trigger.tags = splitList(tags.value);
        } else {
            var projNum = formElement.querySelector('#github_project_number');
            if (projNum && projNum.value) trigger.project_number = parseInt(projNum.value, 10);
            var projScope = formElement.querySelector('input[name="github_project_scope"]:checked');
            if (projScope) trigger.project_scope = projScope.value;
            var projOwner = formElement.querySelector('#github_project_owner');
            if (projOwner && projOwner.value.trim()) trigger.project_owner = projOwner.value.trim();
            var projFilter = formElement.querySelector('#github_project_filter');
            if (projFilter && projFilter.value.trim()) trigger.project_filter = projFilter.value.trim();
            var labels = formElement.querySelector('#github_labels');
            if (labels && labels.value.trim()) trigger.labels = splitList(labels.value);
        }
    }

    // Build the agent section
    var agent = {};
    var agentType = formElement.querySelector('#agent_type');
    if (agentType && agentType.value) agent.agent_type = agentType.value;
    var cursorMode = formElement.querySelector('#cursor_mode');
    if (cursorMode && agentType && agentType.value === 'cursor') {
        agent.cursor_mode = cursorMode.value;
    }
    var model = formElement.querySelector('#model');
    if (model && model.value.trim()) agent.model = model.value.trim();
    var timeout = formElement.querySelector('#timeout_seconds');
    if (timeout && timeout.value) agent.timeout_seconds = parseInt(timeout.value, 10);
    var prompt = formElement.querySelector('#prompt');
    if (prompt) agent.prompt = prompt.value;

    // Build GitHub context sub-object
    var ghHost = formElement.querySelector('#github_host');
    var ghOrg = formElement.querySelector('#github_org');
    var ghRepo = formElement.querySelector('#github_repo');
    var ghBranch = formElement.querySelector('#github_branch');
    var ghCreateBranch = formElement.querySelector('#github_create_branch');
    var ghBaseBranch = formElement.querySelector('#github_base_branch');
    var github = {};
    var hasGithub = false;
    if (ghHost && ghHost.value.trim()) { github.host = ghHost.value.trim(); hasGithub = true; }
    if (ghOrg && ghOrg.value.trim()) { github.org = ghOrg.value.trim(); hasGithub = true; }
    if (ghRepo && ghRepo.value.trim()) { github.repo = ghRepo.value.trim(); hasGithub = true; }
    if (ghBranch && ghBranch.value.trim()) { github.branch = ghBranch.value.trim(); hasGithub = true; }
    if (ghCreateBranch) { github.create_branch = ghCreateBranch.checked; hasGithub = true; }
    if (ghBaseBranch && ghBaseBranch.value.trim()) { github.base_branch = ghBaseBranch.value.trim(); hasGithub = true; }
    if (hasGithub) agent.github = github;

    // Build the retry section
    var retry = {};
    var maxAttempts = formElement.querySelector('#max_attempts');
    if (maxAttempts && maxAttempts.value) retry.max_attempts = parseInt(maxAttempts.value, 10);
    var successPatterns = formElement.querySelector('#success_patterns');
    if (successPatterns && successPatterns.value.trim()) retry.success_patterns = splitList(successPatterns.value, '\n');
    var failurePatterns = formElement.querySelector('#failure_patterns');
    if (failurePatterns && failurePatterns.value.trim()) retry.failure_patterns = splitList(failurePatterns.value, '\n');
    var defaultStatus = formElement.querySelector('input[name="default_status"]:checked');
    if (defaultStatus) retry.default_status = defaultStatus.value;
    var defaultOutcome = formElement.querySelector('#default_outcome');
    if (defaultOutcome && defaultOutcome.value.trim()) retry.default_outcome = defaultOutcome.value.trim();

    // Build the outcomes section
    var outcomeRows = formElement.querySelectorAll('.outcome-row');
    var outcomes = [];
    outcomeRows.forEach(function(row) {
        var outName = row.querySelector('input[name="outcome_name"]');
        var outPatterns = row.querySelector('textarea[name="outcome_patterns"]');
        var outAddTag = row.querySelector('input[name="outcome_add_tag"]');
        if (outName && outName.value.trim()) {
            var outcome = { name: outName.value.trim() };
            if (outPatterns && outPatterns.value.trim()) outcome.patterns = splitList(outPatterns.value, '\n');
            if (outAddTag && outAddTag.value.trim()) outcome.add_tag = outAddTag.value.trim();
            outcomes.push(outcome);
        }
    });

    // Build the lifecycle section
    var lifecycle = {};
    var onStartAddTag = formElement.querySelector('#on_start_add_tag');
    if (onStartAddTag && onStartAddTag.value.trim()) lifecycle.on_start_add_tag = onStartAddTag.value.trim();
    var onCompleteRemoveTag = formElement.querySelector('#on_complete_remove_tag');
    if (onCompleteRemoveTag && onCompleteRemoveTag.value.trim()) lifecycle.on_complete_remove_tag = onCompleteRemoveTag.value.trim();
    var onCompleteAddTag = formElement.querySelector('#on_complete_add_tag');
    if (onCompleteAddTag && onCompleteAddTag.value.trim()) lifecycle.on_complete_add_tag = onCompleteAddTag.value.trim();
    var onFailureAddTag = formElement.querySelector('#on_failure_add_tag');
    if (onFailureAddTag && onFailureAddTag.value.trim()) lifecycle.on_failure_add_tag = onFailureAddTag.value.trim();

    // Build the final request body
    var body = {};
    if (Object.keys(trigger).length > 0) body.trigger = trigger;
    if (Object.keys(agent).length > 0) body.agent = agent;
    if (Object.keys(retry).length > 0) body.retry = retry;
    if (outcomes.length > 0) body.outcomes = outcomes;
    if (Object.keys(lifecycle).length > 0) body.lifecycle = lifecycle;

    fetchWithCsrfRetry(
        '/api/orchestrations/' + encodeURIComponent(name),
        { method: 'PUT', body: JSON.stringify(body) },
        {
            csrfRefreshFailed: UI_MESSAGES.TOAST.EDIT_CSRF_REFRESH_FAILED,
            csrfValidationFailed: UI_MESSAGES.TOAST.EDIT_CSRF_VALIDATION_FAILED,
            genericError: UI_MESSAGES.TOAST.EDIT_FAILED,
            networkError: UI_MESSAGES.TOAST.EDIT_NETWORK_ERROR,
            debugFetchError: UI_MESSAGES.DEBUG.EDIT_FETCH_ERROR,
            onSuccess: function() {
                showToast('success', UI_MESSAGES.TOAST.EDIT_SUCCESS(name));
                // Reload the detail view to show updated values
                var detailContainer = formElement.closest('.orchestration-detail');
                if (detailContainer) {
                    htmx.ajax('GET', '/partials/orchestration_detail/' + encodeURIComponent(name), {
                        target: detailContainer,
                        swap: 'outerHTML'
                    });
                }
            }
        }
    );
}

/**
 * Submit an orchestration create form via the POST API endpoint.
 *
 * Collects form fields, builds nested JSON matching OrchestrationCreateRequest,
 * and submits via fetchWithCsrfRetry with automatic CSRF token handling.
 * Shows toast notification on result and hides the creation form on success.
 *
 * @param {HTMLFormElement} formElement - The form element to collect data from
 */
function submitOrchestrationCreate(formElement) {
    if (!formElement) {
        console.error(UI_MESSAGES.DEBUG.CREATE_FORM_NULL);
        return;
    }

    // Get orchestration name (required)
    var nameInput = formElement.querySelector('#create_name');
    if (!nameInput || !nameInput.value.trim()) {
        showToast('error', UI_MESSAGES.FORM_VALIDATION.NAME_REQUIRED);
        return;
    }
    var name = nameInput.value.trim();

    // Get target file (required)
    var targetFileSelector = formElement.querySelector('#target_file_selector');
    var newFileNameInput = formElement.querySelector('#new_file_name');
    var targetFile = '';

    if (!targetFileSelector || !targetFileSelector.value) {
        showToast('error', UI_MESSAGES.FORM_VALIDATION.TARGET_FILE_REQUIRED);
        return;
    }

    if (targetFileSelector.value === '__new__') {
        if (!newFileNameInput || !newFileNameInput.value.trim()) {
            showToast('error', UI_MESSAGES.FORM_VALIDATION.NEW_FILE_NAME_REQUIRED);
            return;
        }
        targetFile = newFileNameInput.value.trim();
    } else {
        targetFile = targetFileSelector.value;
    }

    // Build request body starting with required fields
    var body = {
        name: name,
        target_file: targetFile
    };

    // Add enabled field
    var enabledInput = formElement.querySelector('#create_enabled');
    if (enabledInput) {
        body.enabled = enabledInput.checked;
    }

    // Add max_concurrent if provided
    var maxConcurrent = formElement.querySelector('#create_max_concurrent');
    if (maxConcurrent && maxConcurrent.value) {
        body.max_concurrent = parseInt(maxConcurrent.value, 10);
    }

    // Build the trigger section
    var triggerSource = formElement.querySelector('input[name="trigger_source"]:checked');
    var trigger = {};
    if (triggerSource) {
        trigger.source = triggerSource.value;
        if (triggerSource.value === 'jira') {
            var project = formElement.querySelector('#jira_project');
            if (project && project.value.trim()) trigger.project = project.value.trim();
            var jqlFilter = formElement.querySelector('#jira_jql_filter');
            if (jqlFilter && jqlFilter.value.trim()) trigger.jql_filter = jqlFilter.value.trim();
            var tags = formElement.querySelector('#jira_tags');
            if (tags && tags.value.trim()) trigger.tags = splitList(tags.value);
        } else {
            var projNum = formElement.querySelector('#github_project_number');
            if (projNum && projNum.value) trigger.project_number = parseInt(projNum.value, 10);
            var projScope = formElement.querySelector('input[name="github_project_scope"]:checked');
            if (projScope) trigger.project_scope = projScope.value;
            var projOwner = formElement.querySelector('#github_project_owner');
            if (projOwner && projOwner.value.trim()) trigger.project_owner = projOwner.value.trim();
            var projFilter = formElement.querySelector('#github_project_filter');
            if (projFilter && projFilter.value.trim()) trigger.project_filter = projFilter.value.trim();
            var labels = formElement.querySelector('#github_labels');
            if (labels && labels.value.trim()) trigger.labels = splitList(labels.value);
        }
    }
    if (Object.keys(trigger).length > 0) body.trigger = trigger;

    // Build the agent section
    var agent = {};
    var agentType = formElement.querySelector('#agent_type');
    if (agentType && agentType.value) agent.agent_type = agentType.value;
    var cursorMode = formElement.querySelector('#cursor_mode');
    if (cursorMode && agentType && agentType.value === 'cursor') {
        agent.cursor_mode = cursorMode.value;
    }
    var model = formElement.querySelector('#model');
    if (model && model.value.trim()) agent.model = model.value.trim();
    var timeout = formElement.querySelector('#timeout_seconds');
    if (timeout && timeout.value) agent.timeout_seconds = parseInt(timeout.value, 10);
    var prompt = formElement.querySelector('#prompt');
    if (prompt && prompt.value.trim()) agent.prompt = prompt.value;

    // Build GitHub context sub-object
    var ghHost = formElement.querySelector('#github_host');
    var ghOrg = formElement.querySelector('#github_org');
    var ghRepo = formElement.querySelector('#github_repo');
    var ghBranch = formElement.querySelector('#github_branch');
    var ghCreateBranch = formElement.querySelector('#github_create_branch');
    var ghBaseBranch = formElement.querySelector('#github_base_branch');
    var github = {};
    var hasGithub = false;
    if (ghHost && ghHost.value.trim()) { github.host = ghHost.value.trim(); hasGithub = true; }
    if (ghOrg && ghOrg.value.trim()) { github.org = ghOrg.value.trim(); hasGithub = true; }
    if (ghRepo && ghRepo.value.trim()) { github.repo = ghRepo.value.trim(); hasGithub = true; }
    if (ghBranch && ghBranch.value.trim()) { github.branch = ghBranch.value.trim(); hasGithub = true; }
    if (ghCreateBranch) { github.create_branch = ghCreateBranch.checked; hasGithub = true; }
    if (ghBaseBranch && ghBaseBranch.value.trim()) { github.base_branch = ghBaseBranch.value.trim(); hasGithub = true; }
    if (hasGithub) agent.github = github;

    if (Object.keys(agent).length > 0) body.agent = agent;

    // Build the retry section
    var retry = {};
    var maxAttempts = formElement.querySelector('#max_attempts');
    if (maxAttempts && maxAttempts.value) retry.max_attempts = parseInt(maxAttempts.value, 10);
    var successPatterns = formElement.querySelector('#success_patterns');
    if (successPatterns && successPatterns.value.trim()) retry.success_patterns = splitList(successPatterns.value, '\n');
    var failurePatterns = formElement.querySelector('#failure_patterns');
    if (failurePatterns && failurePatterns.value.trim()) retry.failure_patterns = splitList(failurePatterns.value, '\n');
    var defaultStatus = formElement.querySelector('input[name="default_status"]:checked');
    if (defaultStatus) retry.default_status = defaultStatus.value;
    var defaultOutcome = formElement.querySelector('#default_outcome');
    if (defaultOutcome && defaultOutcome.value.trim()) retry.default_outcome = defaultOutcome.value.trim();
    if (Object.keys(retry).length > 0) body.retry = retry;

    // Build the outcomes section
    var outcomeRows = formElement.querySelectorAll('.outcome-row');
    var outcomes = [];
    outcomeRows.forEach(function(row) {
        var outName = row.querySelector('input[name="outcome_name"]');
        var outPatterns = row.querySelector('textarea[name="outcome_patterns"]');
        var outAddTag = row.querySelector('input[name="outcome_add_tag"]');
        if (outName && outName.value.trim()) {
            var outcome = { name: outName.value.trim() };
            if (outPatterns && outPatterns.value.trim()) outcome.patterns = splitList(outPatterns.value, '\n');
            if (outAddTag && outAddTag.value.trim()) outcome.add_tag = outAddTag.value.trim();
            outcomes.push(outcome);
        }
    });
    if (outcomes.length > 0) body.outcomes = outcomes;

    // Build the lifecycle section
    var lifecycle = {};
    var onStartAddTag = formElement.querySelector('#on_start_add_tag');
    if (onStartAddTag && onStartAddTag.value.trim()) lifecycle.on_start_add_tag = onStartAddTag.value.trim();
    var onCompleteRemoveTag = formElement.querySelector('#on_complete_remove_tag');
    if (onCompleteRemoveTag && onCompleteRemoveTag.value.trim()) lifecycle.on_complete_remove_tag = onCompleteRemoveTag.value.trim();
    var onCompleteAddTag = formElement.querySelector('#on_complete_add_tag');
    if (onCompleteAddTag && onCompleteAddTag.value.trim()) lifecycle.on_complete_add_tag = onCompleteAddTag.value.trim();
    var onFailureAddTag = formElement.querySelector('#on_failure_add_tag');
    if (onFailureAddTag && onFailureAddTag.value.trim()) lifecycle.on_failure_add_tag = onFailureAddTag.value.trim();
    if (Object.keys(lifecycle).length > 0) body.lifecycle = lifecycle;

    fetchWithCsrfRetry(
        '/api/orchestrations',
        { method: 'POST', body: JSON.stringify(body) },
        {
            csrfRefreshFailed: UI_MESSAGES.TOAST.CREATE_CSRF_REFRESH_FAILED,
            csrfValidationFailed: UI_MESSAGES.TOAST.CREATE_CSRF_VALIDATION_FAILED,
            genericError: UI_MESSAGES.TOAST.CREATE_FAILED,
            networkError: UI_MESSAGES.TOAST.CREATE_NETWORK_ERROR,
            debugFetchError: UI_MESSAGES.DEBUG.CREATE_FETCH_ERROR,
            onSuccess: function() {
                showToast('success', UI_MESSAGES.TOAST.CREATE_SUCCESS(name));
                // Hide the creation form
                document.getElementById('create-form-container').style.display = 'none';
                // Reload the orchestrations list (HTMX will auto-refresh on next poll)
            }
        }
    );
}

/**
 * Submit the file-level GitHub context edit form via the PUT API endpoint.
 *
 * Collects form fields from the file-level GitHub edit form, builds a JSON
 * payload matching FileGitHubEditRequest, and submits via fetchWithCsrfRetry
 * with automatic CSRF token handling. Shows toast notification on result and
 * reloads the edit form partial on success (DS-1082).
 *
 * @param {string} filePath - The relative file path for the orchestration file
 * @param {HTMLFormElement} formElement - The form element to collect data from
 */
function submitFileGitHubEdit(filePath, formElement) {
    if (!formElement) {
        console.error(UI_MESSAGES.DEBUG.FILE_GITHUB_EDIT_FORM_NULL);
        return;
    }

    // Build the request body from form fields
    var body = {};
    var host = formElement.querySelector('#file_github_host');
    if (host && host.value.trim()) body.host = host.value.trim();
    var org = formElement.querySelector('#file_github_org');
    if (org && org.value.trim()) body.org = org.value.trim();
    var repo = formElement.querySelector('#file_github_repo');
    if (repo && repo.value.trim()) body.repo = repo.value.trim();
    var branch = formElement.querySelector('#file_github_branch');
    if (branch && branch.value.trim()) body.branch = branch.value.trim();
    var createBranch = formElement.querySelector('#file_github_create_branch');
    if (createBranch) body.create_branch = createBranch.checked;
    var baseBranch = formElement.querySelector('#file_github_base_branch');
    if (baseBranch && baseBranch.value.trim()) body.base_branch = baseBranch.value.trim();

    fetchWithCsrfRetry(
        '/api/orchestrations/files/' + encodeFilePath(filePath) + '/github',
        { method: 'PUT', body: JSON.stringify(body) },
        {
            csrfRefreshFailed: UI_MESSAGES.TOAST.FILE_EDIT_CSRF_REFRESH_FAILED,
            csrfValidationFailed: UI_MESSAGES.TOAST.FILE_EDIT_CSRF_VALIDATION_FAILED,
            genericError: UI_MESSAGES.TOAST.FILE_GITHUB_EDIT_FAILED,
            networkError: UI_MESSAGES.TOAST.FILE_EDIT_NETWORK_ERROR,
            debugFetchError: UI_MESSAGES.DEBUG.FILE_GITHUB_EDIT_FETCH_ERROR,
            onSuccess: function() {
                showToast('success', UI_MESSAGES.TOAST.FILE_GITHUB_EDIT_SUCCESS);
                // Reload the edit form to show updated values
                var detailContainer = formElement.closest('.orchestration-detail');
                if (detailContainer) {
                    htmx.ajax('GET', '/partials/file_github_edit/' + encodeFilePath(filePath), {
                        target: detailContainer,
                        swap: 'outerHTML'
                    });
                }
            }
        }
    );
}

/**
 * Cancel the file-level GitHub context edit form and reload the view.
 *
 * Reloads the file-level GitHub edit partial via HTMX to discard unsaved
 * changes and restore the original values (DS-1082).
 *
 * @param {string} filePath - The relative file path for the orchestration file
 */
function cancelFileGitHubEdit(filePath) {
    var form = document.getElementById('file-github-edit-form');
    if (form) {
        var detailContainer = form.closest('.orchestration-detail');
        if (detailContainer) {
            htmx.ajax('GET', '/partials/file_github_edit/' + encodeFilePath(filePath), {
                target: detailContainer,
                swap: 'outerHTML'
            });
        }
    }
}

/**
 * Submit the file-level trigger edit form via the PUT API endpoint.
 *
 * Collects form fields from the file-level trigger edit form, builds a JSON
 * payload matching FileTriggerEditRequest, and submits via fetchWithCsrfRetry
 * with automatic CSRF token handling. Shows toast notification on result and
 * reloads the edit form partial on success (DS-1082).
 *
 * @param {string} filePath - The relative file path for the orchestration file
 * @param {HTMLFormElement} formElement - The form element to collect data from
 */
function submitFileTriggerEdit(filePath, formElement) {
    if (!formElement) {
        console.error(UI_MESSAGES.DEBUG.FILE_TRIGGER_EDIT_FORM_NULL);
        return;
    }

    // Build the request body from form fields
    var body = {};
    var source = formElement.querySelector('input[name="file_trigger_source"]:checked');
    if (source) body.source = source.value;
    var project = formElement.querySelector('#file_trigger_project');
    if (project && project.value.trim()) body.project = project.value.trim();
    var projectNumber = formElement.querySelector('#file_trigger_project_number');
    if (projectNumber && projectNumber.value) body.project_number = parseInt(projectNumber.value, 10);
    var projectScope = formElement.querySelector('input[name="file_trigger_project_scope"]:checked');
    if (projectScope) body.project_scope = projectScope.value;
    var projectOwner = formElement.querySelector('#file_trigger_project_owner');
    if (projectOwner && projectOwner.value.trim()) body.project_owner = projectOwner.value.trim();

    fetchWithCsrfRetry(
        '/api/orchestrations/files/' + encodeFilePath(filePath) + '/trigger',
        { method: 'PUT', body: JSON.stringify(body) },
        {
            csrfRefreshFailed: UI_MESSAGES.TOAST.FILE_EDIT_CSRF_REFRESH_FAILED,
            csrfValidationFailed: UI_MESSAGES.TOAST.FILE_EDIT_CSRF_VALIDATION_FAILED,
            genericError: UI_MESSAGES.TOAST.FILE_TRIGGER_EDIT_FAILED,
            networkError: UI_MESSAGES.TOAST.FILE_EDIT_NETWORK_ERROR,
            debugFetchError: UI_MESSAGES.DEBUG.FILE_TRIGGER_EDIT_FETCH_ERROR,
            onSuccess: function() {
                showToast('success', UI_MESSAGES.TOAST.FILE_TRIGGER_EDIT_SUCCESS);
                // Reload the edit form to show updated values
                var detailContainer = formElement.closest('.orchestration-detail');
                if (detailContainer) {
                    htmx.ajax('GET', '/partials/file_trigger_edit/' + encodeFilePath(filePath), {
                        target: detailContainer,
                        swap: 'outerHTML'
                    });
                }
            }
        }
    );
}

/**
 * Cancel the file-level trigger edit form and reload the view.
 *
 * Reloads the file-level trigger edit partial via HTMX to discard unsaved
 * changes and restore the original values (DS-1082).
 *
 * @param {string} filePath - The relative file path for the orchestration file
 */
function cancelFileTriggerEdit(filePath) {
    var form = document.getElementById('file-trigger-edit-form');
    if (form) {
        var detailContainer = form.closest('.orchestration-detail');
        if (detailContainer) {
            htmx.ajax('GET', '/partials/file_trigger_edit/' + encodeFilePath(filePath), {
                target: detailContainer,
                swap: 'outerHTML'
            });
        }
    }
}
