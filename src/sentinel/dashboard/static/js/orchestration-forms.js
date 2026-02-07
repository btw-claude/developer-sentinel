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
 *
 * CSRF Token Auto-Refresh (DS-737):
 * When a page is refreshed, embedded CSRF tokens become invalid (single-use).
 * The refreshCsrfToken() function fetches a fresh token from the API and
 * updates the hidden input, preventing 403 errors on form submission after
 * page refresh.
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
 * and submits via fetch. Shows toast notification on result and reloads
 * the detail view on success.
 *
 * @param {HTMLFormElement} formElement - The form element to collect data from
 * @param {string} name - The orchestration name to update
 */
function submitOrchestrationEdit(formElement, name) {
    if (!formElement) {
        console.error('submitOrchestrationEdit: formElement is null');
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

    // Submit via fetch for proper response handling
    fetch('/api/orchestrations/' + encodeURIComponent(name), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).then(function(response) {
        return response.json().then(function(data) {
            return { status: response.status, data: data };
        });
    }).then(function(result) {
        if (result.status >= 200 && result.status < 300 && result.data.success) {
            showToast('success', 'Orchestration \'' + name + '\' updated successfully');
            // Reload the detail view to show updated values
            var detailContainer = formElement.closest('.orchestration-detail');
            if (detailContainer) {
                htmx.ajax('GET', '/partials/orchestration_detail/' + encodeURIComponent(name), {
                    target: detailContainer,
                    swap: 'outerHTML'
                });
            }
        } else if (result.status === 422) {
            showToast('error', result.data.detail || 'Validation error');
        } else if (result.status === 429) {
            showToast('warning', 'Rate limit exceeded. Please wait before trying again.');
        } else {
            showToast('error', result.data.detail || 'Failed to update orchestration');
        }
    }).catch(function(error) {
        console.error('submitOrchestrationEdit: fetch error:', error);
        showToast('error', 'Network error while updating orchestration');
    });
}

/**
 * Submit an orchestration create form via the POST API endpoint.
 *
 * Collects form fields, builds nested JSON matching OrchestrationCreateRequest,
 * and submits via fetch. Shows toast notification on result and hides the
 * creation form on success.
 *
 * @param {HTMLFormElement} formElement - The form element to collect data from
 */
function submitOrchestrationCreate(formElement) {
    if (!formElement) {
        console.error('submitOrchestrationCreate: formElement is null');
        return;
    }

    // Get orchestration name (required)
    var nameInput = formElement.querySelector('#create_name');
    if (!nameInput || !nameInput.value.trim()) {
        showToast('error', 'Orchestration name is required');
        return;
    }
    var name = nameInput.value.trim();

    // Get target file (required)
    var targetFileSelector = formElement.querySelector('#target_file_selector');
    var newFileNameInput = formElement.querySelector('#new_file_name');
    var targetFile = '';

    if (!targetFileSelector || !targetFileSelector.value) {
        showToast('error', 'Target file must be selected');
        return;
    }

    if (targetFileSelector.value === '__new__') {
        if (!newFileNameInput || !newFileNameInput.value.trim()) {
            showToast('error', 'New file name is required');
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

    // Read CSRF token from hidden input
    var csrfToken = document.getElementById('csrf_token');

    /**
     * Internal helper to perform the create POST request.
     *
     * @param {string} token - CSRF token to include in the request header
     * @param {boolean} isRetry - Whether this is a retry after token refresh
     */
    function doCreate(token, isRetry) {
        fetch('/api/orchestrations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': token
            },
            body: JSON.stringify(body)
        }).then(function(response) {
            return response.json().then(function(data) {
                return { status: response.status, data: data };
            });
        }).then(function(result) {
            if (result.status >= 200 && result.status < 300 && result.data.success) {
                showToast('success', 'Orchestration \'' + name + '\' created successfully');
                // Hide the creation form
                document.getElementById('create-form-container').style.display = 'none';
                // Reload the orchestrations list (HTMX will auto-refresh on next poll)
            } else if (result.status === 403 && !isRetry) {
                // CSRF token invalid (e.g., page refresh) - auto-refresh and retry (DS-737)
                return refreshCsrfToken().then(function(newToken) {
                    if (newToken) {
                        doCreate(newToken, true);
                    } else {
                        showToast('error', 'CSRF token refresh failed. Please reload the page.');
                    }
                });
            } else if (result.status === 403) {
                showToast('error', 'CSRF token validation failed. Please reload the page.');
            } else if (result.status === 422) {
                showToast('error', result.data.detail || 'Validation error');
            } else if (result.status === 429) {
                showToast('warning', 'Rate limit exceeded. Please wait before trying again.');
            } else {
                showToast('error', result.data.detail || 'Failed to create orchestration');
            }
        }).catch(function(error) {
            console.error('submitOrchestrationCreate: fetch error:', error);
            showToast('error', 'Network error while creating orchestration');
        });
    }

    doCreate(csrfToken ? csrfToken.value : '', false);
}
