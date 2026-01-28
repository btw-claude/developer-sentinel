# Event Delegation Architecture

This document describes the event delegation pattern used in the Developer Sentinel dashboard, specifically the decision to use the `<main>` element as the delegation container.

## Table of Contents

- [Overview](#overview)
- [Why Event Delegation](#why-event-delegation)
- [The `<main>` Container Decision](#the-main-container-decision)
- [When This Assumption Might Change](#when-this-assumption-might-change)
- [Impact on Dynamically Loaded Content](#impact-on-dynamically-loaded-content)
- [Implementation Details](#implementation-details)
- [Related Resources](#related-resources)

## Overview

As of DS-376, the dashboard uses event delegation for handling collapsible section interactions. Instead of attaching individual event listeners to each collapsible header element, a single event listener is attached to the `<main>` container that handles all collapsible section clicks.

```javascript
// Event delegation pattern (DS-376)
const mainContainer = document.querySelector('main');
mainContainer.addEventListener('click', function(event) {
    const header = event.target.closest('[data-collapsible-target]');
    if (!header) return;
    // Handle the click...
});
```

## Why Event Delegation

Event delegation was chosen over individual listeners for the following reasons:

### Performance Benefits

- **Reduced Memory Usage**: A single event listener uses less memory than multiple listeners attached to individual elements
- **Faster Page Load**: Fewer listeners to set up during initialization
- **Better Scalability**: Performance remains constant regardless of the number of collapsible sections

### Dynamic Content Support

- **Automatic Handling**: New collapsible sections added via HTMX or JavaScript work immediately without re-initialization
- **No Memory Leaks**: No need to track and remove listeners when elements are removed from the DOM
- **Simplified Code**: No need to call initialization functions after dynamic content loads

### Debugging Benefits

- **Single Point of Attachment**: All collapsible click handling flows through one location
- **Easier Inspection**: One listener to examine in browser developer tools
- **Centralized Logging**: Console warnings and debugging can be added in one place

## The `<main>` Container Decision

The `<main>` element was chosen as the delegation container for the following reasons:

### Semantic Appropriateness

- The `<main>` element contains all primary content that users interact with
- Header and navigation elements (in `<header>`) don't contain collapsible sections
- The semantic boundary aligns with the functional boundary of collapsible content

### Page Structure Alignment

The dashboard layout is structured as:

```html
<body>
    <header>
        <!-- Navigation - no collapsible sections here -->
    </header>
    <main>
        <!-- All dashboard content with collapsible sections -->
        {% block content %}{% endblock %}
    </main>
</body>
```

All collapsible sections exist within the `<main>` element, making it the natural delegation container.

### Event Propagation Efficiency

- Attaching to `<main>` rather than `document.body` reduces the event propagation path
- Events don't need to bubble through `<header>` elements
- Slightly more efficient than using `document` as the delegation target

## When This Assumption Might Change

The `<main>` container assumption may need to be revisited if:

### New Layout Patterns

1. **Modal Dialogs**: If modal dialogs with collapsible sections are rendered outside `<main>` (e.g., as direct children of `<body>`), they would need separate event delegation or the container would need to change to `document.body`

2. **Sidebar Navigation**: If a sidebar with collapsible menu items is added outside `<main>`, the delegation strategy would need adjustment

3. **Footer Collapsibles**: If the footer (outside `<main>`) needs collapsible FAQ sections or similar

### Recommended Approach for Changes

If collapsible sections are needed outside `<main>`:

```javascript
// Option 1: Expand delegation to document.body
const container = document.body;

// Option 2: Multiple delegation points (more complex but scoped)
const mainContainer = document.querySelector('main');
const modalContainer = document.querySelector('#modal-root');
[mainContainer, modalContainer].forEach(container => {
    if (container) {
        container.addEventListener('click', handleCollapsibleClick);
    }
});
```

## Impact on Dynamically Loaded Content

### Content Within `<main>` (Works Automatically)

- HTMX-loaded content injected into `<main>` or its descendants works immediately
- No re-initialization required after HTMX swaps
- Template partials can include `[data-collapsible-target]` elements without special handling

### Content Outside `<main>` (Requires Consideration)

Dynamic content rendered outside `<main>` would not be handled by the current delegation setup. This includes:

- Modals rendered as siblings of `<main>`
- Toast notifications (though these typically don't need collapsible sections)
- Any content injected directly into `<body>`

### HTMX Integration

The event delegation pattern complements HTMX's content replacement model:

```html
<!-- Template partial loaded via HTMX -->
<div class="card collapsible-section">
    <div class="collapsible-header"
         data-collapsible-target="new-content">
        <h2>Dynamically Loaded Section</h2>
    </div>
    <div id="new-content" class="collapsible-content">
        <!-- Content -->
    </div>
</div>
```

This will work immediately when injected into any location within `<main>`.

## Implementation Details

### Key Files

- **`src/sentinel/dashboard/templates/base.html`**: Contains the `initCollapsibleSections()` function with event delegation implementation
- **Template partials**: Use `data-collapsible-target` attribute on clickable headers

### Initialization

Event delegation is set up on DOM ready:

```javascript
document.addEventListener('DOMContentLoaded', initCollapsibleSections);
```

### Error Handling

The implementation includes defensive checks:

```javascript
const mainContainer = document.querySelector('main');
if (!mainContainer) {
    console.warn('initCollapsibleSections: Could not find main container for event delegation');
    return;
}
```

This ensures graceful degradation if the page structure changes unexpectedly.

## Related Resources

- [CSS Style Guide](./CSS_STYLE_GUIDE.md)
- DS-370: Initial refactoring from inline onclick to data attributes
- DS-376: Event delegation implementation
- DS-377: This documentation

---

*Last updated: DS-377*
