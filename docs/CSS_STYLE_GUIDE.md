# CSS Style Guide

This document defines the CSS coding standards and conventions for the Developer Sentinel project.

## Table of Contents

- [Deprecation Annotations](#deprecation-annotations)
- [Comment Blocks](#comment-blocks)
- [BEM Naming Convention](#bem-naming-convention)
- [CSS Custom Properties](#css-custom-properties)

## Deprecation Annotations

When deprecating CSS styles, use the following standardized annotation format to ensure consistency across the codebase.

### Block-Level Deprecation

For deprecating an entire section or group of related styles, use a multi-line comment block:

```css
/* -----------------------------------------------------------------------------
   SECTION TITLE
   @deprecated <Reason for deprecation>
   @since <YYYY-MM-DD> (deprecation added)
   @removal Planned removal: <Quarter Year> (<version>)

   Migration Guide:
   <Describe how to migrate from deprecated styles to new styles>
   ----------------------------------------------------------------------------- */
```

#### Example

```css
/* -----------------------------------------------------------------------------
   LEGACY TOGGLE SLIDER STYLES
   @deprecated - These styles are deprecated and will be removed in a future release.
   @since 2026-01-26 (deprecation added)
   @removal Planned removal: Q2 2026 (v2.0)

   Migration Guide:
   Replace legacy HTML:
     <label class="toggle-switch">
       <input type="checkbox">
       <span class="toggle-slider"></span>
     </label>

   With BEM-structured HTML:
     <label class="toggle-switch">
       <input type="checkbox" class="toggle-switch__input">
       <span class="toggle-switch__slider"></span>
     </label>
   ----------------------------------------------------------------------------- */
```

### Inline Deprecation

For deprecating individual rules, use single-line comments:

```css
/* @deprecated - Use <replacement selector> instead */
```

#### Example

```css
/* @deprecated - Use .toggle-switch__slider instead */
.toggle-switch .toggle-slider {
    /* ... styles ... */
}
```

### Annotation Reference

| Annotation | Required | Format | Description |
|------------|----------|--------|-------------|
| `@deprecated` | Yes | `@deprecated <reason>` | Marks the style as deprecated with the reason |
| `@since` | Yes (for blocks) | `@since <YYYY-MM-DD> (deprecation added)` | Date when the deprecation was added |
| `@removal` | Yes (for blocks) | `@removal Planned removal: <Quarter Year> (<version>)` | Planned removal timeline |

### Best Practices

1. **Provide clear reasoning**: Explain why the style is deprecated
2. **Provide migration guidance**: Include clear instructions for migrating to new styles
3. **Set a removal timeline**: Specify when the deprecated styles will be removed
4. **Group related deprecations**: Place deprecated styles together in clearly marked sections

## Comment Blocks

### Section Headers

Use double-line borders for major sections:

```css
/* =============================================================================
   SECTION TITLE
   Description of the section.
   ============================================================================= */
```

### Subsection Headers

Use single-line borders for subsections:

```css
/* -----------------------------------------------------------------------------
   Subsection Title
   Description of the subsection.
   ----------------------------------------------------------------------------- */
```

## BEM Naming Convention

This project follows the BEM (Block, Element, Modifier) naming convention:

- **Block**: Standalone entity (e.g., `.toggle-switch`)
- **Element**: Part of a block, denoted by double underscore (e.g., `.toggle-switch__slider`)
- **Modifier**: Variation of block/element, denoted by double hyphen (e.g., `.toggle-switch--disabled`)

### Example

```css
/* Block */
.toggle-switch {
    position: relative;
}

/* Element */
.toggle-switch__input {
    opacity: 0;
}

.toggle-switch__slider {
    background-color: var(--text-secondary);
}

/* Modifier */
.toggle-switch--large {
    height: 32px;
}
```

## CSS Custom Properties

Use CSS custom properties (CSS variables) for theming:

```css
.component {
    background-color: var(--bg-primary, #1a1a2e);
    color: var(--text-primary, #ffffff);
}
```

### Naming Convention

- Use kebab-case: `--variable-name`
- Group by purpose: `--bg-*`, `--text-*`, `--border-*`, etc.
- Always provide fallback values

## Live Examples

The following CSS files in this repository demonstrate the patterns described in this guide:

### Deprecation Annotations
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for "LEGACY TOGGLE SLIDER STYLES" to find the block-level deprecation example
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for `@deprecated - Use .toggle-switch__` to find inline deprecation examples

### BEM Naming Convention
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for "TOGGLE SWITCH COMPONENT" to find BEM structure examples (`.toggle-switch`, `.toggle-switch__input`, `.toggle-switch__slider`)
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for "NESTED ACCORDION COMPONENT" to find accordion BEM classes (`.accordion`, `.accordion__item`, `.accordion__header`, `.accordion__content`)

### CSS Custom Properties
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - See the file header comment block for CSS variable definitions and usage documentation
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for "TEXT UTILITIES" or "BACKGROUND UTILITIES" sections for CSS variable usage with fallback values

### Comment Blocks
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for "TEXT UTILITIES" to find section header examples with double-line borders (`===`)
- [`static/css/utilities.css`](../src/sentinel/dashboard/static/css/utilities.css) - Search for "ACCORDION BLOCK" to find subsection header examples with single-line borders (`---`)

## Related Resources

- [Stylelint Configuration](../.stylelintrc.json)
