/**
 * Stylelint plugin for validating CSS deprecation annotation format.
 *
 * This plugin enforces consistent deprecation comment formatting as defined
 * in the CSS Style Guide (docs/CSS_STYLE_GUIDE.md).
 *
 * Validates:
 * - @deprecated annotations include a ticket reference (e.g., DS-123)
 * - @since annotations use YYYY-MM-DD date format
 * - @removal annotations include quarter/year information
 *
 * Related: DS-275 (standardization task), DS-269 (original implementation)
 */

const stylelint = require("stylelint");

const ruleName = "developer-sentinel/deprecation-comments";
const messages = stylelint.utils.ruleMessages(ruleName, {
    missingTicketReference:
        '@deprecated annotation should include a ticket reference (e.g., "@deprecated DS-123 - reason")',
    invalidSinceDateFormat:
        '@since annotation should use YYYY-MM-DD format (e.g., "@since 2026-01-26 (deprecation added)")',
    missingRemovalTimeline:
        '@removal annotation should include a timeline (e.g., "@removal Planned removal: Q2 2026 (v2.0)")',
    deprecatedMissingMigration:
        'Block-level @deprecated comment should include migration guidance in the comment block',
});

const meta = {
    url: "https://github.com/btw-claude/developer-sentinel/blob/main/docs/CSS_STYLE_GUIDE.md#deprecation-annotations",
};

/**
 * Patterns for validating deprecation annotations
 */
const PATTERNS = {
    // Matches: @deprecated DS-123 - reason or @deprecated - Use .class instead
    deprecatedWithTicket: /@deprecated\s+[A-Z]+-\d+\s*-/,
    deprecatedInline: /@deprecated\s+-\s+Use\s+/,

    // Matches: @since YYYY-MM-DD
    sinceDate: /@since\s+\d{4}-\d{2}-\d{2}/,

    // Matches: @removal Planned removal: Q1-Q4 YYYY
    removalTimeline: /@removal\s+.*(?:Q[1-4]\s+\d{4}|\d{4})/i,

    // Check for migration guide in block comments
    migrationGuide: /Migration Guide:|migrate|migration/i,

    // Check if this is a block-level deprecation (multi-line with dashes)
    blockDeprecation: /^\/\*\s*-{10,}/,
};

/** @type {import('stylelint').Rule} */
const ruleFunction = (primaryOption, _secondaryOptions, context) => {
    return (root, result) => {
        const validOptions = stylelint.utils.validateOptions(result, ruleName, {
            actual: primaryOption,
            possible: [true, false],
        });

        if (!validOptions || !primaryOption) {
            return;
        }

        root.walkComments((comment) => {
            const text = comment.text;

            // Skip non-deprecation comments
            if (!text.includes("@deprecated") && !text.includes("@since") && !text.includes("@removal")) {
                return;
            }

            const isBlockDeprecation = PATTERNS.blockDeprecation.test("/*" + text);

            // Validate @deprecated format
            if (text.includes("@deprecated")) {
                const hasTicketRef = PATTERNS.deprecatedWithTicket.test(text);
                const isInlineDeprecation = PATTERNS.deprecatedInline.test(text);

                if (!hasTicketRef && !isInlineDeprecation) {
                    stylelint.utils.report({
                        message: messages.missingTicketReference,
                        node: comment,
                        result,
                        ruleName,
                    });
                }

                // Block deprecations should include migration guidance
                if (isBlockDeprecation && !PATTERNS.migrationGuide.test(text)) {
                    stylelint.utils.report({
                        message: messages.deprecatedMissingMigration,
                        node: comment,
                        result,
                        ruleName,
                    });
                }
            }

            // Validate @since format (only for block deprecations)
            if (text.includes("@since") && isBlockDeprecation) {
                if (!PATTERNS.sinceDate.test(text)) {
                    stylelint.utils.report({
                        message: messages.invalidSinceDateFormat,
                        node: comment,
                        result,
                        ruleName,
                    });
                }
            }

            // Validate @removal format (only for block deprecations)
            if (text.includes("@removal") && isBlockDeprecation) {
                if (!PATTERNS.removalTimeline.test(text)) {
                    stylelint.utils.report({
                        message: messages.missingRemovalTimeline,
                        node: comment,
                        result,
                        ruleName,
                    });
                }
            }
        });
    };
};

ruleFunction.ruleName = ruleName;
ruleFunction.messages = messages;
ruleFunction.meta = meta;

module.exports = stylelint.createPlugin(ruleName, ruleFunction);
