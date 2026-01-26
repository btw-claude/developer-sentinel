/**
 * Unit tests for deprecation-comments.js stylelint plugin.
 *
 * Tests the regex patterns for CSS deprecation annotations.
 * These tests validate pattern matching against various edge cases
 * to aid future maintenance as annotation formats evolve.
 *
 * Note: Integration tests with stylelint v16+ require ESM configuration.
 * These pattern tests provide direct validation of the regex patterns.
 *
 * Related: DS-279 (test implementation), DS-275 (original plugin), DS-269 (deprecation format)
 */

describe("deprecation-comments plugin patterns", () => {
    /**
     * Pattern definitions matching those in the plugin.
     * These are duplicated here for isolated unit testing of regex behavior.
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

    describe("deprecatedWithTicket pattern", () => {
        it("should match standard ticket formats (PROJECT-NUMBER)", () => {
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated DS-123 - reason")).toBe(true);
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated ABC-1 - reason")).toBe(true);
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated PROJ-99999 - reason")).toBe(true);
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated JIRA-1234 - Some detailed reason")).toBe(true);
        });

        it("should match with optional space before hyphen", () => {
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated DS-123- reason")).toBe(true);
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated DS-123  - reason")).toBe(true);
        });

        it("should not match lowercase project keys", () => {
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated ds-123 - reason")).toBe(false);
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated Ds-123 - reason")).toBe(false);
        });

        it("should not match missing hyphen in ticket", () => {
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated DS123 - reason")).toBe(false);
        });

        it("should not match missing ticket entirely", () => {
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated reason")).toBe(false);
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated - reason")).toBe(false);
        });

        it("should not match numbers-only ticket format", () => {
            expect(PATTERNS.deprecatedWithTicket.test("@deprecated 123 - reason")).toBe(false);
        });
    });

    describe("deprecatedInline pattern", () => {
        it("should match inline deprecation with Use keyword", () => {
            expect(PATTERNS.deprecatedInline.test("@deprecated - Use .new-class instead")).toBe(true);
            expect(PATTERNS.deprecatedInline.test("@deprecated - Use .toggle-switch__slider instead")).toBe(true);
            expect(PATTERNS.deprecatedInline.test("@deprecated - Use .component__element--modifier instead")).toBe(true);
        });

        it("should require proper spacing around hyphen", () => {
            expect(PATTERNS.deprecatedInline.test("@deprecated Use .new-class instead")).toBe(false);
            expect(PATTERNS.deprecatedInline.test("@deprecated- Use .new-class instead")).toBe(false);
            expect(PATTERNS.deprecatedInline.test("@deprecated -Use .new-class instead")).toBe(false);
        });

        it("should not match without Use keyword", () => {
            expect(PATTERNS.deprecatedInline.test("@deprecated - Replace with .new-class")).toBe(false);
        });
    });

    describe("sinceDate pattern", () => {
        it("should match YYYY-MM-DD format", () => {
            expect(PATTERNS.sinceDate.test("@since 2026-01-26")).toBe(true);
            expect(PATTERNS.sinceDate.test("@since 2025-12-31 (deprecation added)")).toBe(true);
            expect(PATTERNS.sinceDate.test("@since 1999-01-01")).toBe(true);
            expect(PATTERNS.sinceDate.test("@since 2030-06-15")).toBe(true);
        });

        it("should not match single-digit month or day", () => {
            expect(PATTERNS.sinceDate.test("@since 2026-1-26")).toBe(false);
            expect(PATTERNS.sinceDate.test("@since 2026-01-6")).toBe(false);
        });

        it("should not match reversed date formats", () => {
            expect(PATTERNS.sinceDate.test("@since 26-01-2026")).toBe(false);
            expect(PATTERNS.sinceDate.test("@since 01-26-2026")).toBe(false);
        });

        it("should not match human-readable date formats", () => {
            expect(PATTERNS.sinceDate.test("@since January 26, 2026")).toBe(false);
            expect(PATTERNS.sinceDate.test("@since Jan 26 2026")).toBe(false);
        });

        it("should require space after @since", () => {
            expect(PATTERNS.sinceDate.test("@since2026-01-26")).toBe(false);
        });
    });

    describe("removalTimeline pattern", () => {
        it("should match quarter and year format", () => {
            expect(PATTERNS.removalTimeline.test("@removal Planned removal: Q2 2026")).toBe(true);
            expect(PATTERNS.removalTimeline.test("@removal Planned removal: Q1 2025 (v2.0)")).toBe(true);
            expect(PATTERNS.removalTimeline.test("@removal Q4 2027")).toBe(true);
            expect(PATTERNS.removalTimeline.test("@removal Q3 2025")).toBe(true);
        });

        it("should match year-only format", () => {
            expect(PATTERNS.removalTimeline.test("@removal scheduled for 2026")).toBe(true);
            expect(PATTERNS.removalTimeline.test("@removal in 2025")).toBe(true);
        });

        it("should be case-insensitive for quarter prefix", () => {
            expect(PATTERNS.removalTimeline.test("@removal q2 2026")).toBe(true);
        });

        it("should not match vague timelines without dates", () => {
            expect(PATTERNS.removalTimeline.test("@removal soon")).toBe(false);
            expect(PATTERNS.removalTimeline.test("@removal next release")).toBe(false);
            expect(PATTERNS.removalTimeline.test("@removal TBD")).toBe(false);
        });

        // Note: The current regex allows Q5-Q9 due to the pattern structure.
        // This is an acceptable limitation documented here for awareness.
        // Future enhancement: Use stricter pattern /Q[1-4]\s+\d{4}/ if needed.
    });

    describe("migrationGuide pattern", () => {
        it("should match Migration Guide label", () => {
            expect(PATTERNS.migrationGuide.test("Migration Guide:")).toBe(true);
            expect(PATTERNS.migrationGuide.test("MIGRATION GUIDE:")).toBe(true);
            expect(PATTERNS.migrationGuide.test("migration guide:")).toBe(true);
        });

        it("should match migration keywords in text", () => {
            expect(PATTERNS.migrationGuide.test("See migration instructions")).toBe(true);
            expect(PATTERNS.migrationGuide.test("How to migrate")).toBe(true);
            expect(PATTERNS.migrationGuide.test("Please MIGRATE to the new API")).toBe(true);
        });

        it("should not match unrelated content", () => {
            expect(PATTERNS.migrationGuide.test("deprecated styles")).toBe(false);
            expect(PATTERNS.migrationGuide.test("replacement")).toBe(false);
            expect(PATTERNS.migrationGuide.test("use new class")).toBe(false);
        });
    });

    describe("blockDeprecation pattern", () => {
        it("should match standard block comment with dashes", () => {
            expect(PATTERNS.blockDeprecation.test("/* -----------------------------------------------------------------------------")).toBe(true);
            expect(PATTERNS.blockDeprecation.test("/* ----------")).toBe(true);
            expect(PATTERNS.blockDeprecation.test("/* --------------------")).toBe(true);
        });

        it("should require minimum 10 dashes", () => {
            expect(PATTERNS.blockDeprecation.test("/* ---------")).toBe(false); // 9 dashes
            expect(PATTERNS.blockDeprecation.test("/* -----")).toBe(false); // 5 dashes
        });

        it("should not match regular comments", () => {
            expect(PATTERNS.blockDeprecation.test("/* Regular comment */")).toBe(false);
            expect(PATTERNS.blockDeprecation.test("/* Some text with - dashes - */")).toBe(false);
        });

        it("should require comment start at beginning", () => {
            expect(PATTERNS.blockDeprecation.test("  /* ----------")).toBe(false);
        });
    });

    describe("real-world examples from utilities.css", () => {
        const blockDeprecationExample = `/* -----------------------------------------------------------------------------
   LEGACY TOGGLE SLIDER STYLES
   @deprecated DS-260 - These styles are deprecated and will be removed in a future release.
   @since 2026-01-26 (deprecation added)
   @removal Planned removal: Q2 2026 (v2.0)

   Migration Guide:
   Replace legacy HTML with BEM-structured HTML.
   ----------------------------------------------------------------------------- */`;

        it("should match block deprecation header", () => {
            expect(PATTERNS.blockDeprecation.test("/*" + blockDeprecationExample.split("/*")[1])).toBe(true);
        });

        it("should find ticket reference in block", () => {
            expect(PATTERNS.deprecatedWithTicket.test(blockDeprecationExample)).toBe(true);
        });

        it("should find since date in block", () => {
            expect(PATTERNS.sinceDate.test(blockDeprecationExample)).toBe(true);
        });

        it("should find removal timeline in block", () => {
            expect(PATTERNS.removalTimeline.test(blockDeprecationExample)).toBe(true);
        });

        it("should find migration guide in block", () => {
            expect(PATTERNS.migrationGuide.test(blockDeprecationExample)).toBe(true);
        });

        const inlineDeprecationExample = "/* @deprecated - Use .toggle-switch__slider instead */";

        it("should match inline deprecation format", () => {
            expect(PATTERNS.deprecatedInline.test(inlineDeprecationExample)).toBe(true);
        });
    });
});
