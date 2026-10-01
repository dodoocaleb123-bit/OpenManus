# OpenManus Design Playbook

A normalized, local-first reference distilled from the user-supplied `Downloads.zip` design material (Frontend Design, UI/UX Pro Max, Emil Design Engineering, Impeccable, Taste, and Sleek mobile/design references). Use it only when the DeepSeek control plan assigns a design/creative handoff. This is guidance, not a replacement for the user's explicit requirements or platform safety rules.

## Start with intent, not a template

Identify the audience, task, context, and primary success action. Choose a coherent design mode: **Persuade** for conversion, **Operate** for frequent work, **Read** for long-form content, or **Experience** for an expressive destination. Use one clear visual direction and explain the design rationale. Avoid generic dashboard/card grids, gratuitous gradients, decorative blur, excessive rounding, and interchangeable SaaS layouts. Prefer a small number of deliberate, product-specific choices.

## Define a usable system

Specify hierarchy, layout, spacing, typography, color tokens, density, and component behavior. Make the main action obvious; distinguish primary, secondary, destructive, and disabled actions. Use consistent tokens rather than one-off values. Keep content concrete and scannable, write useful labels, and make empty, loading, success, error, disabled, and confirmation states explicit. Preserve the user's requested functionality and information; do not invent features to make a screen look fuller.

## Responsive and accessible by default

Design for narrow viewports first when the product is mobile-facing, then extend deliberately to tablet and desktop. Avoid fixed-width layouts and hover-only interactions. Ensure readable type, adequate contrast, visible keyboard focus, semantic controls, keyboard operation, accessible names, and touch targets. Prefer native interaction conventions over surprising gestures. Include reduced-motion behavior and do not use color alone to communicate status.

## Motion and implementation performance

Use animation only when it explains a state change, spatial relationship, or hierarchy. Keep transitions short and purposeful; prefer transform and opacity over properties that trigger layout. Respect `prefers-reduced-motion`, do not block interaction, and avoid continuous decorative motion. Keep responsive layouts and visual effects inexpensive enough for ordinary laptops and mobile devices.

## Handoff and verification

Return a practical structured brief: visual direction; audience and hierarchy; palette with contrast intent; typography; layout and responsive rules; component/state behavior; accessibility; motion; and a prioritized QA checklist. For a build request, give the implementation model concrete guidance but do not claim that code or previews exist. For a design-only request, deliver the design artifact/brief without writing project code.

When screenshots or design images are involved, analyze only attachments explicitly included in the current request or screenshots captured from the current project preview. Distinguish observed pixels from assumptions. After implementation, verify actual routes, controls, and states at desktop and mobile sizes; inspect screenshots and report unresolved issues. Never claim visual verification without a browser/screenshot result.

## Tool availability

Design reasoning is local through the configured Llama role. Sleek and Taste are optional external services in the supplied references; neither is a required dependency or part of the zero-cost default. If such a service is unavailable, continue with the local design handoff and the platform's existing browser preview. Never send user project data to an external design service without explicit user authorization.
