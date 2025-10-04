# How to write “Explanation” (discussion) docs

Below is a procedural guideline / checklist that an AI (or human writer) should follow when asked to write an “explanation” style document in a larger documentation system.

1. Decide the topic boundary / scope

Choose a topic that merits a deeper dive — something that benefits from understanding rationale, trade-offs, alternatives, history, or context.

Avoid topics that are already covered well in “how-to” or “reference” docs, unless you are elaborating the “why behind them.”

Define the intended audience: users who already know the basics and want more insight.

Limit the scope: don’t try to explain everything, just pick one coherent theme or area (e.g. “caching strategies for the platform,” or “why environment variables are handled this way”).

2. Provide context and motivation

Start with why this topic matters: what problem, challenge or question it addresses.

Give background: prior states, constraints, historical evolution — how did we arrive here?

Define key concepts or terminology as needed (briefly, without turning into a reference).

3. Explore alternatives (and trade-offs)

Describe possible different approaches or designs, even those not chosen.

Compare and contrast: pros, cons, when one might be more suitable than another.

If opinions or community debates exist, mention them, present reasoning for different sides.

4. Explain rationale and design decisions

For each major choice, explain why it was made (constraints, priorities, trade-offs).

If there are technical constraints (performance, security, backwards compatibility, user expectations), surface them.

Where appropriate, show consequences (positive and negative) of choices.

5. Include illustrative examples or metaphors

Use analogies, metaphors, or stories to help understanding (as Divio uses “cooking” as an analogy)
docs.divio.com

Use diagrams or conceptual visuals if helpful (to illustrate relationships, flows, architecture).

6. Maintain a discursive style (not instructional)

Do not give explicit “step 1, do this; step 2, do that” instructions.

Do not list API syntax, method signatures, or detailed parameters (those belong in reference).

Use more narrative style: “we think about …”, “alternatively …”, “one could consider …”.

7. Structure the document logically

Use headings/subheadings (e.g. “Motivation / Background”, “Alternatives”, “Rationale”, “Examples / Illustrations”, “Implications”).

Progress from general → specific: first big picture, then deeper dives.

You might end with “When this is useful / implications / further reading”.

8. Link to related docs

Provide references (links) to corresponding how-to guides, reference pages, or deeper technical specs.

Encourage the reader to “see also” for performing actions, APIs, or tutorials.

9. Be neutral, clear, and well-reasoned

If you present multiple opinions, try to balance them fairly; if you prefer one, explain why.

Avoid jargon without explanation.

Keep paragraphs reasonably sized; use examples to break up abstract discussion.

10. Review for scope creep and redundancy

Check that you didn’t slip back into instructional or reference mode — if there is instruction, consider moving it to a how-to.

Ensure you didn’t duplicate content from tutorials or reference; if overlap, prefer referencing instead of rehashing.