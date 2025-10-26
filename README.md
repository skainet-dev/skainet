[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)

# SKaiNET

**SKaiNET** is an open-source deep learning framework written in Kotlin, designed with developers in mind to enable the creation modern AI powered applications with ease.

## Development Practices

This project follows established development practices for maintaining code quality and release management:

* **Branching Model**: We use [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) as our branching strategy for managing feature development, releases, and hotfixes.
* **Versioning**: We follow [Semantic Versioning (SemVer)](https://semver.org/) for all releases, ensuring predictable version numbering based on the nature of changes.

## Reflective Documentation (short overview)

SKaiNET includes a reflective documentation system that keeps docs in sync with the code. During the build, a KSP processor extracts operator metadata (signatures, parameters, backend availability, implementation status) into a JSON file. A small DocGen tool then converts this JSON into AsciiDoc fragments and pages.

- Source of truth (generated): skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json
- Generated docs output: docs/modules/operators/_generated_/
- Asciidoctor site output: build/docs/asciidoc/ (if you run an Asciidoctor task locally)

### Quick start: generate reflective docs

Use any of the following Gradle tasks from the project root:

1) Full pipeline (recommended)
   ./gradlew generateDocs
   - Runs KSP to produce operators.json (if needed)
   - Generates AsciiDoc files under docs/modules/operators/_generated_
   - Optionally, you can run an Asciidoctor task to build an HTML site locally (output under build/docs/asciidoc)

2) Operators documentation only
   ./gradlew generateOperatorDocs
   - Depends on KSP; runs the built-in generateDocs task and then Asciidoctor

Open the generated AsciiDoc sources in docs/modules/operators/_generated_ with your preferred AsciiDoc viewer. If you build an HTML site locally with Asciidoctor, open build/docs/asciidoc.

---

## Development Practices

This project follows established development practices for maintaining code quality and release management:

* Branching Model: We use GitFlow as our branching strategy for managing feature development, releases, and hotfixes.
* Versioning: We follow Semantic Versioning (SemVer) for all releases, ensuring predictable version numbering based on the nature of changes.
