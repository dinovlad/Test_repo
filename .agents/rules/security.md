# Security

- Connect to environments only through internal team libraries and env files for sensitive values; env files are excluded from version control.
- Never include in agent prompts or responses: project code in full, trade secrets including data schemas and attributes, or personal data.
- Work with external AI models on corporate laptops only through [CoPilot](https://copilot.x5.ru), without transmitting sensitive information; internal models may be used without restrictions.

# Documentation and references

- Scientific references in code follow the bibliographic rules defined in the project-level `README.md`, using Harvard style; document formatting uses English, title case for headings, lower case for document names, and a line break at the end of each semantic block.
- Complex logic implemented in code is additionally described in `docs/`.
