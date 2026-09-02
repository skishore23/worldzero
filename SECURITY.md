# Security policy

## Supported versions

Security fixes are provided for the latest released minor version. Older research artifacts remain reproducible records but are not maintained software releases.

## Reporting a vulnerability

Do not open a public issue for a vulnerability, credential, or privacy exposure. Use GitHub's private vulnerability reporting feature for this repository. If that feature is unavailable, contact the repository owner privately through their GitHub profile and include “WorldZero security” in the subject.

Include the affected version, a minimal reproduction, impact, and any suggested mitigation. Do not include real credentials or unnecessary personal data. You should receive acknowledgement within seven days and a status update within fourteen days.

## Trust boundary

Community law plugins are trusted in-process Python. Installing one grants it the same filesystem, environment, and network authority as the Python process. WorldZero v1 validates scientific and replay contracts; it does not sandbox third-party code. Review a plugin before installing it and use an isolated environment.

Model endpoints and API keys are optional. Keep keys in a private environment variable, never in manifests, traces, bug reports, examples, or source control. Remote endpoints require explicit opt-in and every model run must use a hard request budget.
