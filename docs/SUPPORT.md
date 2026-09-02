# Support and version policy

WorldZero is an alpha research instrument. The latest released minor version receives bug and security fixes. Help is provided on a best-effort basis through repository issues and discussions; there is no service-level commitment.

## Versioned surfaces

- The Python package and CLI follow semantic versioning.
- The plugin API has its own `api_version`; incompatible SDK changes increment its major version.
- Every law has a `family_version`, implementation fingerprint, and calibration-suite fingerprint.
- Scoring profiles and persistence schemas are versioned independently.
- A behavior, channel, calibration, control, scoring, or source change creates a new run identity.

State-v2 and trace-v2/v3 compatibility is covered by frozen fixtures in the 0.3 series. New plugin-backed runs use state-v3 and trace-v4. A future removal of a compatibility adapter will be announced in the changelog at least one minor release in advance.

Official registry entries authenticate reviewed built-in implementations only. Community plugins remain experimental until their exact identities are reviewed and bundled in a release.
