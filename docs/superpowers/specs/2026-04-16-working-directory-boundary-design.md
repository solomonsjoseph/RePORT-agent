# Working Directory Boundary Design

## Objective

Require the user to choose a working directory before the app initializes the LLM, and constrain all LLM-caused filesystem mutations to that directory.

This boundary covers both:

- Files written by generated Python code during execution
- Files written by the app itself to support execution, such as execution inputs, outputs, and run artifacts

The Python runtime environment remains the existing RePORT-agent environment for now. The design leaves room for future per-working-directory virtual environments without changing the filesystem boundary model.

## Current State

- `streamlit_app.py` initializes model configuration and eventually calls `load_llm(...)` without requiring a working directory first.
- `tools/execution.py` is the main execution boundary. It already performs AST-based policy checks and dispatches execution to either trusted-local or Docker-backed execution.
- Docker-backed execution currently creates temporary input and output directories with `tempfile.TemporaryDirectory(...)`, which places app-managed artifacts outside any user-controlled workspace.
- Trusted-local policy checks can block many filesystem mutations, but when policy-blocked filesystem operations are allowed in trusted-local mode there is no designated-directory boundary.

## Requirements

### Functional

1. The app must require a working directory before `load_llm(...)` is called.
2. The chosen working directory must be normalized to an absolute path.
3. All app-managed execution artifacts must be created under the chosen working directory.
4. All generated-code filesystem mutations must be rejected unless their resolved target path is inside the chosen working directory.
5. Relative paths used by generated code must resolve against the chosen working directory.
6. The app must continue using the current RePORT-agent Python environment for now.

### Non-Functional

1. The filesystem boundary must be enforced centrally, not only in the UI.
2. The design should minimize changes to model-loading and graph-building behavior beyond the required setup gate.
3. The design should preserve a clean extension point for later per-working-directory virtual environments.

## User Flow

### Setup Gate

Before any model initialization, the app presents a required working-directory setup step.

- If no working directory has been chosen, the app blocks progression and does not call `load_llm(...)`.
- The user provides a path through the UI.
- The app normalizes the path to an absolute canonical path.
- If the directory does not exist, the UI can offer explicit creation.
- If the path is invalid or cannot be created, the app surfaces a validation error and stays blocked.

Once configured, the active working directory is shown in the UI and reused for the session.

### Execution Behavior

All LLM-caused writes use the selected working directory as their root. Relative paths are interpreted from that root instead of from the repository root.

## Design

### Execution Profile

Represent execution configuration as a small profile rather than a bare path:

- `working_directory: str`
- `environment_mode: "project_default"`

This keeps the first implementation simple while reserving a stable future shape for per-working-directory environments.

### App-Level Workspace Layout

All app-managed execution artifacts are created inside the selected working directory under a predictable structure, for example:

- `<wd>/runs/<run_id>/input`
- `<wd>/runs/<run_id>/output`

This replaces the current use of arbitrary OS temporary directories for execution artifacts.

### Docker Execution

For Docker-backed execution:

- The app creates per-run `input` and `output` directories inside the working directory.
- The app writes execution helper files such as `code.py` and `dataset.csv` into the run input directory.
- Docker mounts those directories into `/sandbox/input` and `/sandbox/output`.
- The sandbox runner continues writing `result.json`, `stdout.txt`, and optional figures into the mounted output directory.

This keeps all app-managed artifacts under the chosen directory while preserving the existing container model.

### Trusted-Local Execution

For trusted-local execution:

- The app keeps using the current RePORT-agent interpreter and installed packages.
- Generated code runs with the selected working directory as its execution working directory.
- Filesystem mutations are allowed only when the resolved destination path remains inside the chosen directory.

The boundary must apply even when the trusted-local policy toggle allows operations that would otherwise be blocked by the current preflight policy.

### Centralized Path Enforcement

`tools/execution.py` becomes the central enforcement point for the designated-directory rule.

Responsibilities:

- Validate that a working directory is present before execution starts
- Normalize and resolve the configured working directory
- Create per-run app-managed directories inside the working directory
- Resolve candidate filesystem mutation targets and reject paths outside the root
- Return structured policy-blocked errors when code attempts to mutate paths outside the boundary

The Streamlit UI still performs early validation, but the execution layer is the authoritative guardrail.

## Environment Model

The selected working directory is a filesystem boundary only. It does not change the active Python environment in this release.

- `trusted_local` continues to use the RePORT-agent interpreter
- `docker` continues to use the existing sandbox image
- No package installation path or virtual environment selection changes in this release

## Future Extension: Per-Working-Directory Environments

The execution profile is intentionally shaped so a later release can support:

- `environment_mode: "project_default" | "working_directory_venv"`
- Optional `venv_path`

That future change should switch interpreter selection without changing the designated-directory enforcement model.

## Error Handling

The app should surface clear errors for:

- Missing working directory during setup
- Invalid or non-resolvable working directory path
- Working directory creation failure
- Generated-code write attempts outside the working directory
- App-managed execution setup failures inside the working directory

Filesystem-boundary violations should be reported as structured policy-blocked errors so they integrate cleanly with existing review and error-handling flows.

## Testing

### UI

- App blocks before `load_llm(...)` if no working directory is configured
- Invalid directory input keeps the app blocked
- Valid directory input unblocks initialization

### Execution

- Docker execution creates run artifacts under the configured working directory
- Trusted-local execution resolves relative paths from the configured working directory
- Writes inside the configured directory are allowed when local filesystem mutation is enabled
- Writes outside the configured directory are rejected with a policy-blocked error

### Regression

- Existing package availability remains unchanged
- Existing non-filesystem execution behavior remains unchanged
- Existing review and retry flows still receive structured execution errors

## Out of Scope

- Per-working-directory virtual environment selection
- Automatic package installation into user-selected directories
- Broad redesign of model configuration or graph orchestration unrelated to the working-directory gate
