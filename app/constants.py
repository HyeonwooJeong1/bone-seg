"""Application-wide constants."""

# Application (development) version. Bump this per milestone and tag it in git
# (e.g. `git tag -a v1.1.0 -m "..."`). Shown in the window title and stored in
# saved sessions for traceability.
APP_VERSION = "1.0.0"

BASE_DATA_DIR = "11423945"

SESSION_FORMAT = "stanford_medicine_session"
# v2: Stage C switched from connected-component size filter to
#     scipy.ndimage.binary_opening. Old v1 sessions still load,
#     with default opening parameters substituted in.
# v3: Multi-series fusion. Adds 'fusion' block (enabled flag +
#     base_series_index). Older sessions default to fusion ON
#     with base = series 0.
# v4: 'fusion' block extended with `mako_only` + `include_flags`
#     so per-series visibility round-trips. Older sessions get
#     Mako auto-detect at load time.
# v5: Bone separation state + undo history. Meshes saved as .vtk
#     in a sibling directory (<session>_meshes/). Older sessions
#     load fine — separation block is simply absent.
SESSION_VERSION = 5

MAKO_KEYWORDS = ("mako", "stryker")