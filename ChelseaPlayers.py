# chelseaplayers.py

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st
import unicodedata  # for accent handling in filenames
import random



# ==== Streamlit setup ====
st.set_page_config(page_title="Chelsea Players – Toggle Scatter", layout="wide")

# ==== Fonts ====
golos_font = fm.FontProperties(fname="HelveticaforTarget-Bold.ttf")
helve_font = fm.FontProperties(fname="HelveticaforTarget.ttf")
hoves_font = fm.FontProperties(fname="TT Hoves Pro Trial Regular.ttf")
hovesbold_font = fm.FontProperties(fname="TT Hoves Pro Trial DemiBold.ttf")
hoveslight_font = fm.FontProperties(fname="TT Hoves Pro Trial Light.ttf")
hovesmedium_font = fm.FontProperties(fname="TT Hoves Pro Trial Medium.ttf")

st.title("Chelsea Chart Maker")

# --- Load data from local CSV (with 2 header rows: categories + stat names) ---
DATA_PATH = Path(__file__).parent / "ChelsPls.csv"
if not DATA_PATH.exists():
    st.error(f"CSV not found at: {DATA_PATH}. Place ChelsPls.csv next to this script or update DATA_PATH.")
    st.stop()

try:
    # Read raw, no header so we can use first two rows manually
    raw = pd.read_csv(DATA_PATH, header=None)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Expect at least 3 rows: categories, headers, and data
if raw.shape[0] < 3:
    st.error("CSV must have at least 2 header rows (categories + stat names) and 1 data row.")
    st.stop()

# Row 0 = categories, Row 1 = stat names, Rows 2+ = data
cat_row = raw.iloc[0]
name_row = raw.iloc[1]
data = raw.iloc[2:].reset_index(drop=True)

# Set proper column names from the second row
data.columns = name_row
data.columns = [str(c).strip() for c in data.columns]

# Build STAT_CATEGORIES dict from the first row
STAT_CATEGORIES = {}
for col_name, cat in zip(data.columns, cat_row):
    if pd.isna(col_name):
        continue
    STAT_CATEGORIES[str(col_name)] = str(cat) if not pd.isna(cat) else "Uncategorized"

df = data.copy()

# Try to convert everything numeric where possible (leave Pos as text)
for c in df.columns:
    if c == "Pos":
        continue
    df[c] = pd.to_numeric(df[c], errors="ignore")

# --- Initial numeric stat list BEFORE any filters (for categories, percentiles, etc.) ---
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
# Minutes is special; exclude it from stat lists
stat_candidates_all = [c for c in numeric_cols_all if c != "Minutes"]


def get_category(stat_name: str) -> str:
    return STAT_CATEGORIES.get(stat_name, "Uncategorized")


# ==== Sidebar controls ====
with st.sidebar:
    st.header("Display Options")
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    show_labels = st.checkbox("Show player labels", value=True)
    min_minutes = st.slider("Minimum minutes", 0, 1000, 0, 10)
    use_player_images = st.checkbox(
        "Use player images on chart (instead of dots)",
        value=False,
        help="Replaces scatter dots with each player's headshot from PlayerImages/"
    )

    st.markdown("---")
    # Position filter (checkboxes, all checked by default)
    st.subheader("Positions")
    if "Pos" in df.columns:
        pos_raw = (
            df["Pos"].astype(str)
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        primary_pos_all = pos_raw.str.split(r"[/, ]").str[0]
        available_pos = sorted(primary_pos_all.dropna().unique().tolist())
    else:
        available_pos = []

    pos_checks = {}
    for p in (available_pos or ["FW", "MF", "DF"]):
        pos_checks[p] = st.checkbox(p, value=True)
    selected_positions = [p for p, v in pos_checks.items() if v]

    st.markdown("---")
    # Per-player hide / remove
    st.subheader("Player visibility")
    all_player_names_sidebar = sorted(
        df[[c for c in df.columns if c.lower() in {"player", "name", "full_name", "player_name"}][0]]
        .astype(str)
        .unique()
        .tolist()
    )
    hide_name_players = st.multiselect(
        "Hide labels (names only)",
        options=all_player_names_sidebar,
        help="Hides player name on chart but keeps their dot.",
    )
    remove_players = st.multiselect(
        "Remove players entirely",
        options=all_player_names_sidebar,
        help="Removes players from chart, table, profile, and radar.",
    )

    st.markdown("---")
    show_reg = st.checkbox("Show regression line", value=False)
    show_quad_lines = st.checkbox("Show quadrant lines", value=True)

    lock_axes = st.checkbox(
        "Lock chart axis ranges",
        value=False,
        help="Keep x/y ranges fixed based on all players (after minutes filter), regardless of filters.",
    )

    st.markdown("---")
    # Custom title / subtitle / axis labels / quadrant labels
    custom_title = st.text_input("Custom chart title (optional)", value="")
    custom_subtitle = st.text_input("Custom subtitle (optional)", value="")
    custom_x_label = st.text_input("Custom X-axis label (optional)", value="")
    custom_y_label = st.text_input("Custom Y-axis label (optional)", value="")
    reverse_x = st.checkbox("Reverse X-axis (high = bad)", value=False, key="rev_x")
    reverse_y = st.checkbox("Reverse Y-axis (high = bad)", value=False, key="rev_y")

    st.markdown("---")
    # Category filter for axis & radar stats
    st.subheader("Stat categories")
    all_categories = sorted({get_category(s) for s in stat_candidates_all})
    selected_categories = st.multiselect(
        "Categories for axes & radar",
        options=all_categories,
        default=all_categories,
        help="X/Y dropdowns and radar stats will only use these categories.",
    )

    # ── Quadrant labels (BOTTOM OF SIDEBAR) ───────────────────────────────────
    st.markdown("---")
    st.subheader("Quadrant labels")
    enable_quad_labels = st.checkbox("Enable quadrant labels", value=False)

    quad_choices = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    quads_to_label = st.multiselect(
        "Label which quadrants",
        options=quad_choices,
        default=quad_choices if enable_quad_labels else [],
        help="Choose the quadrants to label.",
    )

    if enable_quad_labels:
        st.caption("Customize label text")
        qlabel_tl = st.text_input("Top-Left label", value="High Y • Low X")
        qlabel_tr = st.text_input("Top-Right label", value="High Y • High X")
        qlabel_bl = st.text_input("Bottom-Left label", value="Low Y • Low X")
        qlabel_br = st.text_input("Bottom-Right label", value="Low Y • High X")
        qlabel_size = st.slider("Label font size", 10, 26, 15, 1)
        qlabel_alpha = st.slider("Label background opacity", 0.00, 1.00, 0.20, 0.05)
    else:
        # defaults if off
        qlabel_tl, qlabel_tr = "High Y • Low X", "High Y • High X"
        qlabel_bl, qlabel_br = "Low Y • Low X", "Low Y • High X"
        qlabel_size, qlabel_alpha = 15, 0.20

    st.markdown("---")
    if st.button("Randomize"):
        # Random theme
        theme = random.choice(["Light", "Dark"])

        # We'll set x/y stats later after axis_candidates exist
        st.session_state["randomize_requested"] = True

        # Random toggles (stored in session)
        st.session_state["show_labels"] = random.choice([True, False])
        st.session_state["show_reg"] = random.choice([True, False])
        st.session_state["show_quad_lines"] = random.choice([True, False])
        st.session_state["use_player_images"] = random.choice([True, False])

        st.rerun()   # ✅ NEW VERSION

    # ──────────────────────────────────────────────────────────────────────────

# Validate required columns
label_candidates = [c for c in df.columns if c.lower() in {"player", "name", "full_name", "player_name"}]
label_col = label_candidates[0] if label_candidates else st.sidebar.selectbox("Label column", df.columns)

if "Minutes" not in df.columns:
    st.error("Your CSV must include a 'Minutes' column.")
    st.stop()

# ==== Filter by minutes ====
df = df.copy()
df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
df = df.dropna(subset=["Minutes"])

# If randomizer changed min_minutes, use it
if "min_minutes" in st.session_state:
    min_minutes = st.session_state["min_minutes"]

if min_minutes:
    df = df[df["Minutes"] >= min_minutes]
if df.empty:
    st.warning("No rows after filtering. Try lowering the minimum minutes.")
    st.stop()

# Numeric cols AFTER minutes filter
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Axis/radar candidates = numeric stats (not Minutes) within selected categories
axis_candidates = [
    c for c in numeric_cols if c != "Minutes" and get_category(c) in selected_categories
]

if not axis_candidates:
    st.error("No numeric stats available for the selected categories. Adjust category filter or minutes filter.")
    st.stop()

# ---- Axis selection with swap support (using axis_candidates) ----
if "x_stat_current" not in st.session_state:
    st.session_state["x_stat_current"] = axis_candidates[0]
if "y_stat_current" not in st.session_state:
    st.session_state["y_stat_current"] = axis_candidates[1] if len(axis_candidates) > 1 else axis_candidates[0]

# Apply randomization of axes if requested
if st.session_state.get("randomize_requested"):
    x_stat_random = random.choice(axis_candidates)
    y_stat_random = random.choice(axis_candidates)
    while y_stat_random == x_stat_random and len(axis_candidates) > 1:
        y_stat_random = random.choice(axis_candidates)

    st.session_state["x_stat_current"] = x_stat_random
    st.session_state["y_stat_current"] = y_stat_random

    # also plug back saved toggles if present
    show_labels = st.session_state.get("show_labels", show_labels)
    show_reg = st.session_state.get("show_reg", show_reg)
    show_quad_lines = st.session_state.get("show_quad_lines", show_quad_lines)
    reverse_x = st.session_state.get("reverse_x", reverse_x)
    reverse_y = st.session_state.get("reverse_y", reverse_y)
    use_player_images = st.session_state.get("use_player_images", use_player_images)

    st.session_state["randomize_requested"] = False

col1, col2, col3 = st.columns([1.2, 1.2, 0.8])

with col3:
    st.markdown("<br>", unsafe_allow_html=True)  # small vertical spacing
    if st.button("Swap X ↔ Y"):
        (
            st.session_state["x_stat_current"],
            st.session_state["y_stat_current"],
        ) = (
            st.session_state["y_stat_current"],
            st.session_state["x_stat_current"],
        )

# Figure out indices based on current state (safe fallback if something changed)
try:
    x_index = axis_candidates.index(st.session_state["x_stat_current"])
except ValueError:
    x_index = 0

try:
    y_index = axis_candidates.index(st.session_state["y_stat_current"])
except ValueError:
    y_index = 1 if len(axis_candidates) > 1 else 0

with col1:
    x_stat = st.selectbox(
        "X-axis stat",
        axis_candidates,
        index=x_index,
        key="x_stat_select_main",
    )

with col2:
    y_stat = st.selectbox(
        "Y-axis stat",
        axis_candidates,
        index=y_index,
        key="y_stat_select_main",
    )

# Update our state with any manual changes from the dropdowns
st.session_state["x_stat_current"] = x_stat
st.session_state["y_stat_current"] = y_stat

# Use CSV stats as-is (already per-90)
x_key = x_stat
y_key = y_stat

# ==== Percentiles for ALL numeric stats (except Minutes) vs ALL Chelsea players (after minutes filter) ====
for col in stat_candidates_all:
    if col in df.columns:
        df[f"{col}_pct"] = df[col].rank(pct=True) * 100

# Position-based percentiles (if Pos exists)
if "Pos" in df.columns:
    for col in stat_candidates_all:
        if col in df.columns:
            df[f"{col}_pct_pos"] = df.groupby("Pos")[col].rank(pct=True) * 100

# ==== helper: player images (already cropped, handle accents) ====
IMAGES_DIR = Path(__file__).parent / "PlayerImages"


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def get_player_image_path(player_name: str):
    base = player_name.strip()
    base_no_accents = _strip_accents(base)

    candidates = [
        IMAGES_DIR / f"{base}.png",
        IMAGES_DIR / f"{base.replace(' ', '_')}.png",
        IMAGES_DIR / f"{base_no_accents}.png",
        IMAGES_DIR / f"{base_no_accents.replace(' ', '_')}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ==== Working copy for plotting / position + player filters ====
plot_df = df.copy()

# Filter by Position (using sidebar checkboxes)
if "Pos" in plot_df.columns and selected_positions:
    pos_series_plot = (
        plot_df["Pos"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.split(r"[/, ]")
        .str[0]
    )
    plot_df = plot_df[pos_series_plot.isin(selected_positions)].copy()

# Remove players entirely if requested
if remove_players:
    plot_df = plot_df[~plot_df[label_col].isin(remove_players)].copy()

if plot_df.empty:
    st.warning("No rows after position/player filters. Adjust filters or removed players.")
    st.stop()

# ==== Axis ranges (global vs plot-specific) ====
# Global ranges from df (after minutes filter, before position/remove filters)
x_all = pd.to_numeric(df[x_key], errors="coerce").to_numpy()
y_all = pd.to_numeric(df[y_key], errors="coerce").to_numpy()
valid_all = np.isfinite(x_all) & np.isfinite(y_all)

if valid_all.sum() >= 1:
    global_x_min = np.nanmin(x_all[valid_all])
    global_x_max = np.nanmax(x_all[valid_all])
    global_y_min = np.nanmin(y_all[valid_all])
    global_y_max = np.nanmax(y_all[valid_all])
else:
    global_x_min = global_x_max = global_y_min = global_y_max = None

# Plot-specific ranges
x_vals_plot = pd.to_numeric(plot_df[x_key], errors="coerce").to_numpy()
y_vals_plot = pd.to_numeric(plot_df[y_key], errors="coerce").to_numpy()
valid_plot = np.isfinite(x_vals_plot) & np.isfinite(y_vals_plot)

if not np.any(valid_plot):
    st.warning("No valid x/y values to plot for current filters.")
    st.stop()

if lock_axes and global_x_min is not None:
    x_min_plot, x_max_plot = global_x_min, global_x_max
    y_min_plot, y_max_plot = global_y_min, global_y_max
else:
    x_min_plot = np.nanmin(x_vals_plot[valid_plot])
    x_max_plot = np.nanmax(x_vals_plot[valid_plot])
    y_min_plot = np.nanmin(y_vals_plot[valid_plot])
    y_max_plot = np.nanmax(y_vals_plot[valid_plot])

# Small padding
x_pad = (x_max_plot - x_min_plot) * 0.05 if x_max_plot > x_min_plot else 0.1
y_pad = (y_max_plot - y_min_plot) * 0.05 if y_max_plot > y_min_plot else 0.1
x_min_plot -= x_pad
x_max_plot += x_pad
y_min_plot -= y_pad
y_max_plot += y_pad

# Dot size scaled by Minutes (fixed, sqrt for readability)
mins = plot_df["Minutes"].to_numpy()
mins_norm = (mins - mins.min()) / (mins.max() - mins.min() + 1e-9)
sizes = 55 + 123 * np.sqrt(mins_norm)

# Color by position
pos_colors = {"FW": "#004CFF", "MF": "#008000", "DF": "#FFFF00"}
if "Pos" in plot_df.columns:
    pos_series = (
        plot_df["Pos"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    primary_pos = pos_series.str.split("[,/ ]").str[0]
    color_series = primary_pos.map(pos_colors).fillna("lightgray")
else:
    color_series = pd.Series(["lightgray"] * len(plot_df))


# ==== Aesthetics ====
BG_DARK = "#0E1117"
AX_DARK = "#111827"
BG_LIGHT = "#FAFAF7"
AX_LIGHT = "#F7F7F2"
EDGE_LIGHT = "#E5E7EB"
EDGE_DARK = "#1F2937"
TEXT_LIGHT = "#111827"
TEXT_DARK = "#E5E7EB"

plt.rcParams["lines.antialiased"] = True
plt.rcParams["patch.antialiased"] = True
plt.rcParams["figure.dpi"] = 350
plt.rcParams["savefig.dpi"] = 350

fig, ax = plt.subplots(figsize=(14, 8.8), dpi=350)

if theme == "Dark":
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(AX_DARK)
    label_color = TEXT_DARK
    edge_color = "#E5E7EB"
    spine_color = "#4B5563"
    grid_ls = "-"
    grid_alpha = 0.30
else:
    fig.patch.set_facecolor(AX_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    label_color = TEXT_LIGHT
    edge_color = "#111827"
    spine_color = "#3A3A3A"
    grid_ls = "-"
    grid_alpha = 0.48

# FINAL override on spines (width + base alpha)
for spine in ax.spines.values():
    spine.set_linewidth(0.585)
    spine.set_alpha(0.8)

# ===== Scatter or Player Images =====
dots_z = 10

if use_player_images:
    FIXED_ZOOM = 0.30   # adjust to taste (0.18–0.26 is a good range)

    def crop_top_portion(arr, keep=0.60):
        """Keep only the top `keep` fraction of the image (e.g., head/shoulders)."""
        h = arr.shape[0]
        k = max(1, int(h * keep))
        return arr[:k, :, :]

    matched, missing = 0, 0
    for i, row in plot_df.iterrows():
        x, y = float(row[x_key]), float(row[y_key])
        img_path = get_player_image_path(str(row[label_col]))

        if img_path and img_path.exists():
            try:
                arr = mpimg.imread(str(img_path))          # RGBA if PNG
                arr = crop_top_portion(arr, keep=0.6137)   # top ~60% only
                oi = OffsetImage(arr, zoom=FIXED_ZOOM)     # fixed size
                ab = AnnotationBbox(
                    oi, (x, y),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    zorder=dots_z,
                    clip_on=False,
                )
                ax.add_artist(ab)
                matched += 1
            except Exception:
                c = (color_series.iloc[i] if 'color_series' in locals() else "lightgray")
                ax.scatter([x], [y], s=36, c=[c], zorder=dots_z)
                missing += 1
        else:
            c = (color_series.iloc[i] if 'color_series' in locals() else "lightgray")
            ax.scatter([x], [y], s=36, c=[c], zorder=dots_z)
            missing += 1

    print(f"[Images] matched: {matched}, missing: {missing}, dir: {IMAGES_DIR}")
else:
    ax.scatter(
        plot_df[x_key],
        plot_df[y_key],
        s=sizes,
        c=color_series,
        alpha=.9,
        edgecolors=edge_color,
        linewidths=.55,
        zorder=dots_z,
    )

if reverse_x:
    ax.invert_xaxis()
if reverse_y:
    ax.invert_yaxis()

# ================================
# Vertical gradient: bottom -> top
# ================================
from matplotlib import colors as mcolors

def add_vertical_gradient_axes(ax, bottom_color, top_color,
                               x0=0.0, x1=1.0, y0=0.86, y1=1.0, z=0.5):
    """
    Draw a vertical RGB gradient in axes coordinates (independent of data limits).
    bottom_color at y0 blends to top_color at y1.
    """
    h = 256
    t = np.linspace(0.0, 1.0, h).reshape(h, 1)          # 0 at bottom, 1 at top
    rb, gb, bb, _ = mcolors.to_rgba(bottom_color, 1.0)
    rt, gt, bt, _ = mcolors.to_rgba(top_color,    1.0)

    img = np.zeros((h, 2, 4), dtype=float)              # 2px wide image that will be stretched
    img[..., 0] = (1 - t) * rb + t * rt                 # R
    img[..., 1] = (1 - t) * gb + t * gt                 # G
    img[..., 2] = (1 - t) * bb + t * bt                 # B
    img[..., 3] = 1.0                                   # fully opaque

    ax.imshow(
        img,
        extent=[x0, x1, y0, y1],
        origin="lower",
        transform=ax.transAxes,
        interpolation="bicubic",
        zorder=z,
        clip_on=False,
    )

# Choose colors per theme:
# bottom = chart (axes) facecolor, top = figure background color
if theme == "Dark":
    grad_bottom = AX_DARK
    grad_top    = BG_DARK
else:
    grad_bottom = BG_LIGHT
    grad_top    = AX_LIGHT

# Draw the gradient behind dots/images
add_vertical_gradient_axes(ax, bottom_color=grad_bottom, top_color=grad_top, z=0.5)

# --- Corner hue overlay (axes coords) ---
def add_corner_hue(ax, color, x0, x1, y0, y1, alpha=0.22, z=0.8, pow=1.4, corner="top-right"):
    """
    Draw a soft color wash inside [x0,x1]x[y0,y1] in axes coords (0..1).
    Fade is strongest at the chosen corner and decreases inward.
    corner ∈ {"top-right","top-left","bottom-right","bottom-left"}
    """
    h, w = 128, 128
    u = np.linspace(0, 1, w)  # 0=left, 1=right
    v = np.linspace(0, 1, h)  # 0=bottom, 1=top
    U, V = np.meshgrid(u, v)

    # Weight field: 1 at the specified corner, 0 toward opposite sides
    if corner == "top-right":
        W = (U) * (V)
    elif corner == "top-left":
        W = (1 - U) * (V)
    elif corner == "bottom-right":
        W = (U) * (1 - V)
    else:  # "bottom-left"
        W = (1 - U) * (1 - V)

    W = np.clip(W, 0, 1) ** pow

    r, g, b, _ = mcolors.to_rgba(color, 1.0)
    img = np.zeros((h, w, 4), dtype=float)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    img[..., 3] = W * alpha

    ax.imshow(
        img,
        extent=[x0, x1, y0, y1],
        origin="lower",
        transform=ax.transAxes,
        interpolation="bicubic",
        zorder=z,
        clip_on=False,
    )

# Slightly softer hues in dark mode
hue_alpha = 0.20 if theme == "Dark" else 0.28

# Top-right green
add_corner_hue(
    ax,
    color="#22c55e",
    x0=0.50, x1=1.00,
    y0=0.69, y1=1.00,
    alpha=hue_alpha,
    z=0.8,
    pow=1.0,
    corner="top-right"
)

# Bottom-left red
add_corner_hue(
    ax,
    color="#ef4444",
    x0=0.00, x1=0.50,
    y0=0.00, y1=0.31,
    alpha=hue_alpha,
    z=0.8,
    pow=1.0,
    corner="bottom-left"
)

# Labels (hide when using player images)
if show_labels and not use_player_images:
    for _, row in plot_df.iterrows():
        if row[label_col] in hide_name_players:
            continue

        # Split name
        full_name = str(row[label_col]).split()
        if len(full_name) >= 2:
            first_name = full_name[0]
            last_name = " ".join(full_name[1:])
        else:
            first_name = full_name[0]
            last_name = ""

        # Theme-aware label box
        if theme == "Dark":
            label_box_face = AX_DARK
            label_box_edge = "#E5E7EB"
            label_box_alpha = 0.85
        else:
            label_box_face = "#FCFCFA"
            label_box_edge = "#FFFFFF"
            label_box_alpha = 0.25

        ax.annotate(
            f"{first_name}\n{last_name}",
            (row[x_key], row[y_key]),
            textcoords="offset points",
            xytext=(0, 22.2),
            fontsize=13.7,
            fontproperties=golos_font,
            color=label_color,
            va="center",
            ha="center",
            zorder=20,
            bbox=dict(
                boxstyle="round,pad=0.13",
                edgecolor=label_box_edge,
                facecolor=label_box_face,
                linewidth=0.8,
                alpha=label_box_alpha,
            ),
        )

# Regression line (optional)
if show_reg and len(plot_df) >= 2:
    xvals = x_vals_plot[valid_plot]
    yvals = y_vals_plot[valid_plot]
    if len(xvals) >= 2:
        m, b = np.polyfit(xvals, yvals, 1)
        xs = np.linspace(x_min_plot, x_max_plot, 100)
        ys = m * xs + b
        reg_color = "white" if theme == "Dark" else "black"
        ax.plot(xs, ys, linestyle="-", linewidth=2.3, color=reg_color, alpha=0.75)

# Quadrants & labels (with separate toggles) using the same axis limits
if show_quad_lines:
    x_min, x_max = x_min_plot, x_max_plot
    y_min, y_max = y_min_plot, y_max_plot
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    quad_line_color = TEXT_LIGHT if theme == "Light" else EDGE_LIGHT

    ax.axvline(x_mid, color=quad_line_color, linestyle="-", linewidth=1.7, alpha=0.8)
    ax.axhline(y_mid, color=quad_line_color, linestyle="-", linewidth=1.7, alpha=0.8)

# --- Axis limits (apply reverse toggles by swapping limits) ---
x_lo, x_hi = x_min_plot, x_max_plot
y_lo, y_hi = y_min_plot, y_max_plot

if reverse_x:
    x_lo, x_hi = x_hi, x_lo
if reverse_y:
    y_lo, y_hi = y_hi, y_lo

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)

# ----- Quadrant labels -----
if enable_quad_labels and quads_to_label:
    x_mid = 0.5 * (x_min_plot + x_max_plot)
    y_mid = 0.5 * (y_min_plot + y_max_plot)

    x_left_mid = 0.5 * (x_min_plot + x_mid)
    x_right_mid = 0.5 * (x_mid + x_max_plot)

    pad_frac = 0.015
    yr = y_max_plot - pad_frac * (y_max_plot - y_min_plot)
    yb = y_min_plot + pad_frac * (y_max_plot - y_min_plot)

    if theme == "Dark":
        q_text_color = TEXT_DARK
        q_face_color = AX_DARK
    else:
        q_text_color = "#3A3A3A"
        q_face_color = "#F7F7F2"

    def _qlabel(text, x, y, valign):
        ax.text(
            x, y, text,
            ha="center", va=valign,
            fontsize=qlabel_size,
            fontproperties=hoves_font,
            color=q_text_color,
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=q_face_color,
                edgecolor="none",
                alpha=qlabel_alpha,
            ),
        )

    if "Top-Left" in quads_to_label:
        _qlabel(qlabel_tl, x_left_mid, yr, valign="top")
    if "Top-Right" in quads_to_label:
        _qlabel(qlabel_tr, x_right_mid, yr, valign="top")
    if "Bottom-Left" in quads_to_label:
        _qlabel(qlabel_bl, x_left_mid, yb, valign="bottom")
    if "Bottom-Right" in quads_to_label:
        _qlabel(qlabel_br, x_right_mid, yb, valign="bottom")

# Axes formatting — keep left/bottom spines, remove top/right
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

if theme == "Dark":
    spine_col = "#9CA3AF"
    tick_col = TEXT_DARK
else:
    spine_col = "#3A3A3A"
    tick_col = "#3A3A3A"

ax.spines["left"].set_color(spine_col)
ax.spines["bottom"].set_color(spine_col)

# Tick labels
ax.tick_params(axis="both", labelsize=11.1, colors=tick_col, length=0)
ax.tick_params(axis="x", pad=6)
ax.tick_params(axis="y", pad=6)

# Grid and labels
ax.grid(True, linestyle=grid_ls, alpha=grid_alpha)

# Axis labels (custom if provided)
x_label_final = custom_x_label.strip() if custom_x_label.strip() else x_key
y_label_final = custom_y_label.strip() if custom_y_label.strip() else y_key

ax.set_xlabel(
    x_label_final,
    fontsize=18.4,
    fontproperties=hovesmedium_font,
    color=label_color,
    labelpad=14.5,
)

ax.set_ylabel(
    y_label_final,
    fontsize=18,
    fontproperties=hovesmedium_font,
    color=label_color,
    labelpad=5,
)

# === Chelsea logo (top-left) ===
logo_path = Path(__file__).parent / "Chelsea.png"
if logo_path.exists():
    logo_img = mpimg.imread(logo_path)
    imagebox = OffsetImage(logo_img, zoom=0.05)
    ab = AnnotationBbox(imagebox, (0.07, 0.916), frameon=False, xycoords="axes fraction", zorder=1)
    ax.add_artist(ab)

# Fake extra space around the figure
fig.text(0.5, 1.045, " ", fontsize=1, ha="center", alpha=0)
fig.text(0.5, 0.035, " ", fontsize=1, ha="center", alpha=0)
fig.text(0.068, 0.05, " ", fontsize=1, ha="center", alpha=0)
fig.text(0.914, 0.05, " ", fontsize=1, ha="center", alpha=0)

# === Titles and credits ===
default_main_title = f" {y_key} vs {x_key}"
default_subtitle = "Chelsea Players | All Comps | Per 90"
main_title = custom_title.strip() if custom_title.strip() else default_main_title
subtitle = custom_subtitle.strip() if custom_subtitle.strip() else default_subtitle
credit = "Data via FBref.com and Sofascore"
watermark = "@sambrazy"

fig.text(
    0.5, 0.97,
    main_title,
    fontsize=32,
    fontproperties=hovesbold_font,
    ha="center",
    color=label_color,
)
fig.text(
    0.5, 0.929,
    subtitle,
    fontsize=19.2,
    fontproperties=hoves_font,
    ha="center",
    color=("#222222" if theme == "Light" else TEXT_DARK),
)
fig.text(
    0.759, 0.053,     # moved slightly left to make space for images
    credit,
    fontsize=8.6,
    fontproperties=hoves_font,
    ha="left",
    color=("#555555" if theme == "Light" else TEXT_DARK),
)
fig.text(
    0.22, 0.9,
    watermark,
    fontsize=15,
    fontproperties=hoves_font,
    ha="right",
    color=("black" if theme == "Light" else TEXT_DARK),
)

# === Add image icons next to credit text ===
icon_y = 0.0554   # same vertical level as credit
icon_size = 0.036 # adjust for size

# FBref logo (left icon)
fbref_path = Path(__file__).parent / "fbref.png"
if fbref_path.exists():
    fbref_img = mpimg.imread(fbref_path)
    ab_fb = AnnotationBbox(
        OffsetImage(fbref_img, zoom=.055),
        (0.752, 0.03894),            # x-position right after credit text
        xycoords="figure fraction",
        frameon=False,
        zorder=1
    )
    fig.add_artist(ab_fb)

# Sofascore logo (right icon)
sofa_path = Path(__file__).parent / "sofascore.png"
if sofa_path.exists():
    sofa_img = mpimg.imread(sofa_path)
    ab_sofa = AnnotationBbox(
        OffsetImage(sofa_img, zoom=.03),
        (0.774, 0.038),            # slightly further right
        xycoords="figure fraction",
        frameon=False,
        zorder=1
    )
    fig.add_artist(ab_sofa)
# ==== final render: chart ====
st.pyplot(fig, use_container_width=True, clear_figure=True)

# ==== table with data used in the chart ====
table_df = plot_df[[label_col, "Minutes", x_key, y_key]].copy()

col_names = [
    label_col,
    "Minutes",
    f"{x_label_final} per 90",
    f"{y_label_final} per 90",
]
table_df.columns = col_names

table_df.index = np.arange(1, len(table_df) + 1)

st.subheader("Data Table")
st.dataframe(table_df)

# ==== helper: render stat + percentile bar ====
def render_stat_with_percentile(col, label: str, value: float, pct: float):
    pct_clamped = max(0, min(100, pct))

    # Compute color from red -> yellow -> green
    if pct_clamped <= 50:
        t = pct_clamped / 50.0
        r, g, b = 255, int(255 * t), 0  # (255,0,0) -> (255,255,0)
    else:
        t = (pct_clamped - 50.0) / 50.0
        r, g, b = int(255 * (1 - t)), 255, 0  # (255,255,0) -> (0,255,0)
    bar_color = f"rgb({r},{g},{b})"

    html = f"""
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin-bottom: 0.75rem;">
        <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.1rem;">{label}</div>
        <div style="font-size: 1.35rem; font-weight: 600; margin-bottom: 0.1rem;">{value:.2f}</div>
        <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">{pct_clamped:.0f}th percentile</div>
        <div style="height: 6px; width: 100%; background-color: #e5e7eb; border-radius: 999px; overflow: hidden;">
            <div style="
                height: 100%;
                width: {pct_clamped}%;
                background-color: {bar_color};
                border-radius: 999px;
            "></div>
        </div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)


# ==== player profile picker ====
player_names = table_df.iloc[:, 0].tolist()
if player_names:
    selected_player = st.selectbox(
        "Select a player to view profile",
        player_names,
        index=0,
    )

    player_row = plot_df[plot_df[label_col] == selected_player].iloc[0]

    x_val = float(player_row[x_key])
    y_val = float(player_row[y_key])

    profile_pct_mode = st.radio(
        "Profile percentile mode",
        ["All Chelsea players", "Only same position"],
        horizontal=True,
        key="profile_pct_mode",
    )

    if (
        profile_pct_mode == "Only same position"
        and f"{x_key}_pct_pos" in player_row.index
        and f"{y_key}_pct_pos" in player_row.index
    ):
        x_pct_col = f"{x_key}_pct_pos"
        y_pct_col = f"{y_key}_pct_pos"
    else:
        x_pct_col = f"{x_key}_pct"
        y_pct_col = f"{y_key}_pct"

    x_pct = float(player_row[x_pct_col])
    y_pct = float(player_row[y_pct_col])

    st.markdown("---")
    st.subheader(f"Profile - {selected_player}")

    img_path = get_player_image_path(selected_player)
    if img_path is not None:
        img_cols = st.columns([1, 2, 1])
        with img_cols[1]:
            st.image(str(img_path), use_container_width=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        c1.markdown(
            f"""
            <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin-bottom: 0.75rem;">
                <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.1rem;">Minutes</div>
                <div style="font-size: 1.35rem; font-weight: 600; margin-bottom: 0.1rem;">{int(player_row["Minutes"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_stat_with_percentile(c2, x_label_final, x_val, x_pct)
    render_stat_with_percentile(c3, y_label_final, y_val, y_pct)

    # --- Show more toggle ---
    show_more = st.checkbox("Show more", value=False)

    if show_more:
        st.markdown("#### Full stat profile")

        category_to_stats = {}
        for col_name in stat_candidates_all:
            if col_name == "Minutes":
                continue
            if col_name not in player_row or f"{col_name}_pct" not in player_row:
                continue
            cat = get_category(col_name)
            category_to_stats.setdefault(cat, []).append(col_name)

        for cat in sorted(category_to_stats.keys()):
            stats = category_to_stats[cat]

            pcts = [float(player_row[f"{col_name}_pct"]) for col_name in stats]
            avg_pct = sum(pcts) / len(pcts) if pcts else 0.0

            st.markdown(
                f"""
                <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                            margin-top: 0.75rem; margin-bottom: 0.25rem;">
                    <strong>{cat}</strong>
                    &nbsp;&nbsp;
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cols = st.columns(3)
            for i, col_name in enumerate(stats):
                val = float(player_row[col_name])
                pct = float(player_row[f"{col_name}_pct"])
                render_stat_with_percentile(cols[i % 3], col_name, val, pct)

    # --- Radar chart (Beta) ---
    st.markdown("#### Radar chart (Beta)")

    radar_all_options = axis_candidates.copy()

    radar_categories_available = sorted({get_category(s) for s in radar_all_options})
    radar_category_choice = st.selectbox(
        "Radar stat category",
        ["All"] + radar_categories_available,
        index=0,
        help="Pick a category to build the radar only from that area, or 'All' to mix.",
    )

    if radar_category_choice == "All":
        radar_options = radar_all_options
    else:
        radar_options = [s for s in radar_all_options if get_category(s) == radar_category_choice]

    if not radar_options:
        st.warning("No stats available for this radar category. Try a different category.")
    else:
        default_radar = radar_options

        radar_stats = st.multiselect(
            "Stats to include on radar",
            options=radar_options,
            default=default_radar,
            help="These use percentile values (0–100).",
        )

        mode = st.radio(
            "Radar percentile mode",
            ["All Chelsea players", "Only same position"],
            horizontal=True,
        )

        compare_pool = [p for p in player_names if p != selected_player]
        compare_players = st.multiselect(
            "Compare with (up to 3 other players)",
            options=compare_pool,
            help="Main player is in blue. Additional players use red, yellow, and orange.",
        )
        if len(compare_players) > 3:
            compare_players = compare_players[:3]
            st.info("Only the first 3 comparison players will be used on the radar.")

        if radar_stats:
            if mode == "Only same position" and all(
                f"{col}_pct_pos" in player_row for col in radar_stats
            ):
                suffix = "_pct_pos"
            else:
                suffix = "_pct"

            raw_values_main = [float(player_row[col]) for col in radar_stats]
            pct_values_main = [float(player_row[f"{col}{suffix}"]) for col in radar_stats]
            pct_values_main_loop = pct_values_main + pct_values_main[:1]

            player_objs = [(selected_player, player_row)]
            for name in compare_players:
                match = plot_df[plot_df[label_col] == name]
                if not match.empty:
                    player_objs.append((name, match.iloc[0]))

            color_cycle = ["#2563eb", "#dc2626", "#eab308", "#f97316"]

            angles = np.linspace(0, 2 * np.pi, len(radar_stats), endpoint=False)
            angles_loop = np.concatenate([angles, [angles[0]]])

            radar_col, profile_col = st.columns([6, 1.4])

            with radar_col:
                fig_radar, ax_radar = plt.subplots(
                    subplot_kw={"projection": "polar"},
                    figsize=(5.5, 5.5),
                    dpi=740,
                )

                if theme == "Dark":
                    fig_radar.patch.set_facecolor(BG_DARK)
                    ax_radar.set_facecolor(AX_DARK)
                    radar_label_color = TEXT_DARK
                else:
                    fig_radar.patch.set_facecolor("#FFFFFF")
                    ax_radar.set_facecolor("#FFFFFF")
                    radar_label_color = TEXT_LIGHT

                handles = []
                labels_leg = []

                for idx, (pname, prow) in enumerate(player_objs):
                    color = color_cycle[idx % len(color_cycle)]

                    if idx == 0:
                        values = pct_values_main
                        values_loop = pct_values_main_loop
                    else:
                        values = [float(prow[f"{col}{suffix}"]) for col in radar_stats]
                        values_loop = values + values[:1]

                    line, = ax_radar.plot(
                        angles_loop,
                        values_loop,
                        linewidth=2.2,
                        alpha=.6,
                        color=color,
                    )
                    ax_radar.fill(
                        angles_loop,
                        values_loop,
                        alpha=0.24,
                        color=color,
                        zorder=20
                    )

                    handles.append(line)
                    labels_leg.append(pname)

                ax_radar.set_xticks(angles)
                ax_radar.set_xticklabels(
                    radar_stats,
                    fontproperties=golos_font,
                    fontsize=7.8,
                    color=radar_label_color,
                )

                ax_radar.set_yticks([0, 20, 40, 60, 80, 100])
                ax_radar.set_yticklabels([])
                ax_radar.set_ylim(0, 100)

                for angle, radius, raw_val in zip(angles, pct_values_main, raw_values_main):
                    r_text = min(radius + 8.3, 100)
                    ax_radar.text(
                        angle,
                        r_text,
                        f"{raw_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        fontproperties=golos_font,
                        color=radar_label_color,
                    )

                st.pyplot(fig_radar)

            with profile_col:
                def render_mini_profile(player_label, row, color_hex: str):
                    img_path = get_player_image_path(player_label)
                    if img_path is not None:
                        st.image(str(img_path), width=20000)

                    pos_val = str(row.get("Pos", "N/A"))
                    minutes_val = int(row.get("Minutes", 0))

                    st.markdown(
                        f"""
                        <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                                    border-left: 3px solid {color_hex};
                                    padding-left: 0.5rem;
                                    margin-bottom: 0.75rem;">
                            <div style="font-size: 0.8rem; color: #4b5563; margin-bottom: 0.05rem;">{player_label}</div>
                            <div style="font-size: 0.7rem; color: #6b7280;">
                                Pos: <span style="font-weight: 300;">{pos_val}</span><br/>
                                Minutes: <span style="font-weight: 300;">{minutes_val}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                for idx, (pname, prow) in enumerate(player_objs):
                    render_mini_profile(pname, prow, color_cycle[idx % len(color_cycle)])
