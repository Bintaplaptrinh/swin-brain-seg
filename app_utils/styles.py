"""Readable Vietnamese hospital-inspired visual theme."""
CLINICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
:root { --app-font: 'Be Vietnam Pro', 'Segoe UI', Arial, sans-serif; }
/* Inherit the Vietnamese typeface in native widgets without overriding icon fonts. */
.stApp, .stApp input, .stApp textarea, .stApp button, .stApp select,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMetric"],
.stApp [data-baseweb="select"], [data-baseweb="popover"] {
    font-family: var(--app-font);
}
.stApp { -webkit-font-smoothing: antialiased; }

html { font-size: 18px; }
.stApp { background: #f3f6f8; color: #203447; font-family: var(--app-font); }
[data-testid="stMainBlockContainer"] { padding-top: 4.5rem; max-width: 1560px; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #b7cadb; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 10px 8px; border-bottom: 1px solid #e1e9ef; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #e7f1fb; border-left: 4px solid #07559b; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p,
.stButton button p, .stDownloadButton button p { font-size: 1rem; line-height: 1.65; }
[data-testid="stCaptionContainer"] { color: #435b70; }
h1, h2, h3, h4 { font-family: var(--app-font); color: #064b87; }
h2, h3 { border-bottom: 2px solid #c6d9e9; padding-bottom: 10px; }
.hospital-masthead { background: #07559b; color: white; padding: 14px 22px; font-size: 1rem; font-weight: 700; border-top: 4px solid #d2ad52; line-height: 1.6; }
.hospital-masthead span { display: block; font-size: .9rem; font-weight: 400; }
.clinical-header { background: white; padding: 24px; border: 1px solid #b7cadb; border-top: 0; display:flex; justify-content:space-between; align-items:center; gap:20px; flex-wrap:wrap; margin-bottom: 16px; }
.clinical-title { font-size: 1.45rem; font-weight: 700; margin: 0; line-height: 1.4; color: #064b87; }
.clinical-subtitle { font-size: 1rem; line-height: 1.65; color: #435b70; margin: 8px 0 0; }
.system-status { border-left: 3px solid #07559b; padding-left: 14px; font-size: .9rem; }
.system-status strong { display: block; color: #07559b; margin-top: 6px; font-size: 1rem; }
.med-card, .metric-tile, .clinical-summary-sheet { background: white; border: 1px solid #b7cadb; border-radius: 12px; padding: 20px; margin-bottom: 12px; overflow-wrap: anywhere; box-shadow: none; }
.metric-tile { min-height: 270px; border-top: 4px solid #07559b; }
.metric-tile-label { color: #435b70; font-size: .95rem; font-weight:700; margin-bottom: 12px; }
.metric-tile-value { color: #064b87; font-size: 1.2rem; font-weight:700; line-height:1.4; }
.metric-tile-sub { color: #435b70; font-size: .95rem; margin-top: 10px; line-height:1.6; }
.clinical-summary-title { background:#e7f1fb; color:#064b87; padding:12px; font-size:1.1rem; font-weight:700; border-bottom:2px solid #07559b; }
.clinical-summary-row { display:flex; justify-content:space-between; gap:20px; padding:12px 0; border-bottom:1px solid #d5e0e9; flex-wrap:wrap; }
.summary-val { font-weight:700; }
.medical-disclaimer { border-top:2px solid #07559b; background:#e7f1fb; padding:20px; margin-top:30px; line-height:1.7; color:#294c6d; font-size:1rem; }
.status-pill { display:inline-block; border:1px solid #b7cadb; background:#e7f1fb; color:#07559b; padding:6px 12px; font-size:1rem; }
.stButton button, .stDownloadButton button { border-radius:10px; border:1px solid #07559b; min-height:46px; }
.stButton button[kind="primary"] { background:#07559b; color:white; }
[data-testid="stFileUploader"] { background:white; border:1px dashed #07559b; padding:16px; }
@media (max-width: 900px) { .clinical-title { font-size:1.35rem; } .metric-tile { min-height:160px; } }
[data-testid="stSidebar"] [role="radiogroup"] label { width:100%; }
[data-testid="stFileUploaderDropzoneInstructions"] { font-size:0; }
[data-testid="stFileUploaderDropzoneInstructions"] > div,
[data-testid="stFileUploaderDropzoneInstructions"] small { display:none; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content:"Kéo và thả ảnh vào đây · PNG, JPG, JPEG · Tối đa 200 MB"; font-size:1rem; line-height:1.6; }
[data-testid="stFileUploaderDropzone"] button { font-size:0; }
[data-testid="stFileUploaderDropzone"] button::after { content:"Chọn ảnh"; font-size:1rem; }
/* Compact heading scale: 26px page title, 22px sections, 18px body. */
.stApp h1 { font-size:1.45rem; line-height:1.4; }
.stApp h2 { font-size:1.3rem; line-height:1.45; }
.stApp h3 { font-size:1.2rem; line-height:1.5; }
.stApp h4 { font-size:1.1rem; line-height:1.5; }
.hospital-masthead { border-radius:14px 14px 0 0; }
.clinical-header { border-radius:0 0 14px 14px; }
.clinical-summary-title, .medical-disclaimer, .status-pill { border-radius:10px; }
[data-testid="stSidebar"] [role="radiogroup"] label { border-radius:10px; }
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] { border-radius:12px; }
[data-testid="stMetricValue"] { font-size:1.4rem; }
/* Soft glass controls: opaque fallback keeps labels readable without blur support. */
.stApp { background:radial-gradient(ellipse at 90% 5%, #e0edf8 0%, transparent 45%), #f3f6f8; }
[data-testid="stSidebar"] { background:rgba(249,252,255,.95); }
.nav-group-title { font-size:.9rem; font-weight:700; color:#526b80; letter-spacing:.04em; margin:8px 0 12px; }
.nav-information { margin-top:28px; }
.st-key-primary_navigation { background:linear-gradient(135deg,rgba(220,239,255,.95),rgba(245,251,255,.9)); border:1px solid #c1d9ec; border-radius:24px; padding:16px; box-shadow:inset 0 1px 0 white,0 8px 22px rgba(20,70,115,.08); }
.nav-active-note { display:block; font-size:.9rem; color:#07559b; font-weight:700; }
.stButton button, .stDownloadButton button, [data-testid="stFileUploaderDropzone"] button {
    border-radius:18px; padding:10px 18px; border:1px solid rgba(145,180,210,.65);
    background:linear-gradient(155deg,rgba(255,255,255,.96),rgba(221,237,249,.8));
    color:#124b78; box-shadow:inset 0 1px 0 rgba(255,255,255,.95),inset 0 -1px 0 rgba(100,145,180,.1),0 4px 12px rgba(25,70,110,.09);
    backdrop-filter:blur(14px) saturate(140%); -webkit-backdrop-filter:blur(14px) saturate(140%);
    transition:background .18s ease,box-shadow .18s ease,transform .18s ease;
}
.stButton button[kind="primary"] {
    background:linear-gradient(155deg,#398aca 0%,#1264a5 45%,#08518e 100%);
    border:1px solid rgba(255,255,255,.65); color:#fff;
    box-shadow:inset 0 1px 1px rgba(255,255,255,.65),inset 0 -2px 3px rgba(0,35,80,.15),0 6px 16px rgba(7,85,155,.2);
}
.st-key-primary_navigation button { min-height:62px; border-radius:20px; }
.st-key-primary_navigation button p { font-size:1.15rem; font-weight:700; }
.st-key-information_navigation button { text-align:left; justify-content:flex-start; box-shadow:inset 0 1px 0 white; background:rgba(255,255,255,.55); border:1px solid #dce6ee; }
.st-key-information_navigation button[kind="primary"] { background:#e5eff8; color:#07559b; border:1px solid #bbd3e6; box-shadow:inset 0 1px 0 white; }
.stButton button:hover, .stDownloadButton button:hover { transform:translateY(-1px); box-shadow:inset 0 1px 0 white,0 7px 18px rgba(25,70,110,.16); }
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline:3px solid #5ca3db; outline-offset:3px; }
.stButton button:active, .stDownloadButton button:active { transform:translateY(0); }
.stButton button:disabled { opacity:.55; transform:none; }
.med-card, .metric-tile, .clinical-summary-sheet { border-radius:22px; overflow:hidden; }
.hospital-masthead { border-radius:24px 24px 0 0; overflow:hidden; }
.clinical-header { border-radius:0 0 24px 24px; overflow:hidden; }
.clinical-summary-title, .medical-disclaimer { border-radius:16px; }
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"], [data-testid="stExpander"] details,
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stAlert"] { border-radius:20px; overflow:hidden; }
[data-baseweb="select"] > div, [data-baseweb="input"] { border-radius:14px; }
@media (prefers-reduced-motion: reduce) { .stButton button, .stDownloadButton button { transition:none; transform:none; } }
</style>
"""
