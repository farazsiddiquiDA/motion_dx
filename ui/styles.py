"""
Clinical design tokens and CSS styles for MOTION DX.
Provides a modern medical UI theme with high readability and zero neon/gaming distractions.
"""

CLINICAL_CSS = """
<style>
/* Main layout and container typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Header branding */
.clinical-brand-header {
    background: linear-gradient(135deg, #0d233a 0%, #17385c 100%);
    color: #ffffff;
    padding: 18px 24px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(13, 35, 58, 0.08);
}
.clinical-brand-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    color: #ffffff;
}
.clinical-brand-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
    font-weight: 400;
}

/* Top clinical safety bar */
.clinical-safety-alert {
    background-color: #fff8e6;
    border-left: 4px solid #f59e0b;
    padding: 10px 16px;
    border-radius: 4px;
    font-size: 12.5px;
    color: #78350f;
    margin-bottom: 18px;
    font-weight: 500;
}

/* Metric Display Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.metric-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 4px;
}
.metric-value-huge {
    font-size: 40px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.metric-value-large {
    font-size: 26px;
    font-weight: 600;
    color: #1e293b;
    line-height: 1.2;
}
.metric-subtext {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Status Badges */
.badge-good {
    display: inline-block;
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.badge-fair {
    display: inline-block;
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.badge-poor {
    display: inline-block;
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* Module Selection Cards */
.module-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 22px;
    transition: all 0.2s ease;
    height: 100%;
}
.module-card:hover {
    border-color: #2563eb;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
}
.module-icon {
    font-size: 28px;
    margin-bottom: 12px;
}
.module-title {
    font-size: 18px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 6px;
}
.module-desc {
    font-size: 13px;
    color: #64748b;
    line-height: 1.5;
}

/* Buttons */
.stButton>button {
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    padding: 0.5rem 1rem;
    transition: all 0.15s ease-in-out;
}
</style>
"""

def get_quality_badge_html(quality: str) -> str:
    q = quality.upper()
    if q == "GOOD":
        return '<span class="badge-good">● TRACKING: GOOD</span>'
    elif q == "FAIR":
        return '<span class="badge-fair">▲ TRACKING: FAIR</span>'
    else:
        return '<span class="badge-poor">■ TRACKING: POOR</span>'
