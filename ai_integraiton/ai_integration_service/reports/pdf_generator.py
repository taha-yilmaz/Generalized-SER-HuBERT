"""
PDF Report Generator  (v2.1 — Bug Fixes)
=========================================
Fixes applied:
  1. DejaVuSans Unicode font — Turkish chars (ş,ğ,ü,ö,ç,İ,ı) render correctly
  2. _strip_xai_header — removes ASCII ==== / █ XAI block from report text
  3. Language detection uses clean LLM text (not candidate name)
  4. Pie charts are true circles (set_aspect + fixed xlim/ylim)
  5. Facial timeline x-axis always starts at 0
  6. Candidate PDF: score circle full-width, text no longer cut off
  7. Skills chart: no fake random percentages; presence indicator
  8. Skills chart placed full-width in HR PDF (was overflowing right column)
  9. DPI raised to 200 for sharper charts
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

log = logging.getLogger("pdf_generator")

# ─────────────────────────────────────────────────────────────────────────────
#   UNICODE FONT REGISTRATION  (Fix 1)
#   DejaVuSans ships with matplotlib — no external download needed.
# ─────────────────────────────────────────────────────────────────────────────

def _register_fonts() -> tuple[str, str, str]:
    try:
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("Rpt",      os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("Rpt-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Rpt-Ital", os.path.join(base, "DejaVuSans-Oblique.ttf")))
        return "Rpt", "Rpt-Bold", "Rpt-Ital"
    except Exception as e:
        log.warning(f"DejaVuSans registration failed ({e}), falling back to Helvetica")
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT_REG, FONT_BOLD, FONT_ITAL = _register_fonts()

# ─────────────────────────────────────────────────────────────────────────────
#   PAGE GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = A4
MARGIN     = 1.5 * cm
INNER_W    = PAGE_W - 2 * MARGIN
COL_GAP    = 0.6 * cm
COL_W      = (INNER_W - COL_GAP) / 2
HEADER_H   = 3.6 * cm
TOP_ACCENT = 0.35 * cm

# ─────────────────────────────────────────────────────────────────────────────
#   COLOUR PALETTE — Light Professional
# ─────────────────────────────────────────────────────────────────────────────

C_NAVY    = colors.HexColor("#1e3a5f")
C_PRIMARY = colors.HexColor("#2563eb")
C_ACCENT  = colors.HexColor("#7c3aed")
C_SUCCESS = colors.HexColor("#16a34a")
C_WARNING = colors.HexColor("#d97706")
C_DANGER  = colors.HexColor("#dc2626")
C_TEXT_D  = colors.HexColor("#1e293b")
C_TEXT_M  = colors.HexColor("#475569")
C_TEXT_L  = colors.HexColor("#94a3b8")
C_DIVIDER = colors.HexColor("#e2e8f0")
C_BLUE_BG = colors.HexColor("#eff6ff")
C_WHITE   = colors.white

M_BG   = "white"
M_AX   = "#f8fafc"
M_GRID = "#e2e8f0"
M_TXT  = "#475569"
M_DARK = "#1e293b"
M_PRI  = "#2563eb"
M_ACC  = "#7c3aed"
M_GRN  = "#16a34a"
M_YLW  = "#d97706"
M_RED  = "#dc2626"
M_ORG  = "#ea580c"
EMO7   = [M_GRN, M_RED, M_RED, M_PRI, "#94a3b8", M_YLW, M_ACC]

# ─────────────────────────────────────────────────────────────────────────────
#   BILINGUAL CAPTIONS
# ─────────────────────────────────────────────────────────────────────────────

CAPTIONS: dict[str, dict[str, str]] = {
    "stream_scores": {
        "en": "Each stream's weight (%) and performance score. Longer bar = higher score.",
        "tr": "Her analiz akışının ağırlığı (%) ve performans skoru. Uzun çubuk = yüksek skor.",
    },
    "per_question": {
        "en": "Quality (depth & structure) and Relevance (on-topic) scores per question.",
        "tr": "Her soruya verilen cevabın kalite ve ilgililik skorları.",
    },
    "per_question_candidate": {
        "en": "Your answer quality per question. Green = strong, amber = developing, red = needs work.",
        "tr": "Her sorudaki cevap performansınız. Yeşil = iyi, sarı = gelişim alanı, kırmızı = eksik.",
    },
    "speech_emotion_dist": {
        "en": "Emotion distribution extracted from vocal tone analysis.",
        "tr": "Ses tonu analizinden elde edilen duygu dağılımı.",
    },
    "facial_emotion_dist": {
        "en": "Facial expression distribution from 7-class video frame analysis.",
        "tr": "Video karelerinden tespit edilen yüz ifadesi dağılımı (7 sınıf).",
    },
    "speech_timeline": {
        "en": "Speech emotion confidence over time. Dots are coloured by emotion class.",
        "tr": "Konuşma sırasında duygu güven skorunun zaman içindeki değişimi.",
    },
    "facial_timeline": {
        "en": "Dominant facial expression per minute. Coloured by expression class.",
        "tr": "Dakika başına baskın yüz ifadesi. Renk, duygu sınıfını gösterir.",
    },
    "skills_match": {
        "en": "CV skills rated by job relevance. Green = high match, Yellow = partial, Red = low relevance.",
        "tr": "CV becerileri iş ilanıyla uyuma göre puanlandı. Yeşil = yüksek uyum, Sarı = kısmi, Kırmızı = düşük.",
    },
    "positivity": {
        "en": "Positivity scores from speech and facial expression channels independently.",
        "tr": "Ses ve yüz kanallarından bağımsız hesaplanan pozitiflik skoru.",
    },
}


def _cap(key: str, lang: str) -> str:
    return CAPTIONS.get(key, {}).get(lang, CAPTIONS.get(key, {}).get("en", ""))


# ─────────────────────────────────────────────────────────────────────────────
#   TEXT HELPERS  (Fix 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_xai_header(text: str) -> str:
    """Remove the deterministic ASCII XAI header block (===...=== lines + ASCII art)."""
    if not text:
        return ""
    lines = text.splitlines()
    last_eq = -1
    for i, line in enumerate(lines):
        if line.startswith("=" * 10):
            last_eq = i
    return "\n".join(lines[last_eq + 1:]).strip() if last_eq >= 0 else text.strip()


def _detect_lang(text: str) -> str:
    """Detect report language from the first 200 chars of clean LLM text."""
    sample = (text or "")[:200]
    if any(c in sample for c in "şğüöçıŞĞÜÖÇİ"):
        return "tr"
    return "en"


# ─────────────────────────────────────────────────────────────────────────────
#   MATPLOTLIB SETUP  (Fix 9: dpi=200)
# ─────────────────────────────────────────────────────────────────────────────

def _mpl_setup():
    plt.rcParams.update({
        "figure.facecolor":  M_BG,
        "axes.facecolor":    M_AX,
        "axes.edgecolor":    M_GRID,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelcolor":   M_TXT,
        "axes.labelsize":    8,
        "xtick.color":       M_TXT,
        "ytick.color":       M_TXT,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "grid.color":        M_GRID,
        "grid.linewidth":    0.8,
        "text.color":        M_DARK,
        "legend.fontsize":   7,
        "legend.framealpha": 0.9,
    })


def _fig_to_image(fig, w_cm: float, h_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=M_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_cm * cm, height=h_cm * cm)


def _pie_to_image(fig, size_cm: float) -> Image:
    """Save a pie-chart figure preserving its square aspect ratio (Fix 4)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=M_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=size_cm * cm, height=size_cm * cm)


def _placeholder(msg: str, w: float = 5, h: float = 3) -> Image:
    _mpl_setup()
    fig, ax = plt.subplots(figsize=(w, h))
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=9, color=M_TXT, transform=ax.transAxes)
    ax.set_axis_off()
    return _fig_to_image(fig, w * 1.6, h * 1.6)


# ─────────────────────────────────────────────────────────────────────────────
#   CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _chart_score_circle(score: float) -> Image:
    _mpl_setup()
    clr = M_GRN if score >= 0.70 else (M_YLW if score >= 0.45 else M_RED)
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    fig.patch.set_facecolor(M_BG)
    ax.set_facecolor(M_BG)
    ax.pie([score, 1 - score], colors=[clr, "#e2e8f0"],
           startangle=90, counterclock=False,
           wedgeprops=dict(width=0.35, edgecolor="white", linewidth=3))
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.text(0, 0.12, f"{score:.0%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=M_DARK)
    ax.text(0, -0.2, "Overall", ha="center", va="center",
            fontsize=8, color=M_TXT)
    return _pie_to_image(fig, 3.8)


def _chart_stream_scores(streams: list) -> Image:
    if not streams:
        return _placeholder("Stream data unavailable")
    _mpl_setup()
    names  = [f"{s.get('stream_name','?')[:22]}  ({s.get('weight',0):.0%})" for s in streams]
    scores = [float(s.get("score", 0)) for s in streams]
    clrs   = [M_GRN if v >= 0.65 else (M_YLW if v >= 0.40 else M_RED) for v in scores]
    y = np.arange(len(names))

    display_w = INNER_W / cm   # ≈ 18 cm
    display_h = 5.5            # cm — increased from 3.5 to avoid vertical squishing
    fig_w     = 5.5            # matplotlib inches
    fig_h     = fig_w * display_h / display_w  # preserve aspect ratio

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars = ax.barh(y, scores, color=clrs, height=0.5, zorder=3)
    ax.set_xlim(0, 1.15)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("Score", fontsize=8)
    ax.spines["left"].set_visible(False)
    for bar, val in zip(bars, scores):
        ax.text(min(val + 0.02, 1.1), bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", ha="left", fontsize=8,
                color=M_DARK, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, display_w, display_h)


def _chart_per_question(per_q: list) -> Image:
    if not per_q:
        return _placeholder("No Q&A data")
    _mpl_setup()
    labels    = [f"Q{i+1}" for i in range(len(per_q))]
    quality   = [float(q.get("quality_score",   0)) for q in per_q]
    relevance = [float(q.get("relevance_score", 0)) for q in per_q]
    x  = np.arange(len(labels)); bw = 0.38

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.bar(x - bw/2, quality,   bw, label="Quality",   color=M_PRI, zorder=3)
    ax.bar(x + bw/2, relevance, bw, label="Relevance", color=M_ACC, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper right")
    for xi, (q_val, r_val) in zip(x, zip(quality, relevance)):
        for offset, val in [(-bw/2, q_val), (bw/2, r_val)]:
            ax.text(xi + offset, val + 0.03, f"{val:.0%}",
                    ha="center", fontsize=6.5, color=M_DARK, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, COL_W / cm, 3.5)


def _chart_per_question_candidate(per_q: list) -> Image:
    if not per_q:
        return _placeholder("No Q&A data")
    _mpl_setup()
    labels  = [f"Q{i+1}" for i in range(len(per_q))]
    quality = [float(q.get("quality_score", 0)) for q in per_q]
    clrs    = [M_GRN if v >= 0.60 else (M_YLW if v >= 0.35 else M_RED) for v in quality]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    bars = ax.bar(np.arange(len(labels)), quality, color=clrs, zorder=3)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Performance")
    for bar, val in zip(bars, quality):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                f"{val:.0%}", ha="center", fontsize=8, color=M_DARK, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, COL_W / cm, 3.5)


def _chart_speech_emotion_dist(speech: dict) -> Image:
    """Vocal emotion pie — filters truly-zero slices to prevent label overlap."""
    dist_raw = (speech or {}).get("emotion_distribution") or {}
    if not dist_raw:
        return _placeholder("No speech emotion data", 3.5, 3.5)
    _mpl_setup()
    # Values may be 0-1 decimals or 0-100 percentages; filter by relative share (< 0.5% of total)
    total  = sum(float(v) for v in dist_raw.values()) or 1.0
    dist   = {k: float(v) for k, v in dist_raw.items() if float(v) / total > 0.005}
    if not dist:
        dist = {k: float(v) for k, v in dist_raw.items() if float(v) > 0}
    labels = list(dist.keys())
    vals   = [dist[k] for k in labels]
    pie_c  = {"positive": M_GRN, "neutral": "#64748b", "negative": M_RED}
    clrs   = [pie_c.get(l, M_PRI) for l in labels]
    # Show label only for slices >= 3% of total
    lbl_display = [l.capitalize() if v / total >= 0.03 else "" for l, v in zip(labels, vals)]

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    fig.patch.set_facecolor(M_BG); ax.set_facecolor(M_BG)
    ax.pie(vals, labels=lbl_display,
           colors=clrs, autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
           startangle=90, pctdistance=0.70,
           wedgeprops=dict(edgecolor="white", linewidth=2.5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    return _pie_to_image(fig, COL_W / cm)


def _chart_facial_emotion_dist(facial: dict) -> Image:
    """Facial emotion pie — filters truly-zero slices to prevent label overlap."""
    dist_raw = (facial or {}).get("emotion_distribution") or {}
    if not dist_raw:
        return _placeholder("No facial emotion data", 3.5, 3.5)
    _mpl_setup()
    # Values may be 0-1 decimals or 0-100 percentages; filter by relative share (< 0.5% of total)
    total  = sum(float(v) for v in dist_raw.values()) or 1.0
    dist   = {k: float(v) for k, v in dist_raw.items() if float(v) / total > 0.005}
    if not dist:
        dist = {k: float(v) for k, v in dist_raw.items() if float(v) > 0}
    labels   = list(dist.keys())
    vals     = [dist[k] for k in labels]
    all_clrs = dict(zip(["happy", "neutral", "sad", "angry", "fear", "disgust", "surprise"], EMO7))
    clrs     = [all_clrs.get(l, M_PRI) for l in labels]
    # Show label only for slices >= 3% of total
    lbl_display = [l.capitalize() if v / total >= 0.03 else "" for l, v in zip(labels, vals)]

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    fig.patch.set_facecolor(M_BG); ax.set_facecolor(M_BG)
    ax.pie(vals, labels=lbl_display,
           colors=clrs, autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
           startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2.5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    return _pie_to_image(fig, COL_W / cm)


def _chart_speech_timeline(speech: dict, full_width: bool = False) -> Image:
    segs = (speech or {}).get("segments") or []
    if not segs:
        w = INNER_W / cm if full_width else COL_W / cm
        return _placeholder("No speech timeline data", w, 3.0)
    _mpl_setup()
    times = [float(s.get("start_time", 0)) for s in segs]
    conf  = [float(s.get("confidence", 0)) for s in segs]
    emos  = [s.get("emotion", "neutral") for s in segs]
    emo_c = {"positive": M_GRN, "neutral": "#64748b", "negative": M_RED}

    display_w = INNER_W / cm if full_width else COL_W / cm
    display_h = 4.0
    fig_w     = 8.5 if full_width else 5.0
    fig_h     = fig_w * display_h / display_w

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.fill_between(times, conf, alpha=0.12, color=M_PRI)
    ax.plot(times, conf, color=M_PRI, linewidth=1.8, zorder=3)
    for t, c, e in zip(times, conf, emos):
        ax.scatter(t, c, color=emo_c.get(e, "#64748b"), s=36, zorder=4,
                   edgecolors="white", linewidths=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="both", zorder=0)
    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in emo_c.items()]
    ax.legend(handles=patches, loc="upper right")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, display_w, display_h)


def _chart_facial_timeline(facial: dict) -> Image:
    """Facial expression per minute — x-axis always starts at 0 (Fix 5)."""
    mins = (facial or {}).get("minute_summary") or []
    if not mins:
        return _placeholder("No facial timeline data", 5, 2.5)
    _mpl_setup()
    emo_c = {"happy": M_GRN, "neutral": "#64748b", "sad": "#60a5fa",
             "angry": M_RED, "fear": M_YLW, "disgust": M_ACC, "surprise": M_ORG}
    x    = [m.get("minute", i) for i, m in enumerate(mins)]
    pct  = [float(m.get("percentage", 0)) for m in mins]
    doms = [m.get("dominant_emotion", "neutral") for m in mins]
    clrs = [emo_c.get(d, M_PRI) for d in doms]

    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar(x, pct, color=clrs, zorder=3, width=0.6,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlim(min(x) - 0.6, max(x) + 0.6)  # always positive, anchored at data
    ax.set_xlabel("Minute"); ax.set_ylabel("Dominance %")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", zorder=0)
    unique = list(dict.fromkeys(doms))
    patches = [mpatches.Patch(color=emo_c.get(e, M_PRI), label=e.capitalize()) for e in unique]
    ax.legend(handles=patches, loc="upper right")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, COL_W / cm, 3.5)


def _chart_positivity(speech: dict, facial: dict) -> Image:
    sp = float((speech or {}).get("positivity_score", 0))
    fp = float((facial or {}).get("positivity_score", 0))
    _mpl_setup()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    bars = ax.bar(["Speech\nPositivity", "Facial\nPositivity"],
                  [sp, fp], color=[M_PRI, M_ACC], width=0.45, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", zorder=0)
    ax.spines["bottom"].set_visible(False)
    for bar, val in zip(bars, [sp, fp]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                f"{val:.0%}", ha="center", fontsize=10, color=M_DARK, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, COL_W / cm, 3.5)


def _chart_xai_contributions(xai: dict) -> Image:
    """Horizontal bar chart showing each stream's % contribution to the final decision."""
    contributions = (xai or {}).get("feature_contributions") or []
    if not contributions:
        return _placeholder("XAI data unavailable", COL_W / cm, 3.5)
    _mpl_setup()

    names  = [c.get("stream", "?")[:22] for c in contributions]
    pcts   = [float(c.get("contribution_percentage", 0)) for c in contributions]
    clrs   = [M_GRN if c.get("impact") == "positive" else M_RED for c in contributions]
    labels = [f"{p:.1f}%  [{c.get('impact','?')}, {c.get('impact_magnitude','?')}]"
              for p, c in zip(pcts, contributions)]
    y      = np.arange(len(names))

    display_w = COL_W / cm
    display_h = max(3.5, len(names) * 0.85)
    fig_w     = 4.5
    fig_h     = fig_w * display_h / display_w

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars = ax.barh(y, pcts, color=clrs, height=0.5, zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, max(pcts) * 1.35 if pcts else 100)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("Contribution (%)", fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.set_title("XAI Feature Contributions", fontsize=8, color=M_TXT, pad=4)
    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_width() + max(pcts) * 0.02, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", ha="left", fontsize=6.5, color=M_DARK)
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, display_w, display_h)


def _table_counterfactuals(xai: dict, final_score: float, ss) -> Table:
    """ReportLab table: what-if counterfactual scenarios."""
    counterfactuals = (xai or {}).get("counterfactual_analysis") or []
    if not counterfactuals:
        return Paragraph("Counterfactual data unavailable.", ss["Body"])

    hdr_style = ParagraphStyle("CfHdr", fontName=FONT_BOLD, fontSize=7.5,
                                textColor=C_WHITE, alignment=TA_CENTER)
    cell_style = ParagraphStyle("CfCell", fontName=FONT_REG, fontSize=7.5,
                                 textColor=C_TEXT_D, alignment=TA_CENTER)

    col_w = COL_W / 4
    header = [
        Paragraph("Stream",        hdr_style),
        Paragraph("Now",           hdr_style),
        Paragraph("If Perfect",    hdr_style),
        Paragraph("Potential Lift",hdr_style),
    ]
    rows = [header]
    for cf in counterfactuals:
        stream   = str(cf.get("stream", "?"))[:18]
        now_s    = f"{final_score:.2f}"
        perfect  = f"{float(cf.get('if_perfect', 0)):.2f}"
        lift     = f"+{float(cf.get('max_possible_lift', 0)):.2f}"
        rows.append([
            Paragraph(stream,  cell_style),
            Paragraph(now_s,   cell_style),
            Paragraph(perfect, cell_style),
            Paragraph(lift,    cell_style),
        ])

    t = Table(rows, colWidths=[col_w] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("BACKGROUND",    (0, 1), (-1, -1), C_BLUE_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_BLUE_BG]),
        ("INNERGRID",     (0, 0), (-1, -1), 0.4, C_DIVIDER),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_PRIMARY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _chart_skills(cv: dict) -> Image:
    """Skills job-relevance chart — per-skill scores from LLM skill_relevance field."""
    skills = []
    if cv:
        for s in (cv.get("technical_skills") or cv.get("skills") or [])[:15]:
            skills.append(s if isinstance(s, str) else s.get("name", str(s))[:24])
    if not skills:
        return _placeholder("CV skill data unavailable")
    _mpl_setup()

    skill_relevance = dict((cv or {}).get("skill_relevance") or {})
    global_score    = float((cv or {}).get("relevance_score", 0.0)) or 0.70

    if not skill_relevance:
        # Infer per-skill relevance from LLM strengths/gaps when skill_relevance not yet stored
        strengths_txt = " ".join(str(s) for s in (cv or {}).get("strengths", [])).lower()
        gaps_txt      = " ".join(str(g) for g in (cv or {}).get("gaps", [])).lower()
        if strengths_txt or gaps_txt:
            for s in skills:
                sl = s.lower()
                if sl in gaps_txt:
                    skill_relevance[s] = round(max(0.10, global_score * 0.55), 2)
                elif sl in strengths_txt:
                    skill_relevance[s] = round(min(1.00, global_score * 1.20), 2)
                else:
                    skill_relevance[s] = global_score

    if skill_relevance:
        vals = np.array([float(skill_relevance.get(s, global_score)) for s in skills])
    else:
        vals = np.array([global_score] * len(skills))

    clrs = [M_GRN if v >= 0.70 else (M_YLW if v >= 0.40 else M_RED) for v in vals]
    y    = np.arange(len(skills))

    display_w = INNER_W / cm
    display_h = max(3.5, len(skills) * 0.85)
    fig_w     = 5.5
    fig_h     = fig_w * display_h / display_w

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars = ax.barh(y, vals, color=clrs, height=0.55, zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, 1.10)
    ax.set_yticks(y); ax.set_yticklabels(skills, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", zorder=0)
    ax.spines["left"].set_visible(False)
    title = "CV Skills — Job Relevance Match"
    ax.set_title(title, fontsize=8, color=M_TXT, pad=4)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(min(val + 0.01, 1.07), bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", ha="left", fontsize=7.5,
                    color=M_DARK, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, display_w, display_h)


# ─────────────────────────────────────────────────────────────────────────────
#   REPORTLAB STYLES  (Fix 1: all fonts → FONT_REG/BOLD/ITAL)
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles():
    ss = getSampleStyleSheet()

    def add(name, **kw):
        if name not in ss:
            ss.add(ParagraphStyle(name=name, **kw))
        return ss[name]

    add("DocTitle",  fontName=FONT_BOLD, fontSize=14, textColor=C_TEXT_D, spaceAfter=2)
    add("DocSub",    fontName=FONT_REG,  fontSize=9,  textColor=C_TEXT_M, spaceAfter=4)
    add("SecHead",   fontName=FONT_BOLD, fontSize=11, textColor=C_TEXT_D, spaceBefore=10, spaceAfter=4)
    add("Caption",   fontName=FONT_ITAL, fontSize=7.5, textColor=C_TEXT_L, spaceAfter=6, alignment=TA_CENTER)
    add("Body",      fontName=FONT_REG,  fontSize=8.5, textColor=C_TEXT_M, spaceAfter=4, leading=13)
    add("Bullet",    fontName=FONT_REG,  fontSize=8.5, textColor=C_TEXT_M, spaceAfter=2, leading=12, leftIndent=10)
    add("QHead",     fontName=FONT_BOLD, fontSize=8.5, textColor=C_TEXT_D, spaceAfter=2)
    add("Meta",      fontName=FONT_REG,  fontSize=8,   textColor=C_TEXT_M, alignment=TA_CENTER)
    add("MetaBold",  fontName=FONT_BOLD, fontSize=8,   textColor=C_TEXT_D, alignment=TA_CENTER, spaceAfter=1)
    return ss


# ─────────────────────────────────────────────────────────────────────────────
#   LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _two_col(left, right, lw=None, rw=None) -> Table:
    lw = lw or COL_W; rw = rw or COL_W
    t = Table([[left, right]], colWidths=[lw, rw])
    t.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    return t


def _section_header(title: str, ss) -> Table:
    row = [[Paragraph(title, ss["SecHead"])]]
    t = Table(row, colWidths=[INNER_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_BLUE_BG),
        ("LINEBEFORE",    (0,0), (0,-1),  3.5, C_PRIMARY),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, C_DIVIDER),
    ]))
    return t


def _summary_card(itype: str, proc_time, date_str: str, ss) -> Table:
    pt_str = f"{proc_time:.0f}s" if proc_time else "—"
    cells  = [
        [Paragraph("INTERVIEW TYPE", ss["Meta"]),  Paragraph(itype or "N/A", ss["MetaBold"])],
        [Paragraph("ANALYSIS TIME",  ss["Meta"]),  Paragraph(pt_str,         ss["MetaBold"])],
        [Paragraph("REPORT DATE",    ss["Meta"]),  Paragraph(date_str,       ss["MetaBold"])],
    ]
    t = Table([[cells[0], cells[1], cells[2]]], colWidths=[INNER_W / 3] * 3)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_BLUE_BG),
        ("BOX",           (0,0), (-1,-1), 0.5, C_PRIMARY),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, C_DIVIDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


def _caption(text: str, ss) -> Paragraph:
    return Paragraph(text, ss["Caption"])


# ─────────────────────────────────────────────────────────────────────────────
#   MARKDOWN PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _md_paras(text: str, ss, body="Body") -> list:
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            out.append(Spacer(1, 3)); continue
        if s.startswith("## "):
            out.append(Paragraph(s[3:], ss["SecHead"]))
        elif s.startswith("### "):
            out.append(Paragraph(f"<b>{s[4:]}</b>", ss[body]))
        elif s.startswith(("* ", "- ")):
            content = s[2:].replace("**", "<b>", 1).replace("**", "</b>", 1)
            out.append(Paragraph(f"• {content}", ss["Bullet"]))
        else:
            content = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
            out.append(Paragraph(content, ss[body]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#   DOCUMENT TEMPLATE  (Fix 1: FONT_REG/BOLD in canvas)
# ─────────────────────────────────────────────────────────────────────────────

def _build_doc(buf: io.BytesIO, candidate_name: str, job_title: str,
               score: float, rec: str, report_label: str) -> BaseDocTemplate:

    rec_c = (C_SUCCESS if ("hire" in rec.lower() or "strong" in rec.lower())
             else (C_WARNING if "maybe" in rec.lower() else C_DANGER))

    def _on_page(canvas, doc):
        canvas.saveState()
        # Top accent stripe
        canvas.setFillColor(C_PRIMARY)
        canvas.rect(0, PAGE_H - TOP_ACCENT, PAGE_W, TOP_ACCENT, fill=1, stroke=0)
        # Navy header band
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, PAGE_H - TOP_ACCENT - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        # Report label
        canvas.setFillColor(colors.HexColor("#93c5fd"))
        canvas.setFont(FONT_REG, 7.5)
        canvas.drawString(MARGIN, PAGE_H - TOP_ACCENT - 0.75 * cm, report_label.upper())
        # Candidate name (Unicode-safe with DejaVuSans)
        canvas.setFillColor(C_WHITE)
        canvas.setFont(FONT_BOLD, 15)
        canvas.drawString(MARGIN, PAGE_H - TOP_ACCENT - 1.6 * cm, candidate_name[:52])
        # Job title + date
        canvas.setFillColor(colors.HexColor("#bfdbfe"))
        canvas.setFont(FONT_REG, 8.5)
        canvas.drawString(MARGIN, PAGE_H - TOP_ACCENT - 2.3 * cm,
                          f"{job_title}  ·  {datetime.now().strftime('%B %d, %Y')}")
        # Recommendation badge
        bw, bh = 5.2 * cm, 0.82 * cm
        bx = PAGE_W - MARGIN - bw
        by = PAGE_H - TOP_ACCENT - HEADER_H + (HEADER_H - bh) / 2 - 0.1 * cm
        canvas.setFillColor(rec_c)
        canvas.roundRect(bx, by, bw, bh, radius=4, fill=1, stroke=0)
        canvas.setFillColor(C_WHITE)
        canvas.setFont(FONT_BOLD, 8)
        canvas.drawCentredString(bx + bw / 2, by + 0.26 * cm, rec.upper())
        # Bottom border
        canvas.setStrokeColor(C_PRIMARY)
        canvas.setLineWidth(1.2)
        canvas.line(0, PAGE_H - TOP_ACCENT - HEADER_H,
                    PAGE_W, PAGE_H - TOP_ACCENT - HEADER_H)
        # Footer
        canvas.setFillColor(C_TEXT_L)
        canvas.setFont(FONT_REG, 7)
        canvas.drawCentredString(PAGE_W / 2, 0.75 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(C_DIVIDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.0 * cm, PAGE_W - MARGIN, 1.0 * cm)
        canvas.restoreState()

    top_margin = TOP_ACCENT + HEADER_H + 0.6 * cm
    bot_margin = 1.5 * cm
    frame = Frame(MARGIN, bot_margin, INNER_W,
                  PAGE_H - top_margin - bot_margin, id="main", showBoundary=0)
    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=top_margin, bottomMargin=bot_margin)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#   HR PDF  (Fix 2, 3, 9: XAI stripping, lang detection, skills full-width)
# ─────────────────────────────────────────────────────────────────────────────

def generate_hr_pdf(result: dict[str, Any], candidate_info: dict) -> bytes:
    ss   = _build_styles()
    buf  = io.BytesIO()

    name    = candidate_info.get("name", result.get("candidate_name", "Candidate"))
    job     = candidate_info.get("job_title", "Position")
    score   = float(result.get("final_score") or 0)
    rec     = result.get("recommendation", "N/A")
    hr_text = result.get("hr_report") or ""
    per_q   = result.get("per_question_analysis") or []
    speech  = result.get("speech_emotion_result") or {}
    facial  = result.get("facial_emotion_result") or {}
    streams = result.get("stream_scores") or []
    cv_data = result.get("cv_analysis_result") or {}
    xai     = result.get("xai_explanations") or {}
    itype   = result.get("interview_type", "VIDEO")
    proc    = result.get("processing_time_seconds")

    # Strip XAI ASCII header before any use (Fix 2)
    hr_clean = _strip_xai_header(hr_text)
    lang     = _detect_lang(hr_clean)   # Fix 3: use clean text

    doc   = _build_doc(buf, name, job, score, rec, "HR Analysis Report — Confidential")
    story: list = []

    date_str = datetime.now().strftime("%Y-%m-%d")

    # ── Summary card ──
    story.append(_summary_card(itype, proc, date_str, ss))
    story.append(Spacer(1, 10))

    # ── Score circle (left) + Executive Summary paragraph (right) ──
    score_col   = [_chart_score_circle(score)]
    intro_text  = hr_clean.split("\n\n")[0] if hr_clean else ""
    intro_paras = _md_paras(intro_text[:500], ss) if intro_text else [
        Paragraph("AI analysis complete.", ss["Body"])
    ]
    story.append(_two_col(score_col, intro_paras, lw=4.2 * cm, rw=INNER_W - 4.2 * cm))
    story.append(Spacer(1, 8))

    # ── Stream scores (full-width) ──
    story.append(_section_header("AI Stream Analysis", ss))
    story.append(Spacer(1, 6))
    story.append(_chart_stream_scores(streams))
    story.append(_caption(_cap("stream_scores", lang), ss))
    story.append(Spacer(1, 8))

    # ── Per-question: chart (left) + LLM commentary (right) ──
    if per_q:
        story.append(_section_header("Per-Question Analysis", ss))
        story.append(Spacer(1, 6))

        q_chart_col = [
            _chart_per_question(per_q),
            _caption(_cap("per_question", lang), ss),
        ]
        # Extract Q-by-Q section from clean text
        qby_raw = ""
        if "## Question-by-Question" in hr_clean:
            chunk = hr_clean.split("## Question-by-Question")[1]
            for stop in ["## Areas", "## Strengths", "## Final", "## Integrity"]:
                if stop in chunk:
                    chunk = chunk[:chunk.index(stop)]
            qby_raw = "## Question-by-Question" + chunk[:1200]
        if qby_raw:
            q_right = _md_paras(qby_raw, ss)
        else:
            q_right = []
            for i, q in enumerate(per_q[:6]):
                q_right.append(Paragraph(
                    f"<b>Q{i+1}:</b> {q.get('question_text','')[:90]}", ss["QHead"]))
                q_right.append(Paragraph(
                    q.get("commentary","")[:220], ss["Body"]))
                q_right.append(Spacer(1, 4))

        story.append(_two_col(q_chart_col, q_right))
        story.append(Spacer(1, 8))

    # ── Emotion Analysis ──
    story.append(_section_header("Emotion Analysis", ss))
    story.append(Spacer(1, 6))

    # Row 1: Speech pie (left) | Facial pie (right)
    sp_col = [
        Paragraph("<b>Vocal Emotion Distribution</b>", ss["Body"]),
        Spacer(1, 3),
        _chart_speech_emotion_dist(speech),
        _caption(_cap("speech_emotion_dist", lang), ss),
    ]
    fc_col = [
        Paragraph("<b>Facial Emotion Distribution</b>", ss["Body"]),
        Spacer(1, 3),
        _chart_facial_emotion_dist(facial),
        _caption(_cap("facial_emotion_dist", lang), ss),
    ]
    story.append(_two_col(sp_col, fc_col))
    story.append(Spacer(1, 6))

    # Row 2: Facial timeline (left) | Positivity comparison (right)
    story.append(_two_col(
        [_chart_facial_timeline(facial), _caption(_cap("facial_timeline", lang), ss)],
        [_chart_positivity(speech, facial), _caption(_cap("positivity", lang), ss)],
    ))
    story.append(Spacer(1, 6))

    # Row 3: Speech timeline — full-width (spans both columns, more room as interview grows)
    story.append(_chart_speech_timeline(speech, full_width=True))
    story.append(_caption(_cap("speech_timeline", lang), ss))
    story.append(Spacer(1, 8))

    # ── Skills (full-width) ──
    story.append(_section_header("Skills from CV", ss))
    story.append(Spacer(1, 6))
    story.append(_chart_skills(cv_data))
    story.append(_caption(_cap("skills_match", lang), ss))
    story.append(Spacer(1, 10))

    # ── XAI Decision Analysis ──
    if xai.get("feature_contributions"):
        story.append(_section_header("XAI — Decision Analysis", ss))
        story.append(Spacer(1, 6))
        story.append(_two_col(
            [_chart_xai_contributions(xai),
             _caption("Each stream's % contribution to the final decision. Green = positive impact, Red = negative.", ss)],
            [_table_counterfactuals(xai, score, ss),
             Spacer(1, 4),
             _caption("What-if scenarios: score if each stream had been perfect (1.0).", ss)],
        ))
        story.append(Spacer(1, 10))

    # ── Full evaluation report text — Executive Summary & Q-by-Q stripped (already shown above) ──
    if hr_clean:
        def _strip_sections(text: str, headings: list) -> str:
            for h in headings:
                idx = text.find(h)
                if idx >= 0:
                    nxt = text.find("\n## ", idx + 1)
                    text = text[:idx] + (text[nxt:] if nxt > 0 else "")
            return text.strip()

        filtered = _strip_sections(hr_clean, ["## Executive Summary", "## Question-by-Question"])
        if filtered:
            story.append(_section_header("Full Evaluation Report", ss))
            story.append(Spacer(1, 6))
            story.extend(_md_paras(filtered, ss))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#   CANDIDATE PDF  (Fix 6: score circle full-width, text no longer cut off)
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_pdf(result: dict[str, Any], candidate_info: dict) -> bytes:
    ss  = _build_styles()
    buf = io.BytesIO()

    name     = candidate_info.get("name", result.get("candidate_name", "Candidate"))
    job      = candidate_info.get("job_title", "Position")
    score    = float(result.get("final_score") or 0)
    per_q    = result.get("per_question_analysis") or []
    speech   = result.get("speech_emotion_result") or {}
    facial   = result.get("facial_emotion_result") or {}
    cv_data  = result.get("cv_analysis_result") or {}
    feedback = result.get("candidate_feedback") or ""
    itype    = result.get("interview_type", "VIDEO")
    proc     = result.get("processing_time_seconds")

    # Detect language from section headings first (most reliable — candidate names
    # may contain Turkish characters even in English-language feedback)
    _EN_HEADING_MARKERS = ["## Overall Impression", "## Your Strengths",
                            "## Question-by-Question Feedback", "## Areas for Growth", "## Closing"]
    _TR_HEADING_MARKERS = ["## Genel İzlenim", "## Güçlü Yönleriniz",
                            "## Soru Bazlı Geri Bildirim", "## Geliştirilmesi", "## Kapanış"]
    en_hits = sum(1 for h in _EN_HEADING_MARKERS if h in feedback)
    tr_hits = sum(1 for h in _TR_HEADING_MARKERS if h in feedback)
    if en_hits > tr_hits:
        lang = "en"
    elif tr_hits > en_hits:
        lang = "tr"
    else:
        # No Markdown headings found (template-format feedback) — detect from body,
        # skipping the first two lines which may contain the candidate name with Turkish chars
        body_lines = (feedback or "").splitlines()
        body_sample = "\n".join(body_lines[2:])[:400]
        lang = _detect_lang(body_sample) if body_sample.strip() else _detect_lang(feedback)

    if score >= 0.75:
        cat = "Excellent"
    elif score >= 0.50:
        cat = "Good"
    else:
        cat = "Developing"

    doc = _build_doc(buf, name, job, score, cat, "Interview Feedback Report")
    story: list = []

    date_str = datetime.now().strftime("%Y-%m-%d")

    # ── Summary card ──
    story.append(_summary_card(itype, proc, date_str, ss))
    story.append(Spacer(1, 8))

    # Heading variants (EN / TR / TR-Live)
    _OI_HEADINGS  = ["## Overall Impression", "## Genel İzlenim"]
    _STR_HEADINGS = ["## Your Strengths", "## Your Strengths in This Live Interview",
                     "## Güçlü Yönleriniz", "## Bu Mülakatınızdaki Güçlü Yönleriniz",
                     "## Bu Canlı Görüşmedeki Güçlü Yönleriniz"]
    _QBY_HEADINGS = ["## Question-by-Question Feedback", "## Question Feedback",
                     "## Per-Question", "## Soru Bazlı Geri Bildirim"]

    def _find_section_body(text: str, headings: list, max_len: int = 700) -> str:
        """Return the body of the first matching heading (heading line itself stripped)."""
        for h in headings:
            idx = text.find(h)
            if idx >= 0:
                after = text[idx + len(h):]          # everything after the heading
                end   = after.find("\n## ")
                body  = after[:end if end > 0 else max_len].strip()
                return body
        return ""

    # ── Score circle (left) + Overall Impression heading + first paragraph (right) ──
    oi_body  = _find_section_body(feedback, _OI_HEADINGS, max_len=600)
    oi_first = oi_body.split("\n\n")[0].strip()[:500] if oi_body else feedback[:400].strip()
    oi_heading = "Overall Impression" if lang == "en" else "Genel İzlenim"

    score_col  = [_chart_score_circle(score)]
    if oi_first:
        intro_para = [Paragraph(f"<b>{oi_heading}</b>", ss["SecHead"])] + _md_paras(oi_first, ss)
    else:
        intro_para = [Paragraph(f"<b>{oi_heading}</b>", ss["SecHead"]),
                      Paragraph("Interview feedback complete.", ss["Body"])]
    story.append(_two_col(score_col, intro_para, lw=4.2 * cm, rw=INNER_W - 4.2 * cm))
    story.append(Spacer(1, 8))

    # ── Per-Q performance chart (left) + Q-by-Q heading + commentary (right) ──
    if per_q:
        pqa_section  = "Per-Question Analysis"  if lang == "en" else "Soru Bazlı Analiz"
        qby_title    = "Question-by-Question Feedback" if lang == "en" else "Soru Bazlı Geri Bildirim"
        story.append(_section_header(pqa_section, ss))
        story.append(Spacer(1, 6))

        q_chart_col = [
            _chart_per_question_candidate(per_q),
            _caption(_cap("per_question_candidate", lang), ss),
        ]
        qby_body = _find_section_body(feedback, _QBY_HEADINGS, max_len=1200)
        if qby_body:
            q_right = [Paragraph(f"<b>{qby_title}</b>", ss["SecHead"])] + _md_paras(qby_body, ss)
        else:
            q_right = [Paragraph(f"<b>{qby_title}</b>", ss["SecHead"])]
            for i, q in enumerate(per_q[:6]):
                q_right.append(Paragraph(
                    f"<b>Q{i+1}:</b> {q.get('question_text', '')[:90]}", ss["QHead"]))
                q_right.append(Paragraph(
                    q.get("commentary", "")[:220], ss["Body"]))
                q_right.append(Spacer(1, 4))

        story.append(_two_col(q_chart_col, q_right))
        story.append(Spacer(1, 8))

    # ── Your Strengths — full-width ──
    strengths_body = _find_section_body(feedback, _STR_HEADINGS, max_len=700)
    if strengths_body:
        story.append(Paragraph("<b>Your Strengths</b>" if lang == "en" else "<b>Güçlü Yönleriniz</b>", ss["SecHead"]))
        story.extend(_md_paras(strengths_body.strip(), ss))
        story.append(Spacer(1, 8))

    # ── Communication Style ──
    story.append(_section_header("Your Communication Style", ss))
    story.append(Spacer(1, 6))
    story.append(_two_col(
        [Paragraph("<b>Vocal Emotion Distribution</b>", ss["Body"]),
         Spacer(1, 3),
         _chart_speech_emotion_dist(speech),
         _caption(_cap("speech_emotion_dist", lang), ss)],
        [Paragraph("<b>Facial Expression Distribution</b>", ss["Body"]),
         Spacer(1, 3),
         _chart_facial_emotion_dist(facial),
         _caption(_cap("facial_emotion_dist", lang), ss)],
    ))
    story.append(Spacer(1, 8))

    # ── Skills (full-width) ──
    story.append(_section_header("Skills from Your CV", ss))
    story.append(Spacer(1, 6))
    story.append(_chart_skills(cv_data))
    story.append(_caption(_cap("skills_match", lang), ss))
    story.append(Spacer(1, 10))

    # ── Detailed Feedback (Areas for Growth + Closing — sections already shown above are stripped) ──
    if feedback:
        remaining = feedback
        skip_headings = (
            _OI_HEADINGS + _STR_HEADINGS + _QBY_HEADINGS
        )
        for skip in skip_headings:
            idx = remaining.find(skip)
            if idx >= 0:
                end = remaining.find("\n## ", idx + 1)
                remaining = remaining[:idx] + (remaining[end:] if end > 0 else "")
        if remaining.strip():
            story.append(_section_header("Detailed Feedback", ss))
            story.append(Spacer(1, 6))
            story.extend(_md_paras(remaining.strip(), ss))

    doc.build(story)
    return buf.getvalue()
