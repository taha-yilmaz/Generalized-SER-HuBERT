"""Aggregates frontend integrity signals + FER face-absence log into a score."""
class IntegrityAnalyzer:
    @staticmethod
    def analyze(frontend_signals: dict, face_absence_seconds: float = 0.0,
                multi_face_frames: int = 0) -> dict:
        fs = frontend_signals or {}
        tab_switches = fs.get("tab_switches", []) or []
        focus_losses = fs.get("focus_losses", []) or []
        copy_paste = int(fs.get("copy_paste_attempts", 0) or 0)
        tab_total = sum(float(e.get("duration", 0) or 0) for e in tab_switches)

        n_tabs = len(tab_switches)
        if n_tabs >= 5 or face_absence_seconds >= 60 or multi_face_frames > 0:
            level = "SIGNIFICANT_ANOMALIES"
        elif n_tabs >= 2 or face_absence_seconds >= 30 or copy_paste > 0:
            level = "MINOR_CONCERNS"
        else:
            level = "CLEAN"

        return {
            "raw": {
                "tab_switches_count": n_tabs,
                "tab_switches_total_seconds": round(tab_total, 1),
                "focus_losses_count": len(focus_losses),
                "copy_paste_attempts": copy_paste,
                "face_absence_seconds": round(face_absence_seconds, 1),
                "multi_face_frames": multi_face_frames,
            },
            "score": level,
            "summary": IntegrityAnalyzer._fmt(n_tabs, tab_total, len(focus_losses),
                                              copy_paste, face_absence_seconds,
                                              multi_face_frames, level),
        }

    @staticmethod
    def _fmt(tabs, tab_sec, focus, cp, face_sec, mf, level):
        return (
            f"Tab switches: {tabs} ({tab_sec:.1f}s away) | "
            f"Focus losses: {focus} | Copy-paste blocked: {cp} | "
            f"Face absence: {face_sec:.1f}s | Multi-face frames: {mf} | "
            f"Overall: {level}"
        )
