from __future__ import annotations


def _describe_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def build_fallback_notice(primary_exc: Exception, direct_exc: Exception | None = None) -> dict[str, str]:
    mode_reason = f"fallback:{primary_exc.__class__.__name__}"
    warning = f"Fell back to local evaluation because remote evaluation failed: {_describe_exception(primary_exc)}"
    if direct_exc is not None:
        mode_reason += f";direct_fallback:{direct_exc.__class__.__name__}"
        warning += f"; direct answering also fell back locally: {_describe_exception(direct_exc)}"
    return {
        "mode_reason": mode_reason,
        "warning": warning,
    }
