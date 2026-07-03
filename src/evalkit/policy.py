"""Frozen evaluation policy knobs (SPEC_M2 + gate-documented amendments)."""

THIN_CELL_THRESHOLD = 300  # labeled val matches


def t2_leaderboard_calibration(n_labeled_val_matches: int) -> str:
    """Amendment #1: thin cells (< 300 labeled val matches) use Platt on the
    leaderboard; isotonic is still fitted and reported alongside."""
    return "platt" if n_labeled_val_matches < THIN_CELL_THRESHOLD else "isotonic"
