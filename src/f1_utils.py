"""
Shared utilities used across the feature-building scripts.

This exists specifically to avoid the DNF-classification bug happening a
third time: it was previously duplicated in build_circuit_table.py and
build_features.py separately, and each copy got fixed for one dataset
shape (2024-only) without the other being updated — twice. Now there's
exactly one place this logic lives.
"""

# A driver is a CLASSIFIED FINISHER (not a DNF) if status is "Finished",
# "Lapped", or one of the "+N Lap(s)" variants. Confirmed against the
# full 2018-2025 status value_counts — these are the only status strings
# that represent a car that completed (or was classified as completing)
# the race, just possibly behind on laps. Every other status (Retired,
# Collision, Engine, Disqualified, Did not start, Withdrew, Illness,
# etc.) is a genuine non-finish.
def is_dnf(status: str) -> bool:
    if status == "Finished":
        return False
    if status == "Lapped":
        return False
    if status.startswith("+") and "Lap" in status:
        return False
    return True


# Circuits run on closed public roads / temporary street layouts.
# circuit_id values match Jolpica's naming.
STREET_CIRCUITS = {
    "monaco",
    "baku",
    "marina_bay",
    "jeddah",
    "miami",
    "vegas",
}
