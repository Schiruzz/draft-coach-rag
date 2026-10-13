"""End-of-step check: both databases answer. Run: python scripts/check_services.py"""
from draft_coach.health import mongo_version, qdrant_version

for name, check in [("mongodb", mongo_version), ("qdrant", qdrant_version)]:
    try:
        print(f"{name:<8} ok    version {check()}")
    except Exception as error:  # report any failure instead of crashing on the first one
        print(f"{name:<8} FAIL  {type(error).__name__}: {error}")