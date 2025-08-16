"""
===========================================================================
                        PEOPLE COUNTING SYSTEM - TODOs
===========================================================================

1. Code Review & Refactoring
   - Review the entire modular codebase:
     config.py, video_stream.py, detector.py, tracker_state.py,
     counting.py, vizualization.py, logger.py, main.py, utils.py.
   - Ensure clean separation of concerns and consistent naming.
   - Apply PEP8 formatting and best practices.
   - Handle exceptions (e.g., missing RTSP feed, failed model load).

2. Multi-Line Entry/Exit Detection
   - Add support for 2 additional lines for entry/exit detection.
   - Make line positions configurable in config.py.
   - Ensure accurate counting when a person crosses any line.
   - Implement thresholds and dead zones for line crossing detection.

3. Pseudocode for Line Crossing Logic
   - Detect upward and downward movement across each line.
   - Respect COUNT_COOLDOWN between successive counts.
   - Remove stale IDs based on STALE_TIME.
   - Track which line was crossed and update counts accordingly.

4. Logging & Debugging Enhancements
   - Improve debug logs to indicate the line that triggered entry/exit.
   - Record crossing timestamps and trajectory information for analysis.


===========================================================================
"""
