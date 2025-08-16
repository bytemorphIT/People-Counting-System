
from datetime import datetime
from config import MIDDLE_LINE_Y, COUNT_COOLDOWN

class Counter:
    def __init__(self):
        self.entry_count = 0
        self.exit_count = 0

    def check_crossing(self, tid, positions, state_info):
        if len(positions) < 2:
            return

        last_cy = positions[-2]
        current_cy = positions[-1]
        now = datetime.now()
        can_count = now - state_info['last_count_time'] > COUNT_COOLDOWN

        if last_cy > MIDDLE_LINE_Y >= current_cy and can_count:
            self.entry_count += 1
            state_info['state'] = "inside"
            state_info['last_count_time'] = now
            return "entry"
        elif last_cy < MIDDLE_LINE_Y <= current_cy and can_count:
            self.exit_count += 1
            state_info['state'] = "outside"
            state_info['last_count_time'] = now
            return "exit"
        return None
