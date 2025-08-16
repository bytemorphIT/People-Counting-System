
from collections import deque
from datetime import datetime
from config import MAX_POSITION_HISTORY

class TrackerState:
    def __init__(self):
        self.person_states = {}

    def init_person(self, tid, cy):
        self.person_states[tid] = {
            'last_positions': deque([cy], maxlen=MAX_POSITION_HISTORY),
            'last_count_time': datetime.min,
            'last_seen': datetime.now(),
            'state': 'unknown'
        }

    def update_position(self, tid, cy):
        state_info = self.person_states[tid]
        state_info['last_positions'].append(cy)
        state_info['last_seen'] = datetime.now()

    def remove_stale(self, stale_seconds):
        current_time = datetime.now()
        stale_ids = [tid for tid, s in self.person_states.items()
                     if (current_time - s['last_seen']).total_seconds() > stale_seconds]
        for tid in stale_ids:
            self.person_states.pop(tid)
