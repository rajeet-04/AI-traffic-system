"""Decision engine (rule-based FSM) template.

Consumes a list of tracks and their history to produce simple advisories:
STOP / GO / CAUTION. Uses a majority-vote over recent labels per track.
"""
from typing import List

class DecisionEngine:
    def __init__(self, window: int = 5, red_threshold: float = 0.6):
        self.window = window
        self.red_threshold = red_threshold

    def decide(self, tracks) -> dict:
        """tracks: iterable of Track objects (from tracker.Track)

        Returns a dict summary with 'advisory' and details per track.
        """
        adv = 'NO_DATA'
        track_summaries = []
        red_votes = 0
        green_votes = 0
        total = 0
        for t in tracks:
            hist = t.history[-self.window:]
            total += 1
            red = sum(1 for s in hist if s.lower() == 'red')
            green = sum(1 for s in hist if s.lower() == 'green')
            if red > green:
                red_votes += 1
            elif green > red:
                green_votes += 1
            track_summaries.append({'id': t.id, 'label': t.label, 'history': hist})

        if total == 0:
            adv = 'NO_DETECTIONS'
        else:
            if red_votes / max(1, total) >= self.red_threshold:
                adv = 'STOP'
            elif green_votes >= red_votes:
                adv = 'GO'
            else:
                adv = 'CAUTION'

        return {'advisory': adv, 'total_tracks': total, 'red_votes': red_votes, 'green_votes': green_votes, 'tracks': track_summaries}

if __name__ == '__main__':
    print('DecisionEngine template. Tune thresholds and window size per your needs.')
