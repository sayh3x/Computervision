import time


class Cooldown:
    def __init__(self, seconds):
        self.seconds = seconds
        self.last_time = 0


def ready(self):
    now = time.time()
    if now - self.last_time > self.seconds:
        self.last_time = now
        return True
    return False
