import logging
import time

import numpy as np


class _Profiler:
    def __init__(self):
        self.ongoing_events = dict()
        self.completed_events = dict()
        self.max_events = int(1e7)  # max number instances to store for each event

    def start_event(self, event_name: str):
        assert event_name not in self.ongoing_events, f"event {event_name} already started!"
        self.ongoing_events[event_name] = time.time()

    def end_event(self, event_name: str):
        assert event_name in self.ongoing_events, f"event {event_name} not started!"
        time_taken = round(time.time() - self.ongoing_events[event_name], 2)
        if event_name not in self.completed_events:
            self.completed_events[event_name] = []
        if len(self.completed_events[event_name]) >= self.max_events:  # remove old events
            self.completed_events[event_name].pop(0)
        self.completed_events[event_name].append(time_taken)
        del self.ongoing_events[event_name]

    def get_stats(self):
        stats = dict()
        for event_name, event_times in self.completed_events.items():
            stats[event_name] = {
                "mean": round(np.mean(event_times), 2),
                "std": round(np.std(event_times), 2),
                "count": len(event_times),
                "min": round(np.min(event_times), 2),
                "max": round(np.max(event_times), 2),
                "total": round(np.sum(event_times), 2),
            }
        return stats

    def format_time(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        s, m, h, d = int(s), int(m), int(h), int(d)
        if d > 0:
            return f"{d:d}d {h:02d}h {m:02d}m"
        elif h > 0:
            return f"{h:d}h {m:02d}m"
        elif m > 0:
            return f"{m:d}m {s:02d}s"
        else:
            return f"{seconds:.2f}s"

    def print_stats(self, format_time: bool = True):
        logging.info("-" * 60)
        for event_name, event_stats in self.get_stats().items():
            count = event_stats["count"]
            mean_time = (
                self.format_time(event_stats["mean"])
                if format_time
                else f"{event_stats['mean']:.2f}s"
            )
            std_time = (
                self.format_time(event_stats["std"])
                if format_time
                else f"{event_stats['std']:.2f}s"
            )
            total_time = (
                self.format_time(event_stats["total"])
                if format_time
                else f"{event_stats['total']:.2f}s"
            )
            logging.info(
                f"profiler {event_name} ({count} events): {mean_time} ± {std_time}, total: {total_time}"
            )
        logging.info("-" * 60)


### initialize a global profiler
profiler = _Profiler()


def get_complete_function_name(func):
    if hasattr(func, "__qualname__"):
        return f"{func.__qualname__}"
    else:
        return f"{func.__name__}"


def profiler_decorator(func=None, *, event_name: str = ""):
    def decorator(func):

        # if event name not passed then use function name
        _event_name = event_name if len(event_name) > 0 else get_complete_function_name(func)

        def wrapper(*args, **kwargs):
            profiler.start_event(_event_name)
            result = func(*args, **kwargs)
            profiler.end_event(_event_name)
            return result

        return wrapper

    if func is None:  # Decorator was called with parameters
        return decorator
    return decorator(func)  # Decorator was called without parameters


if __name__ == "__main__":
    from utils.common import setup_logging

    setup_logging()

    event_name1 = "test1.datagen"
    event_name2 = "test2.model_training"
    event_name3 = "test3.inference"

    profiler.start_event(event_name1)
    profiler.end_event(event_name1)
    profiler.start_event(event_name2)
    profiler.end_event(event_name2)
    profiler.start_event(event_name3)
    profiler.end_event(event_name3)

    profiler.completed_events[event_name1] = [1000] * 7
    profiler.completed_events[event_name2] = [2000] * 7
    profiler.completed_events[event_name3] = [80000] * 7

    profiler.print_stats(format_time=False)
    profiler.print_stats(format_time=True)
