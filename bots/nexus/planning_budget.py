"""Cooperative planning slices, shared by all expensive searches of one unit."""


class PlanningBudget:
    def __init__(self):
        self.controller = None
        self.blocks = 0
        self.pending = False

    def begin_turn(self, controller):
        self.controller = controller
        self.blocks = 0
        self.pending = False

    def checkpoint(self):
        self.blocks += 1
        if self.blocks > 12 or (
            self.controller is not None
            and self.controller.get_cpu_time_elapsed() >= 1200
        ):
            self.pending = True
            raise RuntimeError('planning slice complete')


class SearchMemory:
    def __init__(self, budget):
        self.budget = budget
        self.jobs = {}

    def run(self, key, make_generator, version=None):
        job = self.jobs.get(key)
        # New terrain must invalidate a cached failure, but must not restart
        # an unfinished search on every symmetry-backfill turn.
        if job is not None and job[1] and job[3] != version:
            job = None
        if job is None:
            if len(self.jobs) >= 32:
                self.jobs.pop(next(iter(self.jobs)))
            job = [make_generator(), False, None, version]
            self.jobs[key] = job
        if job[1]:
            return job[2]
        while True:
            self.budget.checkpoint()
            # An exception escaping a generator closes it. Keep it out of the
            # cache while it runs, so an interrupted step is retried with a new
            # generator instead of being mistaken for StopIteration(None).
            # Checkpoint suspensions happen before removal and preserve state.
            self.jobs.pop(key, None)
            try:
                next(job[0])
            except StopIteration as completed:
                job[:] = [None, True, completed.value, job[3]]
                self.jobs[key] = job
                return completed.value
            self.jobs[key] = job
