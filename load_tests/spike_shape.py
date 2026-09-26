from locust import LoadTestShape

from load_tests.locustfile import RAGUser


class RAGSpikeLoad(LoadTestShape):
    stages = [
        {"duration": 60, "users": 2, "spawn_rate": 1},
        {"duration": 120, "users": 20, "spawn_rate": 20},
        {"duration": 240, "users": 2, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
