from locust import LoadTestShape

from load_tests.locustfile import RAGUser


class RAGStepLoad(LoadTestShape):
    stages = [
        {"duration": 120, "users": 1, "spawn_rate": 1},
        {"duration": 240, "users": 5, "spawn_rate": 1},
        {"duration": 360, "users": 10, "spawn_rate": 2},
        {"duration": 480, "users": 25, "spawn_rate": 3},
        {"duration": 600, "users": 50, "spawn_rate": 5},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
