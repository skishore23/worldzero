"""Minimal WorldZero Agent SDK example.

This demonstrates the interface only. It deliberately contains no discovery
strategy and is not expected to score well.
"""


class CustomAgent:
    def reset(self, context):
        self.last_result = None

    def act(self, observation):
        return {
            "action": {"type": "WAIT", "duration": 4.0},
            "finding": {"status": "insufficient_evidence"},
        }

    def observe_result(self, result):
        self.last_result = result

    def close(self):
        pass


def create_agent():
    return CustomAgent()
