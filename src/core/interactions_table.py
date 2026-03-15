from data_config import InteractionWeightsMapper
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd


class InteractionMatrix:
    def __init__(self):
        self.matrix = defaultdict(lambda: defaultdict(float))
        self.event_history = []
        self.weights_mapper = InteractionWeightsMapper()
        self.time_window = timedelta(days=1)
        self.decline_factor = 0.95
        self.last_update = datetime.utcnow()

    def _get_weight(self, action: str) -> float:
        return self.weights_mapper.dict_to_map.get(action, 0)

    def add_event(self, event: dict):
        event_time = datetime.strptime(
            event['occurred_at'], "%Y-%m-%dT%H:%M:%SZ")
        self.event_history.append((event, event_time))

    def _apply_decay(self):
        now = datetime.utcnow()
        time_diff = now - self.last_update
        if time_diff >= self.time_window:
            for user1 in self.matrix:
                for user2 in self.matrix[user1]:
                    self.matrix[user1][user2] *= self.decline_factor
            self.last_update = now

    def _process_event(self, event: dict):
        actor = event['actor_user_id']
        target = event['target_user_id']
        action = event['payload']['action'] if 'action' in event['payload'] else None
        score = event['payload'].get('score', 0)

        if action:
            weight = self._get_weight(action)
            self.matrix[actor][target] += weight
            self.matrix[target][actor] += weight

        if score:
            self.matrix[actor][target] += score
            self.matrix[target][actor] += score

    def update_matrix(self):
        self._apply_decay()
        for event, event_time in self.event_history:
            self._process_event(event)
        self.event_history.clear()

    def get_matrix(self):
        return dict(self.matrix)

    def get_popularity_scores(self, top_k=10):
        """
        Возвращает DataFrame с топ-k пользователей по сумме их взаимодействий.
        """
        scores = defaultdict(float)
        for user1 in self.matrix:
            for user2 in self.matrix[user1]:
                scores[user1] += self.matrix[user1][user2]
                scores[user2] += self.matrix[user2][user1]
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)
        top_users = sorted_scores[:top_k]
        top_users_df = pd.DataFrame(
            top_users, columns=["user_id", "popularity_score"])
        return top_users_df
