import random
from typing import List, Tuple

from server.contract_validation_environment import ContractValidationEnvironment
from models import ContractValidationAction


class ContractRLAgent:
    def __init__(self, clauses: List[int], risk_types: List[str]):
        self.clauses = clauses
        self.risk_types = risk_types
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.01

    def _get_state(self, obs) -> frozenset:
        return frozenset(obs.flagged_risks.items())

    def _get_possible_actions(self) -> List[Tuple[int, str]]:
        actions = []
        for c in self.clauses:
            for r in self.risk_types:
                actions.append((c, r))
        actions.append((0, "submit"))
        return actions

    def choose_action(self, state: frozenset) -> Tuple[int, str]:
        possible_actions = self._get_possible_actions()
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in possible_actions}

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state: frozenset, action: Tuple[int, str], reward: float, next_state: frozenset, done: bool):
        possible_actions = self._get_possible_actions()
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in possible_actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in possible_actions}

        # BUG FIX 2: If the episode is done, there is no future reward!
        best_next_q = 0.0 if done else max(self.q_table[next_state].values())

        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * \
            (reward + self.gamma * best_next_q - current_q)

    def train(self, env: ContractValidationEnvironment, episodes: int = 300):
        print(f"Starting training for {episodes} episodes...")
        for episode in range(episodes):
            obs = env.reset(task_level="easy")
            state = self._get_state(obs)
            done = obs.done

            while not done and obs.step_count < 15:
                action_tuple = self.choose_action(state)
                clause_id, risk_type = action_tuple

                is_submit = (risk_type == "submit")

                env_action = ContractValidationAction(
                    clause_id=clause_id,
                    risk_type="none" if is_submit else risk_type,
                    submit_final=is_submit,
                    explanation="RL Agent exploration"
                )

                next_obs = env.step(env_action)
                next_state = self._get_state(next_obs)

                # BUG FIX 1: The environment now returns the exact step reward directly!
                step_reward = next_obs.reward

                # Pass the 'done' flag so the table knows when to stop looking forward
                self.learn(state, action_tuple, step_reward,
                           next_state, next_obs.done)

                state = next_state
                done = next_obs.done

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

        print("Training complete!")

    def test(self, env: ContractValidationEnvironment):
        print("\n--- Testing Trained Agent ---")
        self.epsilon = 0.0

        obs = env.reset(task_level="easy")
        state = self._get_state(obs)
        done = False

        while not done and obs.step_count < 15:
            action_tuple = self.choose_action(state)
            clause_id, risk_type = action_tuple

            is_submit = (risk_type == "submit")

            env_action = ContractValidationAction(
                clause_id=clause_id,
                risk_type="none" if is_submit else risk_type,
                submit_final=is_submit,
                explanation="Agent's learned optimal choice"
            )

            print(
                f"Agent Action -> Clause: {clause_id} | Flagged Risk: {risk_type} | Submitted: {is_submit}")
            obs = env.step(env_action)
            state = self._get_state(obs)
            done = obs.done

        print(f"\nFinal Environment Reward: {obs.reward}")
        print(f"Final Flagged Risks: {obs.flagged_risks}")


if __name__ == "__main__":
    env = ContractValidationEnvironment()
    valid_clauses = [1]
    potential_risks = ["liability", "payment", "none"]

    agent = ContractRLAgent(clauses=valid_clauses, risk_types=potential_risks)
    agent.train(env, episodes=300)
    agent.test(env)
