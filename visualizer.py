
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_rewards(rewards, agent_names):
        plt.figure(figsize=(12, 6))
        for agent, rewards_list in rewards.items():
            plt.plot(rewards_list, label=agent)
        plt.title("Rewards Over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_cumulative_returns(returns, agent_names):
        plt.figure(figsize=(12, 6))
        for agent, returns_list in returns.items():
            plt.plot(returns_list, label=agent)
        plt.title("Cumulative Returns Over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.show()

    @staticmethod
    def compare_metrics(metrics):
        # Compare agents' metrics in a bar chart
        plt.figure(figsize=(12, 6))
        agents = list(metrics.keys())
        values = list(metrics.values())
        plt.bar(agents, values)
        plt.title("Agent Comparison")
        plt.xlabel("Agents")
        plt.ylabel("Metrics")
        plt.show()
