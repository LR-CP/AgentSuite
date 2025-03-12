class AgentManager:
    """
    A simple registry to add, delete, and retrieve agents.
    """
    def __init__(self):
        self.agents = {}

    def add_agent(self, agent_id: str, agent):
        self.agents[agent_id] = agent

    def delete_agent(self, agent_id: str):
        if agent_id in self.agents:
            del self.agents[agent_id]

    def get_agent(self, agent_id: str):
        return self.agents.get(agent_id)

    def list_agents(self):
        return list(self.agents.keys())