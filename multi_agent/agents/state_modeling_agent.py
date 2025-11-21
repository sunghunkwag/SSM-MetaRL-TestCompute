"""State Modeling Agent - Specializes in temporal dynamics capture."""

from typing import Dict, Any, Optional
from crewai import Agent, Task
from ..tools.ssm_tool import SSMTool

class StateModelingAgent:
    """Agent specialized in state space modeling and temporal dynamics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ssm_tool = SSMTool()

        self.agent = Agent(
            role="State Space Model Architect",
            goal="Design efficient SSM architectures for high-dimensional temporal data.",
            backstory="""You are a senior researcher specializing in State Space Models (SSMs).
            You understand the trade-off between state dimension size and computational efficiency.
            Your job is to instantiate models using the SSM Creator tool based on task requirements.""",
            tools=[self.ssm_tool],
            verbose=True,
            allow_delegation=False,
            max_iter=4
        )

    def create_design_task(self, input_dim: int, output_dim: int) -> Task:
        """Create a task to design and instantiate an SSM."""
        return Task(
            description=f"""
            Design and initialize a State Space Model (SSM) for a task with:
            - Input Dimension: {input_dim}
            - Output Dimension: {output_dim}

            Steps:
            1. Determine optimal 'state_dim' and 'hidden_dim' considering the input size.
               (Hint: Start with state_dim around 32-64 for efficiency).
            2. Use the 'SSM Creator' tool to create the model.
            3. Report the 'model_path' and parameter count returned by the tool.
            """,
            agent=self.agent,
            expected_output="A report containing the path to the saved PyTorch model file and its configuration."
        )
