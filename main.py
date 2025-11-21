"""
Autonomous-SSM-MetaRL Entry Point
Orchestrates the collaboration between AI Agents and the Deep Learning Core.
"""
import os
from multi_agent.workflows.research_workflow import ResearchWorkflow

def main():
    print("\n🤖 Initializing Autonomous-SSM-MetaRL Framework...")
    print("===================================================")
    
    # Ensure artifact directory exists
    os.makedirs("saved_models", exist_ok=True)

    # 1. Initialize the Workflow
    workflow = ResearchWorkflow()
    
    # 2. Define the Research Goal (e.g., Solving HalfCheetah)
    # HalfCheetah-v4 has roughly 17 obs dim and 6 action dim
    input_dim = 17
    output_dim = 6
    
    print(f"🎯 Target Task: HalfCheetah-v4 (In: {input_dim}, Out: {output_dim})")

    # 3. Run the Design Phase
    result = workflow.run_ssm_design(input_dim, output_dim)
    
    print("\n\n########################")
    print("##  Research Results  ##")
    print("########################")
    print(result)

if __name__ == "__main__":
    main()
