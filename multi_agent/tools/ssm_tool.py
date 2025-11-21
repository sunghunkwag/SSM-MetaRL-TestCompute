"""SSM Tool - Interface to State Space Model components."""

from typing import Any, Dict
from crewai.tools import BaseTool
import torch
import os
import uuid
import sys

# Ensure we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.ssm import SSM

class SSMTool(BaseTool):
    name: str = "SSM Creator"
    description: str = (
        "Creates and initializes a State Space Model (SSM) architecture based on specifications. "
        "Use this tool when you need to design a new model for sequence data. "
        "It returns the file path of the saved model artifact."
    )

    def _run(self, state_dim: int, input_dim: int, output_dim: int, hidden_dim: int = 128) -> Dict[str, Any]:
        """
        Creates a PyTorch SSM instance and saves it to disk.

        Args:
            state_dim: Dimension of the latent state (recommend 32-256)
            input_dim: Input feature dimension
            output_dim: Output/Target dimension
            hidden_dim: Internal neural network hidden dimension

        Returns:
            Dictionary containing the path to the saved model artifact and parameter count.
        """
        try:
            # Agents operate on CPU by default for safety
            device = 'cpu'

            print(f"🛠️ [SSMTool] Initializing SSM: State={state_dim}, Hidden={hidden_dim}...")

            # 1. Instantiate the actual Core Model
            model = SSM(
                state_dim=int(state_dim),
                input_dim=int(input_dim),
                output_dim=int(output_dim),
                hidden_dim=int(hidden_dim),
                device=device
            )

            # 2. Calculate Statistics (Feedback for the Agent)
            param_count = sum(p.numel() for p in model.parameters())

            # 3. Save Artifact (Agents pass file paths, not objects)
            os.makedirs("saved_models", exist_ok=True)
            model_id = str(uuid.uuid4())[:8]
            save_path = f"saved_models/ssm_v1_{model_id}.pt"
            model.save(save_path)

            result = {
                "status": "success",
                "message": f"SSM initialized and saved successfully.",
                "model_path": save_path,
                "architecture": {
                    "state_dim": state_dim,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "hidden_dim": hidden_dim,
                    "params": f"{param_count:,}"
                }
            }
            return result

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "modeling_accuracy": 0.0
            }
