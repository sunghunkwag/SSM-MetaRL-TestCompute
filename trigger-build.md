# Build Trigger

This file is created to trigger the Docker build workflow.

Build timestamp: 2025-10-22T13:09:00+09:00

Container should be available at: `ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest`

## Check Container

To verify the container was built successfully:

```bash
# Pull the container
docker pull ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest

# Run the container
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-testcompute:latest python main.py --help
```