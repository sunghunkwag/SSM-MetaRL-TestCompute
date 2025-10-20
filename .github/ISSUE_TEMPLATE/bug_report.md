---
name: Bug report
about: Report a bug with detailed debug information
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## ⚠️ REQUIRED: Debug Mode Output

**Please run your code with DEBUG mode enabled and paste the FULL output below.**

### How to enable DEBUG mode:

```bash
# Linux/macOS
export DEBUG=True
python your_script.py

# Windows (PowerShell)
$env:DEBUG="True"
python your_script.py

# Windows (CMD)
set DEBUG=True
python your_script.py
```

### Debug Output:

```
Paste the COMPLETE error output here, including:
- Full stack trace
- All error messages
- Any warning messages
- Debug logs if available
```

## To Reproduce

Steps to reproduce the behavior:

1. Run command: `...`
2. With these parameters: `...`
3. See error

### Minimal Reproducible Example

```python
# Please provide a minimal code example that reproduces the issue
# Include all necessary imports and setup

```

## Expected Behavior

A clear and concise description of what you expected to happen.

## Environment Information

**Please provide the following information:**

- **Python version:** [Run: `python --version`]
- **PyTorch version:** [Run: `python -c "import torch; print(torch.__version__)"`]
- **CUDA version (if using GPU):** [Run: `python -c "import torch; print(torch.version.cuda)"`]
- **Operating System:** [e.g., Ubuntu 22.04, Windows 11, macOS 14]
- **Installation method:** [e.g., pip, conda, from source]

### Package Versions

```bash
# Run and paste output:
pip list | grep -E "torch|numpy|gymnasium"
```

## Additional Context

Add any other context about the problem here.

### Screenshots (if applicable)

If applicable, add screenshots to help explain your problem.

### Related Issues

Link to any related issues or pull requests.

---

## Checklist

Before submitting, please check:

- [ ] I have enabled DEBUG mode and included the full output above
- [ ] I have provided a minimal reproducible example
- [ ] I have included my environment information (Python, PyTorch, CUDA versions)
- [ ] I have searched for similar issues
- [ ] I have read the [Debug & Development Mode](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute#debug--development-mode) section in the README

---

### For Maintainers

**Debug Commands for Investigation:**

```bash
# Run tests with full debug output
DEBUG=True python -m pytest -v --tb=long -s --log-cli-level=DEBUG

# Run specific test with detailed traceback
DEBUG=True python -m pytest path/to/test.py::test_function -v --tb=long -s

# Run with profiling (if needed)
DEBUG=True python -m cProfile -o profile.stats your_script.py
```
