# Subset sum with Grover's algorithm

## Installation
The easiest way to get the python notebook up and run is to install uv and then run the following command within the folder:
```bash
uv run quantum-subset-sum
```
This will install all dependencies for both the package and the notebook.

To run the tests run the following command:

```bash
uv run pytest -v
```

The main sampling routine is with the function:

```python
from quantum_subset_sum.subset_sum import sampling_routine

sampling_routine(input_list, target)
```

The input list must only contain positive integers and the target must also be a positive integer.