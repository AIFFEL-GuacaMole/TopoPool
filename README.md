# Topological Pooling on Graphs

## Wit-TopoPool

A fork of [TopologicalPool](https://github.com/topologicalpooling/TopologicalPool), which is the official codebase of the paper "Topological Pooling on Graphs" (AAAI 2023).

## Environment

### Dependency

**Python 3.10**

|    **Package**  | **Version** *(if matters)* | *remarks* |
|:---------------:|:-------------:|:--:|
|      torch      |    2.1        | same version in Colab,<br>the original repo uses v1.11.0 |
|      gudhi      |    3.10.1     | *up to date* |
|     networkx    |    2.8.4      | *up to date* |
|      numpy      | 1.26.4 (<2.0) | same version in Colab |
|      scipy      |    1.15.1     | *up to date* |
|      persim     |    0.3.1      | *up to date* |
|     dionysus    | 2.0.10 (>2.0) | need [**Boost**](https://formulae.brew.sh/formula/boost-python3) as a prerequisite to install |
| torch-geometric | 2.3.1 (<2.4)  | must be lower than v2.4 to use `topk()` and `filter_adj()` |

### Tool

This repo recommends to use [UV](https://docs.astral.sh/uv/getting-started/installation/#homebrew) as the Python environment manager.  
If you have UV, execute the commands below in the repo's root directory. It will automatically setup the environment.

```shell
uv venv
uv sync
```