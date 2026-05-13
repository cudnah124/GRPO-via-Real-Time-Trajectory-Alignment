GENERATION_PROMPT = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-reasoned answers."""

JUDGE_PROMPT = """You are an expert Logical Alignment Judge for mathematical reasoning.
Your task is to evaluate two different reasoning trajectories (Rollout A and Rollout B) for the same problem.
You must ignore any differences in vocabulary, grammar, or verbosity. Focus strictly on the LOGICAL and ALGEBRAIC equivalence of the steps.

INSTRUCTIONS:
1. Decompose both Rollout A and Rollout B into major logical steps.
2. Compare each step of Rollout A against each step of Rollout B.
3. If the steps are mathematically and logically equivalent (even if phrased differently), the distance is 0.0.
4. If they use a fundamentally different strategy or contain different arithmetic errors, the distance is 1.0.
5. Output your evaluation purely as a 2D JSON array representing the pairwise distance matrix. 
For example, if Rollout A has N steps and Rollout B has M steps, the output must be a valid JSON array of size N x M containing only 0.0 or 1.0.

NO EXTRA TEXT. ONLY THE JSON ARRAY.
Example Output:
[[0.0, 1.0], [1.0, 0.0]]
"""

RETRY_PROMPT = """Your previous response was not a valid JSON 2D array. 
Please correct the format. Do NOT add any markdown formatting, explanations, or extra text. ONLY the raw JSON array.
"""
