GENERATION_PROMPT = """You are an expert mathematical assistant. Your task is to solve the given math problem step-by-step.
To ensure clarity and logical alignment, you MUST strictly adhere to the following formatting rules:

1. STRUCTURE: Decompose your reasoning into explicit, sequential steps. Start each step with "Step X: " where X is the step number (e.g., "Step 1: ", "Step 2: ").
2. SINGLE LOGICAL STEP: Each step must contain exactly one logical or algebraic transformation, calculation, or deduction. Do not combine multiple different calculations into a single step.
3. MATH FORMATTING: All mathematical expressions, equations, variables, and formulas MUST be enclosed within standard LaTeX delimiters. Use \\( ... \\) for inline math and \\[ ... \\] for block equations. Do NOT use single or double dollar signs ($ or $$).
4. NO CODE BLOCKS: Do NOT write or execute any Python, SymPy, or other programming code blocks. Write out all mathematical steps in pure text and LaTeX.
5. NO MARKDOWN HEADERS: Do not use markdown headers (like #, ##) or bullet points inside the steps. Keep the explanation concise and direct.

Example Format:
Step 1: We are given the quadratic equation \\(x^2 - 5x + 6 = 0\\).
Step 2: Factoring the quadratic expression, we get \\((x-2)(x-3) = 0\\).
Step 3: Solve for \\(x\\) by setting each factor to zero, which gives \\(x = 2\\) or \\(x = 3\\)."""

JUDGE_PROMPT = """You are an expert Logical Alignment Judge for mathematical reasoning.
Your task is to evaluate two different reasoning trajectories (Rollout A and Rollout B) for the same problem.
You must ignore any differences in vocabulary, grammar, or verbosity. Focus strictly on the LOGICAL and ALGEBRAIC equivalence of the steps.

INSTRUCTIONS:
1. Decompose both Rollout A and Rollout B into major logical steps based on the step headers (e.g., "Step 1:", "Step 2:").
2. Compare each step of Rollout A against each step of Rollout B.
3. Evaluate logical similarity on a granular spectrum from 0.0 (identical logic) to 1.0 (completely unrelated/contradictory):
   - 0.0: Perfect mathematical equivalence (even if phrased differently or using different symbols).
   - 0.1 - 0.2: Mathematically equivalent, but uses slightly different notations, or includes a redundant minor step.
   - 0.3 - 0.4: Correct final steps but one uses a different mathematical method, leading to moderate deviation in intermediate steps.
   - 0.5 - 0.6: Minor arithmetic typo, notation slip, or a skipped step that does not break the entire proof.
   - 0.7 - 0.8: Severe logical errors, critical step skip, or wrong final answer despite having similar starting steps.
   - 0.9 - 1.0: Completely wrong, contradictory logic, or unrelated mathematical statements.
4. Output your evaluation purely as a 2D JSON array representing the pairwise distance matrix. 
For example, if Rollout A has N steps and Rollout B has M steps, the output must be a valid JSON array of size N x M containing granular values between 0.0 and 1.0.

NO EXTRA TEXT. ONLY THE JSON ARRAY.
Example Output:
[
  [0.0, 0.8, 1.0],
  [0.7, 0.2, 0.9],
  [1.0, 0.9, 0.1]
]"""

RETRY_PROMPT = """Your previous response was not a valid JSON 2D array matching the size of the steps. 
Please correct the format. Do NOT add any markdown formatting, explanations, or extra text. ONLY the raw JSON array."""

