# Model Selection Rationale

This note collects reusable text for explaining why the benchmark currently evaluates a set of Chinese commercial LLMs rather than directly including GPT, Claude, or Gemini in the same study. The goal is to stay honest about the practical reason, including cost, while presenting the decision in a methodologically defensible way.

## Paper Limitations Paragraph

Our model pool was restricted to a set of commercially accessible models that could be stably evaluated within a single deployment ecosystem and within our budgeted large-scale API evaluation setting. As a result, the benchmark does not yet include cross-provider frontier models such as GPT, Claude, or Gemini, and our findings should therefore be interpreted as ecosystem-specific rather than as a universal ranking over all closed-source LLMs. We made this restriction deliberately to reduce provider-level confounders, including differences in hidden system prompts, reasoning-control interfaces, safety filtering, timeout behavior, rate limits, and context management, all of which can affect both Direct and RAG conditions independently of the RAG system itself. Future work should extend the study with a smaller cross-provider anchor set to test the external validity of the present conclusions.

## Reviewer Response

We thank the reviewer for raising this point. Our goal in this benchmark was not to construct a universal leaderboard over all commercial frontier LLMs, but to measure the incremental benefit of RAG over Direct under a controlled and reproducible large-scale evaluation setup. In practice, the current model set was constrained by stable API accessibility and evaluation budget, and we agree that this limits the breadth of the model pool. We have therefore revised the manuscript to state this limitation explicitly rather than leaving it implicit.

At the same time, we chose not to mix providers such as GPT, Claude, and Gemini into the main benchmark because doing so would introduce substantial provider-level confounders beyond model capability itself. These include differences in hidden system prompts, reasoning-control interfaces, safety and refusal policies, timeout behavior, rate limits, and context-window management. Since our primary research question concerns the relative gain of RAG over Direct, we prioritized internal validity and reproducibility over maximal provider coverage. We therefore interpret the current results as valid within the evaluated deployment ecosystem, and we identify cross-provider validation with a smaller anchor set as an important direction for future work.

## Shorter Reviewer Version

We appreciate the reviewer's suggestion. The current benchmark was designed to evaluate the gain of RAG over Direct under a controlled, reproducible, and budget-feasible API setting rather than to provide a universal leaderboard across all frontier LLMs. Our present model pool was therefore limited to models that could be stably accessed at the required evaluation scale within one deployment ecosystem. We now state this explicitly as a limitation. We did not include GPT, Claude, or Gemini in the main benchmark because cross-provider comparison would introduce additional confounders, such as differences in hidden prompts, reasoning controls, safety policies, rate limits, and timeout behavior. Future work will extend the study with a smaller cross-provider anchor set to assess external validity.

## Tone Guidance

When using the text above, keep the following principles:

Do not say that GPT, Claude, or Gemini were excluded because they were "unimportant."

Do not deny the role of cost if the real reason included budget constraints.

Do not frame the current benchmark as a universal ranking over all commercial LLMs.

Do say that the current study prioritized internal validity, reproducibility, and feasible large-scale evaluation.

Do say that cross-provider validation is future work rather than pretending the limitation does not exist.
