# Tracking GenAI Applications with MLflow

Classical ML tracking assumes a training run: fixed dataset in, metrics out. GenAI applications break
that assumption. There is often no training run at all, the interesting artifact is a prompt, and the
failure modes are latency, cost, and hallucination rather than validation loss.

MLflow covers this with four pieces. Tracing records the span tree of a single request, so retrieval,
prompt construction, and generation each carry their own latency and inputs. Experiments and runs
capture a configuration sweep, so top_k or chunk size or prompt version become comparable rows.
The Prompt Registry versions prompt templates independently of code, with aliases pointing at the
production version. The Model Registry holds the deployable pyfunc wrapper and its aliases.

The practical discipline is to log the same metric family for every configuration: a quality metric,
a cost metric, and a latency metric. A configuration that improves groundedness by four points while
tripling cost per query is a decision, not an improvement, and the comparison view should make that
trade visible in one screen.

Autologging covers common frameworks, but the manual span API is worth using for custom pipelines
because it lets you name spans after your own architecture instead of after a library's internals.
