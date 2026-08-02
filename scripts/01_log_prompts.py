"""Register prompt templates in the MLflow Prompt Registry.

Why bother: once a prompt is a registered, versioned entity, a run can record *which*
prompt version produced it, and you can promote a version with an alias instead of
editing a string in source and hoping the deployment picked it up.
"""
import _bootstrap  # noqa: F401
import mlflow

from genai_app.prompts import PROMPTS

_bootstrap.init()

registered = []
for version, template in PROMPTS.items():
    name = f"rag-answer-{version}"
    try:
        p = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=f"RAG answer prompt, variant {version}",
            tags={"task": "rag-qa", "domain": "industrial"},
        )
        registered.append((name, p.version))
        print(f"registered prompt {name} v{p.version}")
    except Exception as exc:  # older MLflow, or registry unavailable
        print(f"prompt registry unavailable ({type(exc).__name__}); logging as artifact instead")
        with mlflow.start_run(run_name=f"prompt-{version}"):
            mlflow.log_text(template, f"prompts/{name}.txt")
            mlflow.set_tags({"artifact_type": "prompt", "prompt_version": version})
        registered.append((name, "artifact"))

print(f"\n{len(registered)} prompt variants tracked")
