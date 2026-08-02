# Fine-Tuning TinyLlama for Edge Inference

Energy-forecasting explanations had to be generated on-site, in plants where there is no datacentre
GPU and often no reliable outbound network. That rules out an API call to a frontier model and puts
a hard ceiling on parameter count.

TinyLlama-1.1B was fine-tuned with QLoRA on more than ten thousand domain instruction pairs covering
forecast narration, anomaly explanation, and maintenance summarisation. Four-bit quantization keeps
the served footprint inside 6 GB of VRAM, which fits the industrial PCs already installed on the
floor.

Token-level accuracy on the held-out domain set reached 93.4 percent. That number matters less than
the failure mode: the model declines to speculate about equipment it has not seen, because the
instruction set includes explicit refusal examples.

Adapters are versioned separately from the base weights, so a site can roll back a fine-tune without
re-downloading the base model over a metered link.
