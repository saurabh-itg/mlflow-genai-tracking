# Li-Ion Battery State of Health on Embedded Targets

State of charge and state of health estimation for lithium-ion packs is a two-timescale problem.
Local signal shape over a few seconds tells you about charge; degradation across hundreds of cycles
tells you about health. A single architecture rarely serves both well.

The production model is a hybrid: one-dimensional convolutional layers extract local features from
the voltage and current traces, and LSTM layers carry state across cycles to capture degradation.
Training data covers more than fifty industrial assets and is versioned through Azure ML pipelines.

Deployment is the harder half. Inference has to run on Jetson Nano and on Raspberry Pi class
hardware without maintaining two codebases, so the inference path is written in Rust and compiled to
WebAssembly. The same artifact runs on both targets, and the WASM sandbox gives a clean memory
boundary on devices where a segfault means a truck roll.

Retraining is triggered by drift detection on the input distribution rather than on a fixed
calendar, because pack chemistry ages unevenly across sites.
