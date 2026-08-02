# MILP Boiler Scheduling Against On-Site Solar

A plant that runs gas boilers and electric boilers alongside on-site solar generation has a
scheduling problem, not a modelling problem. The decision is which boiler serves demand in each
interval, given the solar output forecast for that interval and the tariff for grid import.

We formulate it as a mixed-integer linear program. Binary variables select the active boiler per
interval, continuous variables carry thermal load, and the objective minimises fuel cost subject to
demand satisfaction, ramp-rate limits, and minimum-uptime constraints.

The solver runs on a fifteen minute cadence. Solar forecast error is absorbed by re-solving rather
than by carrying a large safety margin, which is what makes the savings real. Measured against the
prior rule-based schedule, natural gas consumption fell by 3 percent and overall plant efficiency
rose by 2 to 3 percent.

A live blowdown recommender runs alongside the scheduler. It issues hourly maintenance triggers
based on conductivity trend, so that boiler efficiency does not silently decay between scheduled
service windows. Telemetry moves over Kafka, workloads are orchestrated on Rancher, and the metrics
land in Prometheus with Grafana dashboards for the plant operators.
