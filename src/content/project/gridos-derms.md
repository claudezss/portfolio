---
title: "GridOS DERMS"
description: "GridOS® Distributed Energy Resource Management"
pubDate: "Sep 10 2022"
heroImage: "/project.png"
tag: "Industry · GE Digital"
featured: true
order: 1
role: "Software Developer → Senior"
period: "2019 – 2024"
company: "GE Vernova"
stack: ["Python", "GAMS", "Flask", "Kafka"]
---

[GridOS-DERMS](https://www.ge.com/digital/sites/default/files/download_assets/opus-one-derms-from-ge-digital.pdf) brings together the full breadth of Opus One’s innovations in model-based decision making,
real-time grid visibility, and distributed energy resource valuation.

Features include:

- Online power flow for full distribution visibility, based on breakthrough distribution system state estimation
- Direct DER monitoring, control, and dispatch or aggregator integration
- Optimization under two-way power flows, including VVO, CVR, and restoration capacity
- Constraint management using dynamic, nodal level, real-time, and forecasted hosting capacity
- Optimal asset utilization and grid benefits through locational marginal pricing of resources
- Day-ahead clearing and unit commitment, with hour- ahead and five-minute-ahead adjustments (based on security-constrained optimal power flow)

## My part

I owned parts of the `optimization engine`, the Python package that runs power-flow analysis.
I built OPF objectives in GAMS (cost minimization, operating envelopes, bid fulfillment),
wrapped them in Flask and Kafka services, and validated results against published IEEE PES data.
