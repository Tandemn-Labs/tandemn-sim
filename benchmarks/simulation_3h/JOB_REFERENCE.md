# Three-hour simulation job reference

RPS values are derived from each workload's concurrency and target service
time. Estimated max RPS describes the intended traffic peak; AIPerf enforces
the concurrency cap rather than a hard maximum RPS.

| Online job | Pattern | Avg RPS | Estimated max RPS | Concurrency cap |
|---|---:|---:|---:|---:|
| Chat Free | Diurnal | 348.837 | 697.674 | 12,000 |
| Chat Premium | Diurnal | 22.959 | 45.918 | 900 |
| Code Autocomplete | Bursty | 256.016 | 896.057 | 5,000 |
| Safety Moderation | Bursty | 21,008.403 | 73,529.412 | 25,000 |
| Embeddings | Constant | 330,578.512 | 330,578.512 | 40,000 |
| Search RAG | Poisson | 95.588 | 191.176 | 2,600 |
| Search Rerank | Bursty | 17,746.229 | 62,111.801 | 10,000 |
| Support Assistant | Bursty | 70.644 | 247.253 | 4,500 |
| Code Morning Burst | Bursty | 829.082 | 2,901.786 | 13,000 |
| Enterprise Chat | Diurnal | 44.101 | 88.203 | 1,600 |
| Product Launch Chat | Bursty | 624.086 | 2,184.300 | 32,000 |
| Product Launch Moderation | Bursty | 51,948.052 | 181,818.182 | 60,000 |
| Premium Agents Kimi | Bursty | 0.300 | 1.499 | 280 |
| Frontier GLM Reasoning | Bursty | 0.472 | 2.358 | 300 |
| MoE Routing Qwen3 | Bursty | 4.337 | 21.687 | 900 |
| Safety P0 Live | Bursty | 123.457 | 370.370 | 2,500 |
| Meeting Summary | Bursty | 0.635 | 2.222 | 180 |
| Legal/Finance Review | Diurnal | 3.692 | 7.385 | 240 |
