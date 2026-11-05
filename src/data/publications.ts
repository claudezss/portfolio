/** Papers. Awards come from the award content collection. */

export interface Publication {
  year: number;
  title: string;
  venue: string;
  url: string;
}

export const publications: Publication[] = [
  {
    year: 2025,
    title: "Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control",
    venue: "arXiv preprint",
    url: "https://arxiv.org/abs/2508.05702",
  },
  {
    year: 2022,
    title:
      "Three-Phase Distribution Locational Marginal Pricing for Competitive Electricity Markets with Distributed Generators and Flexible Loads",
    venue: "IEEE PES Innovative Smart Grid Technologies (ISGT)",
    url: "https://ieeexplore.ieee.org/abstract/document/9817498",
  },
];
