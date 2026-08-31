/**
 * Single source of truth for education and work history.
 *
 * Both the home page timeline (src/components/home/ExperienceTimeline.astro)
 * and the CV page (src/pages/cv.astro) render from here, so a role only ever
 * needs editing in one place. `highlights` is the condensed copy shown on the
 * home page; `details` is the full copy shown on the CV.
 */

export interface Education {
  degree: string;
  school: string;
  startDate: string;
  endDate: string;
  logo: string;
}

export interface Role {
  /** Short label for the home page timeline, e.g. "2024 — PRESENT". */
  period: string;
  /** Home page heading — includes the product where it matters. */
  title: string;
  /** Home page sub-heading. */
  org: string;
  /** CV heading. */
  position: string;
  /** CV sub-heading. */
  company: string;
  startDate: string;
  endDate: string;
  logo: string;
  /** Condensed bullets for the home page. May contain inline HTML. */
  highlights: string[];
  /** Optional lead-in paragraph shown above the CV bullets. */
  intro?: string;
  /** Full bullets for the CV. */
  details: string[];
}

export const education: Education[] = [
  {
    degree: "Master of Engineering - MEng, ECE",
    school: "University of Toronto, Toronto, Canada",
    startDate: "2023",
    endDate: "2026",
    logo: "/uoft.png",
  },
  {
    degree: "Bachelor of Engineering - BE, ECE",
    school: "University of Western Ontario, London, Canada",
    startDate: "2013",
    endDate: "2017",
    logo: "/uwo.png",
  },
];

export const roles: Role[] = [
  {
    period: "2024 — PRESENT",
    title: "Senior Software Engineer",
    org: "ThinkLabs AI Inc.",
    position: "Senior Software Engineer",
    company: "ThinkLabs AI Inc.",
    startDate: "2024",
    endDate: "Present",
    logo: "/thinklabs_ai_logo.png",
    highlights: [
      "GNN-based models for Distribution System State Estimation (<code>DSSE</code>)",
      "Temporal-Spatial GNN (<code>TSGNN</code>) for grid measurement anomaly detection",
      "Heterogeneous GNN (<code>HGNN</code>) for time-series power flow analysis",
      "Scalable data-generation, training &amp; inference pipelines on <code>Ray.io</code> + <code>Kubernetes</code>",
    ],
    details: [
      "Developed and optimized Graph Neural Network (GNN)-based AI models for Distribution System State Estimation (DSSE), improving accuracy and scalability.",
      "Designed and implemented Temporal-Spatial Graph Neural Network (TSGNN)-based models for grid measurement anomaly detection, enhancing grid reliability and predictive capabilities.",
      "Created advanced Heterogeneous Graph Neural Network (HGNN)-based AI models for time-series power flow analysis.",
      "Built and enhanced scalable data generation, model training, and inference pipelines using Ray.io and Kubernetes (K8s) to streamline end-to-end workflows and ensure high-performance deployment.",
    ],
  },
  {
    period: "2023 — 2024",
    title: "Senior Software Developer — GridOS-DERMS",
    org: "GE Digital / GE Vernova",
    position: "Senior Software Developer",
    company: "GridOS-DERMS, GE Digital, GE Vernova",
    startDate: "2023",
    endDate: "2024",
    logo: "/gevernova_logo.jpeg",
    highlights: [
      "Scoped milestones with product management; broke down features and estimated delivery",
      "Cross-functional design with architects, PM and QA; evaluated new tools &amp; frameworks",
      "Mentored junior developers",
    ],
    intro:
      "My responsibilities include all items outlined in the Software Developer role at GridOS-DERMS, GE Digital, GE Vernova, as well as:",
    details: [
      "Assisting the product manager in defining milestone scopes and refining acceptance criteria for features.",
      "Breaking down large features into smaller, manageable tasks and providing time estimation to the product manager.",
      "Collaborating with cross-functional teams, including architects, product managers, and quality assurance engineers to design integration solutions.",
      "Contributing to the evaluation of new tools and frameworks that have the potential to enhance both application performance and the overall development process.",
      "Providing mentorship and guidance to junior developers, addressing their questions in the workplace, and actively sharing knowledge with them.",
    ],
  },
  {
    period: "2019 — 2023",
    title: "Software Developer — GridOS-DERMS",
    org: "GE Digital / GE Vernova",
    position: "Software Developer",
    company: "GridOS-DERMS, GE Digital, GE Vernova",
    startDate: "2019",
    endDate: "2023",
    logo: "/gevernova_logo.jpeg",
    highlights: [
      "Mathematical optimization models of grids with DERs in <code>GAMS</code>",
      "Maintained the optimization-engine Python package for power flow analysis",
      "OPF objectives: cost minimization, operational envelopes, bid fulfillment",
      "RESTful microservice APIs (<code>Flask</code>) and <code>Kafka</code> queue-based services",
      "Validated OPF results against IEEE PES published data",
    ],
    details: [
      "Design, implement and improve mathematical optimization models to simulate electric power grid with distributed energy resources (DERs) by using GAMS",
      "Maintain `optimization engine`: the python package to run power flow analysis",
      "Design, implement and improve OPF(optimal power flow) objectives, e.g : cost minimization, operational envelope, bid fulfillment, etc",
      "Design, implement and improve RESTful APIs on micro-service to run powerflow analyses. (flask)",
      "Design, implement and improve Queue-based micro-service to run powerflow analyses. (Kafka)",
      "Design, implement and improve benchmark testing tools",
      "Write documentations and white papers for the optimization models and applications developed",
      "Analysis and validate optimal power flow results of customers' networks, and compare results with published data from IEEE Power &amp; Energy Society (IEEE PES)",
    ],
  },
  {
    period: "2017 — 2019",
    title: "Full Stack Developer",
    org: "GreenfieldSCM",
    position: "Full Stack Developer",
    company: "GreenfieldSCM, Supply Chain Management",
    startDate: "2017",
    endDate: "2019",
    logo: "/greenfieldscm.jpeg",
    highlights: [
      "Visualized supply-chain management system (tracking, payment, invoicing) — <code>React</code> + <code>Django</code> + <code>MySQL</code>",
    ],
    details: [
      "Developing Visualized Supply Chain Management System, include tracking, payment, and invoice generation functions. This web app is built by React (for frontend), MYSQL and Django (for backend).",
    ],
  },
];
