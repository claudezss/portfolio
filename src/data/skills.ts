/** Skill groups, shown on the home page and the CV. */

export interface SkillGroup {
  name: string;
  skills: string[];
}

export const skillGroups: SkillGroup[] = [
  { name: "Power", skills: ["AC OPF", "DSSE", "DERMS", "Power flow", "GAMS", "Optimization"] },
  { name: "ML", skills: ["PyTorch", "PyG", "GNNs", "RL", "LLM agents", "Ray"] },
  {
    name: "Software",
    skills: ["Python", "Rust", "React", "FastAPI", "Flask", "Kafka", "Docker", "Kubernetes", "AWS · GCP · Azure", "CI/CD", "System design"],
  },
];
