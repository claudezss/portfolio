/**
 * Contact links. The home page shows them as buttons under the intro; every
 * page shows them as icons in the footer. `icon` names a symbol in
 * src/components/Icon.astro.
 */

export type IconName = "mail" | "github" | "linkedin" | "orcid" | "bilibili";

export interface SocialLink {
  label: string;
  href: string;
  icon: IconName;
}

export const email = "claude.yan.zhang@gmail.com";

export const social: SocialLink[] = [
  { label: "GitHub", href: "https://github.com/claudezss", icon: "github" },
  { label: "LinkedIn", href: "https://www.linkedin.com/in/yan-zhang-p-eng-a21428113/", icon: "linkedin" },
  { label: "ORCID", href: "https://orcid.org/0000-0001-9867-4924", icon: "orcid" },
  { label: "Bilibili", href: "https://space.bilibili.com/6151386", icon: "bilibili" },
];
