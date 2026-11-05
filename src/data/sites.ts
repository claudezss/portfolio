/**
 * Live products that run on their own sites. They show at the bottom of the
 * home page and the projects page. To add a site, add one entry here.
 */

export interface LiveSite {
  name: string;
  url: string;
  /** One sentence. Leave out to show only the name and the address. */
  description?: string;
}

export const sites: LiveSite[] = [
  { name: "Airlox", url: "https://airlox.app/" },
];
