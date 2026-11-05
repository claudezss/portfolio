import { getCollection, type CollectionEntry } from "astro:content";

/** Posts, newest first. */
export async function getPosts(): Promise<CollectionEntry<"blog">[]> {
  return (await getCollection("blog")).sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/** Projects by `order`, then newest first. */
export async function getProjects(): Promise<CollectionEntry<"project">[]> {
  return (await getCollection("project")).sort((a, b) => {
    const ao = a.data.order ?? Number.MAX_SAFE_INTEGER;
    const bo = b.data.order ?? Number.MAX_SAFE_INTEGER;
    if (ao !== bo) return ao - bo;
    return b.data.pubDate.valueOf() - a.data.pubDate.valueOf();
  });
}

/** Awards, newest first. */
export async function getAwards(): Promise<CollectionEntry<"award">[]> {
  return (await getCollection("award")).sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/** "Jun 10, 2022". Dates are stored as UTC midnight. */
export function longDate(date: Date): string {
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

/** About 200 words a minute; at least one minute. */
export function readingTime(body: string | undefined): string {
  const words = (body ?? "").split(/\s+/).filter(Boolean).length;
  return `${Math.max(1, Math.round(words / 200))} min read`;
}
