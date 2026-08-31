import type { CollectionEntry } from "astro:content";

export type BlogPost = CollectionEntry<"blog">;

export interface TagGroup {
  /** Display label, using the casing from the first post that used the tag. */
  tag: string;
  /** URL-safe slug used by /blog/tags/[tag]. */
  slug: string;
  /** Posts carrying the tag, newest first. */
  posts: BlogPost[];
}

/**
 * URL-safe slug for a tag: "AI&ML" -> "ai-ml", "Foundations of DA & ML" ->
 * "foundations-of-da-ml". Raw tags contain spaces and ampersands, which do not
 * belong in a path segment.
 */
export function tagSlug(tag: string): string {
  return tag
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Groups posts by tag slug, alphabetically by label. Tags that slugify to the
 * same value are merged into one group rather than producing duplicate routes.
 */
export function groupPostsByTag(posts: BlogPost[]): TagGroup[] {
  const byDate = [...posts].sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );
  const groups = new Map<string, TagGroup>();

  for (const post of byDate) {
    const seen = new Set<string>();
    for (const tag of post.data.tags ?? []) {
      const slug = tagSlug(tag);
      if (!slug || seen.has(slug)) continue; // ignore repeats within one post
      seen.add(slug);

      const group = groups.get(slug);
      if (group) group.posts.push(post);
      else groups.set(slug, { tag, slug, posts: [post] });
    }
  }

  return [...groups.values()].sort((a, b) => a.tag.localeCompare(b.tag));
}
