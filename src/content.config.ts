import { z, defineCollection } from "astro:content";
import { glob } from "astro/loaders";

const blogSchema = z.object({
    title: z.string(),
    description: z.string(),
    tags: z.array(z.string()),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    badge: z.string().optional(),
    video: z.string().optional(),
    video_title: z.string().optional(),
});

const projectSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    badge: z.string().optional(),
    video: z.string().optional(),
    video_title: z.string().optional(),
});

const awardSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    badge: z.string().optional(),
});

export type BlogSchema = z.infer<typeof blogSchema>;
export type ProjectSchema = z.infer<typeof projectSchema>;
export type AwardSchema = z.infer<typeof awardSchema>;

const blogCollection = defineCollection({
    loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/blog" }),
    schema: blogSchema,
});
const projectCollection = defineCollection({
    loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/project" }),
    schema: projectSchema,
});
const awardCollection = defineCollection({
    loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/award" }),
    schema: awardSchema,
});

export const collections = {
    'blog': blogCollection,
    'project': projectCollection,
    'award': awardCollection,
};
