import { z, defineCollection } from "astro:content";

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

const blogCollection = defineCollection({ schema: blogSchema });
const projectCollection = defineCollection({ schema: projectSchema });
const awardCollection = defineCollection({ schema: awardSchema });

export const collections = {
    'blog': blogCollection,
    'project': projectCollection,
    'award': awardCollection
}
