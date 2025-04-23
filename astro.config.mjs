import {defineConfig} from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";
import partytown from "@astrojs/partytown";
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypeKatex from 'rehype-katex'
import rehypeStringify from 'rehype-stringify'

// https://astro.build/config
export default defineConfig({
    site: 'https://claudezss.com',

    markdown: {
        remarkPlugins: [remarkParse, remarkMath, remarkRehype],
        rehypePlugins: [rehypeKatex, rehypeStringify],
    },
    integrations: [mdx(), sitemap(), tailwind(),
        partytown({
            // Adds dataLayer.push as a forwarding-event.
            config: {
                forward: ["dataLayer.push"],
            },
        }),
    ]
});
