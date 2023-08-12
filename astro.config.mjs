import {defineConfig} from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";
import partytown from "@astrojs/partytown";

// https://astro.build/config
import image from "@astrojs/image";

// https://astro.build/config
export default defineConfig({
    site: 'https://claudezss.com',
    integrations: [mdx(), sitemap(), tailwind(), image(
        {
            serviceEntryPoint: '@astrojs/image/sharp',
            cacheDir: "./.cache/image",
            logLevel: 'debug',
        }
    ),
        partytown({
            // Adds dataLayer.push as a forwarding-event.
            config: {
                forward: ["dataLayer.push"],
            },
        }),
    ]
});