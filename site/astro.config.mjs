// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// The live simulator (antennaknobs' FastAPI workbench) — the primer's
// "turn the knob yourself" links land there.
const SIMULATOR_URL = "https://app.antennaknobs.dev/";

// https://astro.build/config
export default defineConfig({
  site: "https://momwire.dev",
  integrations: [
    starlight({
      title: "momwire",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/stevenmburns/momwire",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      // IBM Plex Sans / Mono — brand parity with the antennaknobs app and
      // docs site (custom.css points --sl-font / --sl-font-mono at them).
      head: [
        {
          tag: "link",
          attrs: { rel: "preconnect", href: "https://fonts.googleapis.com" },
        },
        {
          tag: "link",
          attrs: {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: true,
          },
        },
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap",
          },
        },
        // Cloudflare Web Analytics — momwire.dev's OWN site/token (created
        // 2026-08-15, "Enable with JS Snippet installation" mode: the zone
        // is DNS-only — Fly terminates TLS — so Cloudflare cannot inject
        // the beacon; this manual snippet is the only path that records).
        // Formerly shared antennaknobs' token, a rationale that died when
        // the site left that zone. The token is a public client-side id,
        // not a secret.
        {
          tag: "script",
          attrs: {
            defer: true,
            src: "https://static.cloudflareinsights.com/beacon.min.js",
            "data-cf-beacon": '{"token": "a1446be3e55f4096893cb315122220e7"}',
          },
        },
      ],
      sidebar: [
        {
          label: "Act I — From a wire to a matrix",
          items: [
            { label: "The question", slug: "act-1/the-question" },
            { label: "Solve for coefficients", slug: "act-1/coefficients" },
            { label: "The feed and the answer", slug: "act-1/the-feed" },
          ],
        },
        {
          label: "Act II — Bases and accuracy",
          items: [
            { label: "Sinusoids, NEC's bet", slug: "act-2/sinusoids" },
            { label: "Splines and junctions", slug: "act-2/splines" },
            { label: "Integrals done honestly", slug: "act-2/quadrature" },
            { label: "How do you know it's right?", slug: "act-2/validation" },
          ],
        },
        {
          label: "Act III — The ground",
          items: [
            { label: "Mirror worlds", slug: "act-3/mirror-worlds" },
            { label: "Real dirt, cheap", slug: "act-3/real-dirt" },
            { label: "Sommerfeld, or paying full price", slug: "act-3/sommerfeld" },
          ],
        },
        {
          label: "Act IV — Scale",
          items: [
            { label: "N² is the enemy", slug: "act-4/scaling" },
            { label: "Matrices that are secretly small", slug: "act-4/compression" },
            { label: "Arrays know their own symmetry", slug: "act-4/arrays" },
            { label: "Epilogue: the same math, twice", slug: "act-4/epilogue" },
          ],
        },
        {
          label: "Act V — The instrument",
          items: [
            { label: "The fourth cell", slug: "act-5/the-fourth-cell" },
          ],
        },
        {
          // Reference material, not narrative: normative contracts the code
          // is held to. Sits beside the acts rather than inside them.
          label: "Reference",
          items: [
            { label: "The nec2 deck dialect", slug: "reference/deck-grammar-nec2" },
          ],
        },
        {
          label: "Elsewhere",
          items: [
            { label: "Turn the knobs live", link: SIMULATOR_URL, attrs: { target: "_blank" } },
            { label: "antennaknobs docs", link: "https://antennaknobs.dev/", attrs: { target: "_blank" } },
          ],
        },
      ],
    }),
  ],
});
