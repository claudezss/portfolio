import * as THREE from "three";

/**
 * Home page animations: the hero grid-network scene and the React Bits-style
 * micro-interactions (scramble text, typewriter, card swap, timeline reveal,
 * spotlight tracking). The orb background is shared with the inner pages and
 * lives in orb-background.ts.
 *
 * The site uses <ClientRouter/>, so every listener, observer, timer and WebGL
 * context registered here is tracked and torn down on astro:before-swap.
 */

type Cleanup = () => void;
let cleanups: Cleanup[] = [];

function on<K extends keyof WindowEventMap>(
  target: Window | Document | Element,
  type: K | string,
  handler: EventListenerOrEventListenerObject,
  options?: AddEventListenerOptions
) {
  target.addEventListener(type, handler, options);
  cleanups.push(() => target.removeEventListener(type, handler, options));
}

function track(fn: Cleanup) {
  cleanups.push(fn);
}

function teardown() {
  cleanups.forEach((fn) => {
    try {
      fn();
    } catch {
      /* keep tearing down the rest */
    }
  });
  cleanups = [];
}

function initHome() {
  teardown(); // guard against double-init on repeat navigations

  const hero = document.querySelector<HTMLElement>("header.home-hero");
  if (!hero) return; // not the home page

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- Nav scroll state ---------------- */
  const nav = document.getElementById("home-nav");
  if (nav) {
    const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 30);
    on(window, "scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ---------------- Scroll reveal ---------------- */
  const revealEls = document.querySelectorAll<HTMLElement>(".reveal:not(.is-visible)");
  if (!reducedMotion && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    revealEls.forEach((el) => io.observe(el));
    track(() => io.disconnect());
  } else {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  }

  /* ---------------- Decrypt / scramble name ---------------- */
  const scrambleEl = document.getElementById("scramble-name");
  if (scrambleEl && !reducedMotion) {
    const finalText = scrambleEl.textContent ?? "";
    const glyphs = "!<>-_\\/[]{}—=+*^?#01";
    const totalFrames = 46;
    let frame = 0;
    let rafId = 0;
    scrambleEl.textContent = "";
    const tick = () => {
      let out = "";
      const revealCount = Math.floor((frame / totalFrames) * finalText.length);
      for (let i = 0; i < finalText.length; i++) {
        if (i < revealCount) out += finalText[i];
        else if (finalText[i] === " ") out += " ";
        else out += glyphs[(Math.random() * glyphs.length) | 0];
      }
      scrambleEl.textContent = out;
      frame++;
      if (frame <= totalFrames) rafId = requestAnimationFrame(tick);
      else scrambleEl.textContent = finalText;
    };
    rafId = requestAnimationFrame(tick);
    track(() => {
      cancelAnimationFrame(rafId);
      scrambleEl.textContent = finalText;
    });
  }

  /* ---------------- Rotating role typewriter ---------------- */
  const roleEl = document.getElementById("role-text");
  if (roleEl) {
    let roles: string[] = [];
    try {
      roles = JSON.parse(roleEl.dataset.roles ?? "[]");
    } catch {
      roles = [];
    }
    if (roles.length) {
      if (reducedMotion) {
        roleEl.textContent = roles.join(" · ");
      } else {
        let roleIdx = 0;
        let charIdx = 0;
        let deleting = false;
        let timer: ReturnType<typeof setTimeout>;
        const type = () => {
          const cur = roles[roleIdx];
          if (!deleting) {
            charIdx++;
            roleEl.textContent = cur.slice(0, charIdx);
            if (charIdx === cur.length) {
              deleting = true;
              timer = setTimeout(type, 2100);
              return;
            }
            timer = setTimeout(type, 46);
          } else {
            charIdx--;
            roleEl.textContent = cur.slice(0, charIdx);
            if (charIdx === 0) {
              deleting = false;
              roleIdx = (roleIdx + 1) % roles.length;
              timer = setTimeout(type, 350);
              return;
            }
            timer = setTimeout(type, 22);
          }
        };
        timer = setTimeout(type, 900);
        track(() => clearTimeout(timer));
      }
    }
  }

  /* ---------------- Projects: card swap ---------------- */
  const stage = document.getElementById("swap-stage");
  if (stage) {
    const swapCards = Array.from(stage.querySelectorAll<HTMLElement>(".swap-card"));
    const listBtns = Array.from(document.querySelectorAll<HTMLButtonElement>("#swap-list button"));
    let order = swapCards.map((_, i) => i); // order[slot] = card index
    let swapping = false;

    const slotTransform = (slot: number) =>
      `translateX(${slot * 6}%) translateY(${-slot * 7.5}%) translateZ(${-slot * 78}px)`;

    const applySlots = () => {
      order.forEach((ci, slot) => {
        const el = swapCards[ci];
        el.style.zIndex = String(swapCards.length - slot);
        el.classList.toggle("front", slot === 0);
        el.style.transform = slotTransform(slot);
        el.style.opacity = slot > 2 ? "0.15" : String(1 - slot * 0.12);
      });
      listBtns.forEach((b) => {
        b.classList.toggle("active", Number(b.dataset.idx) === order[0]);
      });
    };
    applySlots();

    const swapNext = () => {
      if (reducedMotion) {
        order.push(order.shift() as number);
        applySlots();
        return;
      }
      if (swapping) return;
      swapping = true;
      const frontCard = swapCards[order[0]];
      frontCard.classList.add("dropping");
      frontCard.style.transform = "translateX(4%) translateY(120%) translateZ(30px) rotate(4deg)";
      frontCard.style.opacity = "0";
      const t1 = setTimeout(() => {
        order.push(order.shift() as number);
        frontCard.classList.remove("dropping");
        // jump the dropped card to the back slot without transition
        frontCard.style.transition = "none";
        frontCard.style.transform = slotTransform(swapCards.length - 1);
        frontCard.style.zIndex = "1";
        void frontCard.offsetWidth; // reflow
        frontCard.style.transition = "";
        applySlots();
        const t2 = setTimeout(() => {
          swapping = false;
        }, 400);
        track(() => clearTimeout(t2));
      }, 480);
      track(() => clearTimeout(t1));
    };

    let swapTimer: ReturnType<typeof setInterval> | null = null;
    let stageVisible = true;
    let stageHovered = false;
    const startTimer = () => {
      if (swapTimer || reducedMotion) return;
      swapTimer = setInterval(() => {
        if (stageVisible && !stageHovered && !document.hidden) swapNext();
      }, 4000);
    };
    const restartTimer = () => {
      if (swapTimer) {
        clearInterval(swapTimer);
        swapTimer = null;
      }
      startTimer();
    };

    const bringToFront = (idx: number) => {
      if (order[0] === idx) return;
      const pos = order.indexOf(idx);
      order = order.slice(pos).concat(order.slice(0, pos)); // keep cyclic order
      applySlots();
      restartTimer();
    };
    listBtns.forEach((b) => {
      on(b, "click", () => bringToFront(Number(b.dataset.idx)));
    });

    on(stage, "mouseenter", () => {
      stageHovered = true;
    });
    on(stage, "mouseleave", () => {
      stageHovered = false;
    });
    if ("IntersectionObserver" in window) {
      const so = new IntersectionObserver(
        (entries) => {
          stageVisible = entries[0].isIntersecting;
        },
        { threshold: 0.2 }
      );
      so.observe(stage);
      track(() => so.disconnect());
    }
    startTimer();
    track(() => {
      if (swapTimer) clearInterval(swapTimer);
    });
  }

  /* ---------------- Spotlight tracking (project + experience cards) ---------------- */
  document.querySelectorAll<HTMLElement>(".swap-card, .tl-card").forEach((card) => {
    on(card, "mousemove", (event) => {
      const e = event as MouseEvent;
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    });
  });

  /* ---------------- Experience timeline: ignite + scroll-drawn line ---------------- */
  const tlItems = Array.from(document.querySelectorAll<HTMLElement>(".tl-item"));
  const tlProgress = document.getElementById("tl-progress");
  const timelineEl = document.getElementById("timeline");
  if (tlItems.length) {
    if (reducedMotion || !("IntersectionObserver" in window)) {
      tlItems.forEach((it) => it.classList.add("lit"));
      if (tlProgress) tlProgress.style.height = "100%";
    } else {
      const tlo = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add("lit");
              tlo.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.25 }
      );
      tlItems.forEach((it) => tlo.observe(it));
      track(() => tlo.disconnect());

      if (tlProgress && timelineEl) {
        let ticking = false;
        const update = () => {
          ticking = false;
          const r = timelineEl.getBoundingClientRect();
          const anchor = window.innerHeight * 0.7;
          const h = Math.max(0, Math.min(anchor - r.top, r.height - 12));
          tlProgress.style.height = `${h}px`;
        };
        const onScrollTl = () => {
          if (!ticking) {
            ticking = true;
            requestAnimationFrame(update);
          }
        };
        on(window, "scroll", onScrollTl, { passive: true });
        on(window, "resize", update);
        update();
      }
    }
  }

  /* ---------------- Hero: 3D power-grid network ---------------- */
  const gridCanvas = document.getElementById("grid-canvas") as HTMLCanvasElement | null;
  if (gridCanvas) {
    let renderer: THREE.WebGLRenderer | null = null;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: gridCanvas, antialias: true, alpha: true });
    } catch {
      renderer = null;
    }

    if (renderer) {
      const gridRenderer = renderer;
      gridRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x05080f, 0.016);
      const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 200);
      camera.position.set(0, 2.5, 30);

      const group = new THREE.Group();
      scene.add(group);

      const isMobile = window.innerWidth < 720;
      const NODE_COUNT = isMobile ? 70 : 130;
      const SPREAD_X = 26;
      const SPREAD_Y = 11;
      const SPREAD_Z = 12;

      // node positions: flattened cloud, denser toward the center
      const nodePos: THREE.Vector3[] = [];
      for (let i = 0; i < NODE_COUNT; i++) {
        const rx = Math.random() + Math.random() - 1; // triangular distribution
        const ry = Math.random() + Math.random() - 1;
        const rz = Math.random() + Math.random() - 1;
        nodePos.push(new THREE.Vector3(rx * SPREAD_X, ry * SPREAD_Y, rz * SPREAD_Z));
      }

      // edges: k nearest neighbours, deduped
      const edges: Array<[number, number]> = [];
      const edgeSet = new Set<string>();
      for (let a = 0; a < NODE_COUNT; a++) {
        const dists: Array<[number, number]> = [];
        for (let b = 0; b < NODE_COUNT; b++) {
          if (a === b) continue;
          dists.push([nodePos[a].distanceTo(nodePos[b]), b]);
        }
        dists.sort((u, v) => u[0] - v[0]);
        const k = 2 + (a % 2); // 2 or 3 neighbours
        for (let n = 0; n < k && n < dists.length; n++) {
          const j = dists[n][1];
          if (dists[n][0] > 11) continue;
          const key = `${Math.min(a, j)}_${Math.max(a, j)}`;
          if (!edgeSet.has(key)) {
            edgeSet.add(key);
            edges.push([a, j]);
          }
        }
      }

      const makeGlowTexture = (inner: string, outer: string) => {
        const c = document.createElement("canvas");
        c.width = c.height = 64;
        const g = c.getContext("2d")!;
        const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
        grad.addColorStop(0, inner);
        grad.addColorStop(0.3, outer);
        grad.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = grad;
        g.fillRect(0, 0, 64, 64);
        return new THREE.CanvasTexture(c);
      };
      const cyanGlow = makeGlowTexture("rgba(255,255,255,0.95)", "rgba(34,211,238,0.55)");
      const greenGlow = makeGlowTexture("rgba(255,255,255,0.95)", "rgba(52,211,153,0.6)");

      // edge lines
      const linePositions = new Float32Array(edges.length * 6);
      edges.forEach((e, idx) => {
        const pa = nodePos[e[0]];
        const pb = nodePos[e[1]];
        linePositions.set([pa.x, pa.y, pa.z, pb.x, pb.y, pb.z], idx * 6);
      });
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x2b7f99,
        transparent: true,
        opacity: 0.28,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      group.add(new THREE.LineSegments(lineGeo, lineMat));

      const pointsFrom = (
        indices: number[],
        size: number,
        color: number,
        tex: THREE.Texture,
        opacity: number
      ) => {
        const arr = new Float32Array(indices.length * 3);
        indices.forEach((ni, idx) => {
          arr.set([nodePos[ni].x, nodePos[ni].y, nodePos[ni].z], idx * 3);
        });
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
        const mat = new THREE.PointsMaterial({
          size,
          map: tex,
          color,
          transparent: true,
          opacity,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          sizeAttenuation: true,
        });
        return new THREE.Points(geo, mat);
      };

      const allIdx: number[] = [];
      const subIdx: number[] = [];
      for (let q = 0; q < NODE_COUNT; q++) (q % 9 === 0 ? subIdx : allIdx).push(q);
      group.add(pointsFrom(allIdx, 1.05, 0x67e8f9, cyanGlow, 0.85));
      const substations = pointsFrom(subIdx, 2.4, 0xa5f3fc, cyanGlow, 1.0);
      group.add(substations);

      // energy-flow particles travelling along edges
      const FLOW_COUNT = isMobile ? 60 : 110;
      const flows = Array.from({ length: FLOW_COUNT }, () => ({
        edge: (Math.random() * edges.length) | 0,
        t: Math.random(),
        speed: 0.003 + Math.random() * 0.008,
        dir: Math.random() < 0.5 ? 1 : -1,
      }));
      const flowArr = new Float32Array(FLOW_COUNT * 3);
      const flowGeo = new THREE.BufferGeometry();
      flowGeo.setAttribute("position", new THREE.BufferAttribute(flowArr, 3));
      const flowMat = new THREE.PointsMaterial({
        size: 0.7,
        map: greenGlow,
        color: 0x6ee7b7,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      });
      group.add(new THREE.Points(flowGeo, flowMat));

      const updateFlows = () => {
        for (let f = 0; f < FLOW_COUNT; f++) {
          const fl = flows[f];
          fl.t += fl.speed;
          if (fl.t >= 1) {
            fl.t = 0;
            fl.edge = (Math.random() * edges.length) | 0;
            fl.dir = Math.random() < 0.5 ? 1 : -1;
          }
          const e = edges[fl.edge];
          const pa = nodePos[fl.dir === 1 ? e[0] : e[1]];
          const pb = nodePos[fl.dir === 1 ? e[1] : e[0]];
          flowArr[f * 3] = pa.x + (pb.x - pa.x) * fl.t;
          flowArr[f * 3 + 1] = pa.y + (pb.y - pa.y) * fl.t;
          flowArr[f * 3 + 2] = pa.z + (pb.z - pa.z) * fl.t;
        }
        flowGeo.attributes.position.needsUpdate = true;
      };
      updateFlows();

      const resize = () => {
        const w = hero.clientWidth;
        const h = hero.clientHeight;
        gridRenderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      on(window, "resize", resize);
      resize();

      let targetRX = 0;
      let targetRY = 0;
      if (!isMobile) {
        on(
          window,
          "mousemove",
          (event) => {
            const e = event as MouseEvent;
            targetRY = (e.clientX / window.innerWidth - 0.5) * 0.22;
            targetRX = (e.clientY / window.innerHeight - 0.5) * 0.12;
          },
          { passive: true }
        );
      }

      let running = false;
      let rafId: number | null = null;
      const t0 = performance.now();
      const frameLoop = (now: number) => {
        rafId = null;
        if (!running) return;
        const t = (now - t0) / 1000;
        group.rotation.y += 0.0011;
        group.rotation.x += (targetRX - group.rotation.x) * 0.04;
        group.rotation.z += (targetRY * 0.35 - group.rotation.z) * 0.04;
        group.position.y = Math.sin(t * 0.4) * 0.5;
        substations.material.size = 2.4 + Math.sin(t * 2.2) * 0.35;
        updateFlows();
        gridRenderer.render(scene, camera);
        rafId = requestAnimationFrame(frameLoop);
      };
      const start = () => {
        if (!running) {
          running = true;
          rafId = requestAnimationFrame(frameLoop);
        }
      };
      const stop = () => {
        running = false;
        if (rafId) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
      };

      if (reducedMotion) {
        gridRenderer.render(scene, camera); // one static frame
      } else {
        let heroVisible = true;
        if ("IntersectionObserver" in window) {
          const ho = new IntersectionObserver(
            (entries) => {
              heroVisible = entries[0].isIntersecting;
              if (heroVisible && !document.hidden) start();
              else stop();
            },
            { threshold: 0.02 }
          );
          ho.observe(hero);
          track(() => ho.disconnect());
        }
        on(document, "visibilitychange", () => {
          if (document.hidden) stop();
          else if (heroVisible) start();
        });
        start();
      }

      track(() => {
        stop();
        scene.traverse((obj) => {
          const anyObj = obj as THREE.Mesh;
          if (anyObj.geometry) anyObj.geometry.dispose();
          const mat = anyObj.material as THREE.Material | THREE.Material[] | undefined;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else if (mat) mat.dispose();
        });
        cyanGlow.dispose();
        greenGlow.dispose();
        gridRenderer.dispose();
        gridRenderer.forceContextLoss();
      });
    }
  }

}

document.addEventListener("astro:page-load", initHome);
document.addEventListener("astro:before-swap", teardown);
