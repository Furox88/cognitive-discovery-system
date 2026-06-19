// Render assets/promo_hero.html frame-by-frame to PNG via Playwright.
// Each frame is set deterministically via window.CDS_PROMO.setFrame(i),
// then screenshotted. Pillow (scripts/build_gif.py) assembles the GIF.
//
// Usage:
//   node scripts/render_promo.mjs
//   node scripts/render_promo.mjs --frames 120        # override frame count
//   node scripts/render_promo.mjs --only 0,45,89      # render specific frames
//   node scripts/render_promo.mjs --out myframe       # single-shot to myframe.png
//
// Output: assets/_promo_frames/frame_000.png ... frame_NNN.png

import { chromium } from 'playwright';
import { existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const HTML = join(ROOT, 'assets', 'promo_hero.html');
const OUTDIR = join(ROOT, 'assets', '_promo_frames');

// --- arg parsing ---
const args = process.argv.slice(2);
function arg(name) {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : null;
}
const onlyArg = arg('only');
const singleOut = arg('out');
let total = 90;
{
  const f = arg('frames');
  if (f) total = parseInt(f, 10);
}

const frames =
  onlyArg
    ? onlyArg.split(',').map(n => parseInt(n, 10))
    : Array.from({ length: total }, (_, i) => i);

async function main() {
  if (!existsSync(OUTDIR)) mkdirSync(OUTDIR, { recursive: true });

  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 2,            // crisp 2x for better GIF quality
  });
  const page = await ctx.newPage();

  page.on('console', m => {
    if (m.type() === 'error') console.error('[page error]', m.text());
  });

  await page.goto('file:///' + HTML.replace(/\\/g, '/'));
  // wait for the expose + fonts
  await page.waitForFunction(() => window.CDS_PROMO && typeof window.CDS_PROMO.setFrame === 'function');
  await page.waitForTimeout(300);

  // read authoritative total from the page CONFIG (keep JS the source of truth)
  const cfgTotal = await page.evaluate(() => window.CDS_PROMO.CONFIG.totalFrames);
  if (!onlyArg && !arg('frames')) total = cfgTotal;

  const list = onlyArg ? frames : Array.from({ length: total }, (_, i) => i);

  const t0 = Date.now();
  for (let k = 0; k < list.length; k++) {
    const i = list[k];
    await page.evaluate(n => window.CDS_PROMO.setFrame(n), i);
    // let layout/fonts settle for this frame
    await page.waitForTimeout(40);

    let outPath;
    if (singleOut && list.length === 1) {
      outPath = join(OUTDIR, '..', singleOut.endsWith('.png') ? singleOut : singleOut + '.png');
    } else {
      outPath = join(OUTDIR, `frame_${String(i).padStart(3, '0')}.png`);
    }
    await page.screenshot({ path: outPath, omitBackground: false });
    if ((k + 1) % 10 === 0 || k === list.length - 1) {
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      process.stdout.write(`  rendered ${k + 1}/${list.length} frames (${dt}s)\n`);
    }
  }

  await browser.close();
  console.log(`\nDone. ${list.length} frame(s) in ${OUTDIR}`);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
