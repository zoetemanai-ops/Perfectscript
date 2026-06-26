// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (GPT Image 2 + text layer)
//
// Trigger:  POST { script_run_id }
//   Background (waitUntil): Claude art director -> 3 concepts -> GPT Image 2
//   renders the SCENE (no text, 1 call/concept, creator NEVER named) -> we draw
//   the 2 words as a typographic layer (Oswald, 3 styles) -> upload -> complete.
//
// Text styles (art director picks one per concept):
//   marker      — sharp outlined word + tapered brush underline (default)
//   block       — second word on an accent bar (for hard / alarm concepts)
//   gold-italic — second word slanted in gold (for insider / "secret" concepts)
//
// ── Prerequisites ────────────────────────────────────────────────────────────
//   npm i @anthropic-ai/sdk openai @napi-rs/canvas @supabase/supabase-js @vercel/functions
//   Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
//   Buckets: creator-photos (PRIVATE, <client_slug>/*.jpg), thumbnails (PUBLIC)
//   maxDuration 300 needs Vercel Pro.
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from '@anthropic-ai/sdk';
import OpenAI, { toFile } from 'openai';
import { createCanvas, loadImage, GlobalFonts } from '@napi-rs/canvas';
import { createClient } from '@supabase/supabase-js';
import { waitUntil } from '@vercel/functions';

export const config = { maxDuration: 300 };

// ── clients ──────────────────────────────────────────────────────────────────
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// ── config ───────────────────────────────────────────────────────────────────
const ART_DIRECTOR_MODEL = 'claude-opus-4-8';
const IMAGE_MODEL = 'gpt-image-2';
const IMAGE_SIZE = '1536x864';   // 16:9 (both divisible by 16)
const IMAGE_QUALITY = 'high';    // if the edits endpoint rejects this, remove the quality line in gptImage()
const REF_BUCKET = 'creator-photos';
const OUT_BUCKET = 'thumbnails';

// text layer — Oswald 700 (static woff2 via @fontsource on a CDN)
const FONT_URLS = [
  'https://cdn.jsdelivr.net/npm/@fontsource/oswald/files/oswald-latin-700-normal.woff2',
  'https://unpkg.com/@fontsource/oswald/files/oswald-latin-700-normal.woff2',
];
const FONT_NAME = 'Oswald700';
const ACCENT_HEX = '#E11D2A';   // red — used by marker underline + block bar
const GOLD_HEX = '#F4C430';     // gold — used by gold-italic style

const ART_DIRECTOR_SYSTEM = `SYSTEM — "Perfect Thumbnail · Art Director v2"

You are the art director for a YouTube thumbnail factory serving top finance
and business creators. You read a video script and output THREE genuinely
distinct thumbnail concepts, score them, and pick the strongest. Each must stop
the scroll on a mobile feed and earn the click.

INPUTS (JSON):
- creator_name
- creator_visual_notes   (recurring brand look — may be empty)
- video_title
- hook
- main_idea

NON-NEGOTIABLE PRINCIPLES:
1. ONE idea, ONE focal point per concept. The eye lands in under 0.3s.
2. Emotion + trigger run HOT — they earn the click. TWO layers:
   (a) Expression: push the creator to a strong, dramatic, high-arousal register
   — intense locked stare, jaw-set alarm, genuine shock, knowing/conspiratorial
   intensity, or urgent warning. Believable for a credible finance authority —
   steer only ~5% from the literal cartoon tells (perfectly symmetrical
   bug-eyes, open-mouth scream, eyebrows pinned at max). Never calm or flat.
   (b) Trigger: do NOT default to fear alone. For finance and business viewers
   the strongest click driver is usually curiosity + mild indignation +
   self-interest — "there's a legal move the rich use that you don't," "you're
   quietly losing money and don't know it," "a secret that's allowed." Build the
   concept around that insider-edge / forbidden-but-legal / hidden-loss register,
   not just pure alarm.
3. Abstract -> concrete. Finance ideas become ONE physical visual metaphor the
   creator reacts to.
4. Contrast + depth. Sharp subject, desaturated/blurred background, warm rim
   light, complementary colors that pop in a feed.
5. Mobile-first. Must read at 120px wide.
6. Curiosity gap. Image + 2-word overlay open a loop the TITLE closes. Tension,
   never a summary.
7. Honor creator_visual_notes if provided.
8. Creator framing (default). Frame the creator chest-up (head and shoulders),
   large and dominant — the primary magnet — while still leaving room for the
   visual metaphor. They look directly into the lens, locking eyes with the
   viewer. Break eye contact ONLY when a concept is clearly stronger with an
   averted gaze (e.g. looking toward the thing the metaphor depicts). Never
   default to a small, passive, side-of-frame subject. Keep the face clearly
   visible, well-lit and roughly front-facing or three-quarter — avoid extreme
   profile, far-away, heavily shadowed, or partially hidden faces, which break
   the likeness.

TITLE + THUMBNAIL = ONE HOOK:
On YouTube the viewer always sees the thumbnail and the title TOGETHER — they
are one weapon, not two. Do NOT let the thumbnail resolve the curiosity on its
own. Design each concept as the OTHER HALF of the video_title: the title states
one thing, the thumbnail shows what the title leaves unsaid, and only together
do they form an unanswered question that demands the click. Use the title as an
active partner — complete it, contradict it, or raise the stakes on it — never
merely avoid repeating its words.

ANTI-CLICHÉ (do this FIRST):
- List the 4-6 most overused thumbnail images for THIS exact topic — the ones a
  viewer has already seen on 50 other finance videos.
- You may NOT use any of them as a primary metaphor.
- The three concepts must be genuinely different from each other (different
  metaphor AND different composition — not three renders of one idea).
- At least ONE concept must take an oblique, unexpected angle competitors would
  not think of. Reward yourself for the idea that is NOT the first thing that
  came to mind.
- Avoid depending on an exact COUNT of objects (e.g. "five envelopes") — image
  models miscount. Use "one red among plain ones" or a single hero object.

THE OVERLAY — 2 words, occasionally 3 (per concept):
- Two words by default, short and punchy. Add a third ONLY if needed to keep the
  phrase natural; never more than three. They stack on two lines and MUST read as
  a coherent, natural micro-phrase (e.g. "IRS / TRAP", "RETIRE / BROKE",
  "TAX BREAK / BAIT", "GONE / LEGALLY"). Never pair words that read as nonsense
  (e.g. "ONE / TRICKS").
- ANCHOR + TENSION. One part must make the SUBJECT recognizable at a glance —
  name or clearly evoke the real topic (the tax, the account, the rule, the
  retiree, the agency, the benefit) — and one part must create tension or
  intrigue. The viewer should know roughly WHAT it is about AND feel they must
  click to learn the catch. Avoid purely abstract metaphor words that could fit
  any video (e.g. "WRONG MOVE", "DON'T OPEN", "PEEL BACK").
- Still implies, never explains — do not state the lesson or spoil the answer.
- Never repeats or summarizes the title, and never reuses a word from the title.
- No creator name, no hashtags. Punctuation only if it adds tension.
- Output the words plain, separated by single spaces only — never a slash, pipe,
  dash, bullet, or any other separator between them.

TEXT IS RENDERED SEPARATELY (by code, not by you):
- The 2 words are drawn onto the image afterwards as a typographic layer.
- So scene_prompt must contain NO text at all, and must keep one corner clean.
- For each concept choose:
  * subject_side = the side the creator's face/body occupies, "left" or "right".
  * text_zone   = a corner on the OPPOSITE horizontal side from subject_side, so
                  the text never lands on the face. If the face is on the right,
                  the text goes left, and vice versa. Pick top or bottom by where
                  the emptiest space is.
  * text_on_dark = true if that corner is dark, false if it is light.
  * text_style  = one of:
      - "marker"      : default. Outlined word + a brush underline. Use for most
                        concepts.
      - "block"       : the second word on a solid accent bar. Use for HARD,
                        urgent, alarm / "stop" concepts.
      - "gold-italic" : the second word slanted in gold. Use ONLY for the
                        insider / secret / "a legal move the rich use" angle.
- In scene_prompt, place the creator on subject_side and explicitly keep the
  opposite side (the text_zone side) clean, low-detail and even-toned.

AVOID (AI-slop tells): cluttered scenes, multiple focal points, generic stock
look, over-saturation, plastic skin, gibberish text, extra logos/watermarks,
any text in the image at all.

OUTPUT — return ONLY valid JSON, no preamble:
{
  "topic_read": "<one line: what actually makes this clickable>",
  "cliches_banned": ["<overused image>", "<overused image>"],
  "concepts": [
    {
      "id": "A",
      "angle": "<the distinct direction in a phrase>",
      "visual_metaphor": "...",
      "subject_direction": "<expression (hot but credible), pose, gesture, framing, placement>",
      "composition": "<focal point, rule-of-thirds, fg/bg, which corner is the clean text_zone>",
      "color_and_lighting": "...",
      "overlay": {"words": "TWO OR THREE WORDS", "rationale": "...", "score": 0},
      "subject_side": "<left|right>",
      "text_zone": "<top-left|top-right|bottom-left|bottom-right|top-center|bottom-center>",
      "text_on_dark": true,
      "text_style": "<marker|block|gold-italic>",
      "freshness_score": 0,
      "click_score": 0,
      "scene_prompt": "<ONE paragraph, ABSOLUTELY NO TEXT, for Nano Banana Pro: 16:9 photorealistic thumbnail; describe the creator WITH an explicit instruction to keep face and identity exactly consistent with the supplied reference images, same person; the metaphor; the hot-but-credible emotion; framing chest-up with eye contact by default; composition; lighting. Keep the chosen text_zone corner clean, low-detail and even-toned so a text overlay reads on top. End with 'Do not render any text, letters, numbers, words, logos, or watermarks anywhere in the image. One clear focal point, readable as a small mobile thumbnail.'>"
    },
    { "id": "B" },
    { "id": "C" }
  ],
  "recommended": "<A|B|C>",
  "why_recommended": "<one line>"
}

Be decisive — no hedging. The recommended concept maximizes click_score while
keeping freshness_score >= 7.`;

// ── handler ──────────────────────────────────────────────────────────────────
export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { script_run_id } = req.body || {};
  if (!script_run_id) return res.status(400).json({ error: 'script_run_id is required' });

  const { data: script, error: scriptErr } = await supabase
    .from('script_runs')
    .select('id, client_slug, video_title, main_idea, full_script')
    .eq('id', script_run_id)
    .single();
  if (scriptErr || !script) return res.status(404).json({ error: 'script_run not found' });

  const { data: run, error: runErr } = await supabase
    .from('thumbnail_runs')
    .insert({
      script_run_id: script.id,
      client_slug: script.client_slug,
      status: 'pending',
      video_title: script.video_title,
      model: IMAGE_MODEL,
    })
    .select('id')
    .single();
  if (runErr || !run) return res.status(500).json({ error: 'Could not create thumbnail run' });

  res.status(202).json({ thumbnail_run_id: run.id, status: 'pending' });
  waitUntil(generate(run.id, script));
}

// ── background pipeline ──────────────────────────────────────────────────────
async function generate(runId, script) {
  try {
    await update(runId, { status: 'generating' });

    const { data: client } = await supabase
      .from('client_profiles')
      .select('client_name, thumbnail_visual_notes')
      .eq('client_slug', script.client_slug)
      .single();

    const creatorName = client?.client_name || script.client_slug;
    const visualNotes = client?.thumbnail_visual_notes || '';
    const hook = (script.full_script || '').slice(0, 800);

    const brief = await runArtDirector({
      creator_name: creatorName,
      creator_visual_notes: visualNotes,
      video_title: script.video_title || '',
      hook,
      main_idea: script.main_idea || '',
    });

    const concepts = (brief.concepts || []).filter((c) => c?.scene_prompt).slice(0, 3);
    if (!concepts.length) throw new Error('Art director returned no usable concepts');

    const refFiles = await loadReferencePhotos(script.client_slug);
    const images = await Promise.all(concepts.map((c) => renderConcept(runId, c, refFiles, creatorName)));

    await update(runId, {
      status: 'complete',
      art_director_json: brief,
      images,
      recommended_concept: brief.recommended || concepts[0].id,
    });
  } catch (err) {
    console.error('[generate-thumbnail]', err);
    await update(runId, { status: 'error', error: String(err?.message || err) });
  }
}

// ── art director ─────────────────────────────────────────────────────────────
async function runArtDirector(input) {
  const msg = await anthropic.messages.create({
    model: ART_DIRECTOR_MODEL,
    max_tokens: 4000,
    system: ART_DIRECTOR_SYSTEM,
    messages: [{ role: 'user', content: JSON.stringify(input, null, 2) }],
  });
  const text = msg.content.map((b) => (b.type === 'text' ? b.text : '')).join('').trim();
  return JSON.parse(stripFences(text));
}

// ── render one concept: 1 Banana call (scene) -> draw 2 words -> upload ──────
async function renderConcept(runId, concept, refFiles, creatorName) {
  const sceneB64 = await gptImage(refFiles, concept.scene_prompt, creatorName);

  const finalBuffer = await compositeText(Buffer.from(sceneB64, 'base64'), {
    words: concept.overlay?.words || '',
    zone: resolveZone(concept.text_zone, concept.subject_side),
    onDark: concept.text_on_dark !== false,
    style: concept.text_style || 'marker',
  });

  const path = `${runId}/${concept.id}.png`;
  const { error: upErr } = await supabase.storage
    .from(OUT_BUCKET)
    .upload(path, finalBuffer, { contentType: 'image/png', upsert: true });
  if (upErr) throw new Error(`Upload failed: ${upErr.message}`);

  const { data: pub } = supabase.storage.from(OUT_BUCKET).getPublicUrl(path);

  return {
    concept_id: concept.id,
    url: pub.publicUrl,
    overlay_words: concept.overlay?.words || '',
    text_zone: concept.text_zone || 'top-left',
    text_style: concept.text_style || 'marker',
    angle: concept.angle || '',
    score: null, // filled later by CleanCut
  };
}

// ── GPT Image 2 -> base64 image (creator name kept OUT of the prompt) ─────────
async function gptImage(refFiles, scenePrompt, creatorName) {
  const prompt = sanitizePrompt(scenePrompt, creatorName);
  let res;
  try {
    res = await openai.images.edit({
      model: IMAGE_MODEL,
      image: refFiles,
      prompt,
      size: IMAGE_SIZE,
      quality: IMAGE_QUALITY,
      output_format: 'png',
    });
  } catch (e) {
    // surface the real reason (moderation_blocked, param error, etc.)
    const code = e?.code || e?.error?.code || e?.status;
    const msg = e?.error?.message || e?.message || String(e);
    throw new Error(`GPT Image request failed [${code}]: ${msg}`);
  }

  const item = res?.data?.[0] || {};
  console.log('[gptImage] keys:', Object.keys(item), '| b64?', !!item.b64_json, '| url?', !!item.url);

  let b64 = item.b64_json;
  // some responses return a URL instead of base64 — fetch it and convert
  if (!b64 && item.url) {
    const r = await fetch(item.url);
    b64 = Buffer.from(await r.arrayBuffer()).toString('base64');
  }
  if (!b64) {
    throw new Error(`GPT Image: no image field. data[0]=${JSON.stringify(item).slice(0, 300)}`);
  }

  // strip a data-URI prefix if the API included one (otherwise it decodes to garbage)
  b64 = String(b64).replace(/^data:image\/\w+;base64,/, '');

  const buf = Buffer.from(b64, 'base64');
  const magic = buf.slice(0, 8).toString('hex');
  // canvas in this build reliably decodes PNG and JPEG; webp/others fall through to the SVG error
  const isImg =
    buf.slice(0, 4).toString('hex') === '89504e47' ||      // png
    buf.slice(0, 3).toString('hex') === 'ffd8ff';           // jpeg
  console.log('[gptImage] bytes:', buf.length, '| magic:', magic, '| isImg:', isImg);
  if (!isImg) {
    throw new Error(`GPT Image: not PNG/JPEG (magic=${magic}, ${buf.length} bytes) — canvas can't decode it.`);
  }
  return b64;
}

// Strip the creator's name (full name and first/last parts) from the scene prompt
// so the request never names a public figure — GPT relies on the reference photos.
function sanitizePrompt(scenePrompt, creatorName) {
  let p = String(scenePrompt || '');
  const parts = String(creatorName || '').trim().split(/\s+/).filter((s) => s.length > 2);
  const variants = [creatorName, ...parts].filter((s) => s && s.length > 2);
  for (const v of variants) {
    p = p.replace(new RegExp(`\\b${escapeRegExp(v)}\\b`, 'gi'), 'the person in the reference photo');
  }
  return p;
}
function escapeRegExp(s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

// ── TEXT LAYER ───────────────────────────────────────────────────────────────
let _fontReady = false;
async function ensureFont() {
  if (_fontReady) return;
  let lastErr;
  for (const url of FONT_URLS) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      GlobalFonts.register(Buffer.from(await res.arrayBuffer()), FONT_NAME);
      _fontReady = true;
      return;
    } catch (e) { lastErr = e; }
  }
  throw new Error('Font fetch failed: ' + (lastErr?.message || 'all CDNs'));
}

async function compositeText(sceneBuffer, { words, zone, onDark, style }) {
  await ensureFont();
  const img = await loadImage(sceneBuffer);
  const W = img.width, H = img.height;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, W, H);

  const tokens = cleanWords(words).split(/\s+/).filter(Boolean).map((t) => t.toUpperCase());
  if (!tokens.length) return canvas.toBuffer('image/png');
  const [line1, line2] = splitBalanced(ctx, tokens);

  const margin = Math.round(W * 0.05);
  let size = Math.round(H * 0.20);
  const widest = () => {
    ctx.font = `${size}px ${FONT_NAME}`;
    return Math.max(ctx.measureText(line1).width, line2 ? ctx.measureText(line2).width : 0);
  };
  while (widest() > W * 0.52 && size > 28) size -= 2;
  ctx.font = `${size}px ${FONT_NAME}`;

  const gap = size * 1.02;
  const nLines = line2 ? 2 : 1;
  const isBottom = zone.startsWith('bottom');
  const isRight = zone.endsWith('right');
  const isCenter = zone.endsWith('center');

  let align, x;
  if (isCenter) { align = 'center'; x = Math.round(W / 2); }
  else if (isRight) { align = 'right'; x = W - margin; }
  else { align = 'left'; x = margin; }

  let by1;
  if (isBottom) {
    const by2 = H - margin - size * 0.20;
    by1 = by2 - (nLines - 1) * gap;
  } else {
    by1 = margin + size * 0.78;
  }
  const by2 = by1 + gap;

  // for the gold style, pick line-1 color from the ACTUAL background brightness
  // where it sits (white on dark, dark on light) — the "It's Not" look.
  const line1Dark = style === 'gold-italic'
    ? measureOnDark(ctx, line1, x, by1, size, align, W, H)
    : onDark;

  // line 1 is always the plain outlined word
  outlineText(ctx, line1, x, by1, size, line1Dark, align);

  if (!line2) return canvas.toBuffer('image/png');

  if (style === 'block') {
    drawBar(ctx, line2, x, by2, size, align, ACCENT_HEX);
  } else if (style === 'gold-italic') {
    italicWord(ctx, line2, x, by2, size, align, GOLD_HEX);
  } else {
    // marker (default)
    outlineText(ctx, line2, x, by2, size, onDark, align);
    drawMarker(ctx, line2, x, by2, size, align, ACCENT_HEX);
  }
  return canvas.toBuffer('image/png');
}

function outlineText(ctx, text, x, by, size, onDark, align) {
  ctx.font = `${size}px ${FONT_NAME}`;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  ctx.save();
  ctx.lineJoin = 'round'; ctx.miterLimit = 2;
  ctx.lineWidth = size * 0.085;
  ctx.strokeStyle = onDark ? '#0a0a0a' : '#ffffff';
  ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = size * 0.10; ctx.shadowOffsetY = size * 0.045;
  ctx.strokeText(text, x, by);
  ctx.restore();
  ctx.fillStyle = onDark ? '#ffffff' : '#141414';
  ctx.fillText(text, x, by);
}

// sample the scene brightness where a line of text will sit -> true if dark (use white text)
function measureOnDark(ctx, text, x, by, size, align, W, H) {
  ctx.font = `${size}px ${FONT_NAME}`;
  const tw = ctx.measureText(text).width;
  let sx;
  if (align === 'right') sx = x - tw;
  else if (align === 'center') sx = x - tw / 2;
  else sx = x;
  const rx = Math.max(0, Math.floor(sx));
  const ry = Math.max(0, Math.floor(by - size * 0.72));
  const rw = Math.min(W - rx, Math.ceil(tw));
  const rh = Math.min(H - ry, Math.ceil(size * 0.8));
  if (rw < 2 || rh < 2) return true;
  try {
    const data = ctx.getImageData(rx, ry, rw, rh).data;
    let sum = 0, n = 0;
    for (let i = 0; i < data.length; i += 16) {
      sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      n++;
    }
    return (sum / n / 255) < 0.58; // dark bg -> white text
  } catch (e) {
    return true;
  }
}

function drawBar(ctx, text, x, by, size, align, color) {
  ctx.font = `${size}px ${FONT_NAME}`;
  ctx.textBaseline = 'alphabetic';
  const m = ctx.measureText(text);
  const tw = m.width, asc = m.actualBoundingBoxAscent, desc = m.actualBoundingBoxDescent;
  const padX = size * 0.16, padY = size * 0.13;
  let barLeft, textX, textAlign;
  if (align === 'right') { barLeft = x - (tw + padX * 2); textX = x - padX; textAlign = 'right'; }
  else if (align === 'center') { barLeft = x - (tw / 2 + padX); textX = x; textAlign = 'center'; }
  else { barLeft = x; textX = x + padX; textAlign = 'left'; }
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.4)'; ctx.shadowBlur = size * 0.10; ctx.shadowOffsetY = size * 0.03;
  roundRect(ctx, barLeft, by - asc - padY, tw + padX * 2, asc + desc + padY * 2, size * 0.07);
  ctx.fillStyle = color; ctx.fill();
  ctx.restore();
  ctx.textAlign = textAlign;
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, textX, by);
}

function italicWord(ctx, text, x, by, size, align, color) {
  ctx.font = `${size}px ${FONT_NAME}`;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  const tw = ctx.measureText(text).width;
  let ox = x;
  if (align === 'right') ox = x - tw;
  else if (align === 'center') ox = x - tw / 2;
  ox -= size * 0.06; // compensate the italic slant so it lines up under line 1
  ctx.save();
  ctx.translate(ox, by);
  ctx.transform(1, 0, -0.22, 1, 0, 0);
  ctx.lineJoin = 'round'; ctx.miterLimit = 2; ctx.lineWidth = size * 0.085;
  ctx.strokeStyle = '#3a2c00';
  ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = size * 0.10; ctx.shadowOffsetY = size * 0.045;
  ctx.strokeText(text, 0, 0);
  ctx.shadowColor = 'transparent';
  ctx.fillStyle = color; ctx.fillText(text, 0, 0);
  ctx.restore();
}

function drawMarker(ctx, text, x, by, size, align, color) {
  ctx.font = `${size}px ${FONT_NAME}`;
  ctx.textAlign = align;
  const tw = ctx.measureText(text).width;
  let left;
  if (align === 'left') left = x;
  else if (align === 'right') left = x - tw;
  else left = x - tw / 2;
  const x0 = left - size * 0.05, x1 = left + tw + size * 0.05;
  const yc = by + size * 0.18, thick = size * 0.16, w = x1 - x0, tilt = w * 0.035;
  const n = 26, pts = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const cx = x0 + w * t;
    const cy = yc + tilt * t + Math.sin(t * Math.PI * 1.25) * thick * 0.12;
    let prof;
    if (t < 0.07) prof = t / 0.07;                      // fine left tip
    else if (t > 0.88) prof = Math.max((1 - t) / 0.12, 0) * 0.7; // wispy right flick
    else prof = 1 - Math.abs(t - 0.42) * 0.22;          // slight mid bulge
    pts.push([cx, cy, thick * 0.5 * Math.max(prof, 0.03)]);
  }
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.3)'; ctx.shadowBlur = thick * 0.35; ctx.shadowOffsetY = thick * 0.15;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1] - pts[0][2]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1] - pts[i][2]);
  for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(pts[i][0], pts[i][1] + pts[i][2]);
  ctx.closePath();
  ctx.fillStyle = color; ctx.fill();
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// choose the line break that balances the two lines (minimizes the wider line so
// the text renders as large as possible; tiebreak on the most even split)
function splitBalanced(ctx, tokens) {
  if (tokens.length <= 1) return [tokens[0] || '', ''];
  ctx.font = `100px ${FONT_NAME}`;
  let best = null;
  for (let i = 1; i < tokens.length; i++) {
    const a = tokens.slice(0, i).join(' ');
    const b = tokens.slice(i).join(' ');
    const wa = ctx.measureText(a).width, wb = ctx.measureText(b).width;
    const score = Math.max(wa, wb) * 1000 + Math.abs(wa - wb);
    if (!best || score < best.score) best = { a, b, score };
  }
  return [best.a, best.b];
}

// ── load 3-6 creator reference photos from private storage ───────────────────
async function loadReferencePhotos(clientSlug) {
  const { data: files, error } = await supabase.storage.from(REF_BUCKET).list(clientSlug);
  if (error || !files?.length) return [];
  const usable = files.filter((f) => /\.(jpe?g|png|webp)$/i.test(f.name)).slice(0, 4);
  return Promise.all(
    usable.map(async (f) => {
      const { data, error: dErr } = await supabase.storage
        .from(REF_BUCKET)
        .download(`${clientSlug}/${f.name}`);
      if (dErr) throw new Error(`Reference photo download failed: ${dErr.message}`);
      const buf = Buffer.from(await data.arrayBuffer());
      const lower = f.name.toLowerCase();
      const mime = lower.endsWith('.png') ? 'image/png' : lower.endsWith('.webp') ? 'image/webp' : 'image/jpeg';
      return toFile(buf, f.name, { type: mime });
    })
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────
// Force the text to the side opposite the creator's face. If the art director
// put the text on the same horizontal side as the subject, flip it.
function resolveZone(zone, subjectSide) {
  const z = String(zone || 'top-left').toLowerCase();
  const vert = z.startsWith('bottom') ? 'bottom' : 'top';
  let horiz = z.endsWith('right') ? 'right' : z.endsWith('center') ? 'center' : 'left';
  const side = String(subjectSide || '').toLowerCase();
  if (side === 'left' && horiz === 'left') horiz = 'right';
  else if (side === 'right' && horiz === 'right') horiz = 'left';
  return `${vert}-${horiz}`;
}
function cleanWords(s) {
  return String(s)
    .replace(/[\/\\|]+/g, ' ')     // slashes / pipes -> space
    .replace(/\s[-–—]\s/g, ' ')    // standalone dashes between words -> space
    .replace(/\s+/g, ' ')
    .trim();
}
function update(runId, patch) {
  return supabase.from('thumbnail_runs').update(patch).eq('id', runId);
}
function stripFences(s) {
  return s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
}
