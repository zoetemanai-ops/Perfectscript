// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (Nano Banana Pro + text layer)
//
// Trigger:  POST { script_run_id }
//   Background (waitUntil): Claude art director -> 3 concepts -> Nano Banana Pro
//   renders the SCENE (no text, 1 call/concept) -> we draw the 2 words as a
//   typographic layer (Oswald, 3 styles) -> upload -> row complete.
//
// Text styles (art director picks one per concept):
//   marker      — sharp outlined word + tapered brush underline (default)
//   block       — second word on an accent bar (for hard / alarm concepts)
//   gold-italic — second word slanted in gold (for insider / "secret" concepts)
//
// ── Prerequisites ────────────────────────────────────────────────────────────
//   npm i @anthropic-ai/sdk @google/genai @napi-rs/canvas @supabase/supabase-js @vercel/functions
//   Env: ANTHROPIC_API_KEY, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
//   Buckets: creator-photos (PRIVATE, <client_slug>/*.jpg), thumbnails (PUBLIC)
//   maxDuration 300 needs Vercel Pro.
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI, Modality } from '@google/genai';
import { createCanvas, loadImage, GlobalFonts } from '@napi-rs/canvas';
import { createClient } from '@supabase/supabase-js';
import { waitUntil } from '@vercel/functions';

export const config = { maxDuration: 300 };

// ── clients ──────────────────────────────────────────────────────────────────
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// ── config ───────────────────────────────────────────────────────────────────
const ART_DIRECTOR_MODEL = 'claude-opus-4-8';
const IMAGE_MODEL = 'gemini-3-pro-image-preview';
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

THE 2-WORD OVERLAY (per concept):
- Exactly two words. Short, punchy. They are stacked on two lines, so they MUST
  read as a coherent, natural micro-phrase when stacked (e.g. "DON'T / OPEN",
  "CLOSING / SOON", "GONE / LEGALLY"). Never pair two words that read as
  nonsense together (e.g. "ONE / TRICKS").
- ADDS intrigue — never repeats/summarizes the title, never reuses a title word.
- No creator name, no hashtags. Punctuation only if it adds tension.
- Output the two words plain, separated by ONE space only — never put a slash,
  pipe, dash, bullet, or any other separator character between them.
- It implies, threatens, or teases. It does not explain.

TEXT IS RENDERED SEPARATELY (by code, not by you):
- The 2 words are drawn onto the image afterwards as a typographic layer.
- So scene_prompt must contain NO text at all, and must keep one corner clean.
- For each concept choose:
  * text_zone   = the corner with the best empty negative space (away from the
                  face and the metaphor's focal point).
  * text_on_dark = true if that corner is dark, false if it is light.
  * text_style  = one of:
      - "marker"      : default. Outlined word + a brush underline. Use for most
                        concepts.
      - "block"       : the second word on a solid accent bar. Use for HARD,
                        urgent, alarm / "stop" concepts.
      - "gold-italic" : the second word slanted in gold. Use ONLY for the
                        insider / secret / "a legal move the rich use" angle.
- Compose so the chosen text_zone stays clean, low-detail and even-toned.

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
      "overlay": {"words": "TWO WORDS", "rationale": "...", "score": 0},
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

    const refParts = await loadReferencePhotos(script.client_slug);
    const images = await Promise.all(concepts.map((c) => renderConcept(runId, c, refParts)));

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
async function renderConcept(runId, concept, refParts) {
  const sceneB64 = await bananaImage([...refParts, { text: concept.scene_prompt }]);

  const finalBuffer = await compositeText(Buffer.from(sceneB64, 'base64'), {
    words: concept.overlay?.words || '',
    zone: concept.text_zone || 'top-left',
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

// ── Nano Banana Pro -> base64 image ──────────────────────────────────────────
async function bananaImage(parts) {
  const response = await genai.models.generateContent({
    model: IMAGE_MODEL,
    contents: [{ role: 'user', parts }],
    config: { responseModalities: [Modality.TEXT, Modality.IMAGE] },
  });
  const out = response.candidates?.[0]?.content?.parts || [];
  const img = out.find((p) => p.inlineData?.data);
  if (!img) throw new Error('Nano Banana returned no image (possible safety block)');
  return img.inlineData.data;
}

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

  const tokens = cleanWords(words).split(/\s+/).filter(Boolean);
  if (!tokens.length) return canvas.toBuffer('image/png');
  const line1 = tokens[0].toUpperCase();
  const line2 = tokens.slice(1).join(' ').toUpperCase(); // '' if a single word

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

  // line 1 is always the plain outlined word
  outlineText(ctx, line1, x, by1, size, onDark, align);

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
  ox -= size * 0.085; // compensate the italic slant so it lines up under line 1
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

// ── load 3-6 creator reference photos from private storage ───────────────────
async function loadReferencePhotos(clientSlug) {
  const { data: files, error } = await supabase.storage.from(REF_BUCKET).list(clientSlug);
  if (error || !files?.length) return [];
  const usable = files.filter((f) => /\.(jpe?g|png|webp)$/i.test(f.name)).slice(0, 6);
  return Promise.all(
    usable.map(async (f) => {
      const { data, error: dErr } = await supabase.storage
        .from(REF_BUCKET)
        .download(`${clientSlug}/${f.name}`);
      if (dErr) throw new Error(`Reference photo download failed: ${dErr.message}`);
      const b64 = Buffer.from(await data.arrayBuffer()).toString('base64');
      const lower = f.name.toLowerCase();
      const mime = lower.endsWith('.png') ? 'image/png' : lower.endsWith('.webp') ? 'image/webp' : 'image/jpeg';
      return { inlineData: { mimeType: mime, data: b64 } };
    })
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────
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
