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
import { Transformer } from '@napi-rs/image';
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
   (a) Expression: push the creator to a strong, intense, high-conviction
   register — a determined locked stare, knowing/conspiratorial intensity, a
   subtle smirk of "I know something you don't", or a hard jaw-set warning.
   Prefer a CLOSED or composed mouth: determined and intense beats wide-eyed
   open-mouth shock (the dated "shock face" tests 15-20% worse on CTR).
   Believable for a credible finance authority. Never calm, flat, or goofy.
   (b) Trigger: do NOT default to fear alone. For finance and business viewers
   the strongest click driver is usually curiosity + mild indignation +
   self-interest — "there's a legal move the rich use that you don't," "you're
   quietly losing money and don't know it," "a secret that's allowed." Build the
   concept around that insider-edge / forbidden-but-legal / hidden-loss register,
   not just pure alarm.
3. Abstract -> concrete. Finance ideas become ONE physical visual metaphor the
   creator reacts to.
4. Contrast + depth — go CINEMATIC. Light the scene like a film still, not a flat
   product shot: a strong directional key light, real shadows and falloff, with
   clear separation between subject and background so the face and object read
   instantly on mobile. PREFER rich, deep, dark and moody backgrounds — that is
   the default look — but NEVER a flat, pure-black, lifeless void: give the
   background depth with a subtle gradient, soft pools of light, gentle texture or
   atmosphere. A lighter or colored background is fine when a concept is clearly
   stronger that way, but dark-and-cinematic is the default. Motivated colored
   light is welcome when it serves the metaphor (e.g. a red or green glow thrown
   from the object onto the face) — but keep it MOTIVATED and believable, never
   random neon or a gimmicky sticker glow. Dramatic and premium, never plastic,
   garish, or over-stylized. The creator must still read as a real, credible
   authority.
5. Mobile-first. Must read at 120px wide.
6. Curiosity gap. Image + 2-word overlay open a loop the TITLE closes. Tension,
   never a summary.
7. Honor creator_visual_notes if provided.
8. Creator framing (default). Frame the creator LARGE and close — a tight
   head-and-shoulders crop where the face is the single biggest element in the
   frame, filling a generous share of the height (faces get ~42% of viewing time,
   so make it count). Shoot from a slight LOW angle looking up, which makes the creator read as powerful and
   larger-than-life. They are the primary magnet — leave room for the visual
   metaphor, but never shrink the face. They look directly into the lens, locking
   eyes with the viewer. Break eye contact ONLY when a concept is clearly stronger
   with an averted gaze (e.g. looking toward the thing the metaphor depicts).
   Never default to a small, passive, side-of-frame subject. Keep the face clearly
   visible, well-lit and roughly front-facing or three-quarter — avoid extreme
   profile, far-away, heavily shadowed, or partially hidden faces, which break
   the likeness.
9. Number ONLY if the script gives one. If the hook or main_idea provides a
   strong, concrete figure — a real dollar amount, percentage, age, year, or count
   — you MAY feature that single number prominently as a hero element in the scene
   (a sharp real number is one of the strongest click drivers in finance). It must
   be a real figure pulled from the script, never invented, rounded into a fake
   stat, or added just to have a number. If the script offers no strong figure, do
   NOT force one — skip it entirely.

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
  metaphor AND different composition — not three renders of one idea). Vary the
  setting, framing and light direction across them so a run never looks like three
  versions of one image — even though they can share the same rich, dark,
  cinematic tone.
- Fresh with ONE surprising twist, but graspable in 0.3 seconds. Take a
  recognizable object or situation and give it one unexpected turn — not the 50th
  generic "person frowning at a chart", but also NOT an abstract, cryptic puzzle
  or an obscure visual riddle the viewer has to decode. If a normal finance
  viewer wouldn't get it almost instantly, simplify it.
- FAVOR a complete, atmospheric SCENE over a person-plus-object on a bare
  backdrop. Place the creator inside a believable environment with real depth — at
  a desk or table, in a study, office, or room — with foreground, midground and a
  background that has lamps, shelves, furniture or texture. This can absolutely be
  DARK and moody (dark with depth and warm pools of light is ideal), it just must
  not be a flat, empty void. A cleaner background is still fine when a single hero
  object clearly carries the shot, but lean toward the richer, fuller scene.
- PREMIUM hero objects. Whatever prop carries the metaphor should look refined,
  detailed and high-quality (like a finely sculpted porcelain cake-topper) — never
  a cheap, plasticky, or crude AI object.
- Keep the environment SUBORDINATE: softly lit / gently blurred so the face and
  hero object stay sharp and dominant in front. Avoid props the viewer cannot
  place (cracked rock or tree stumps, drifting fog, floating embers/particles,
  glowing cracks, swirling debris).
- Avoid depending on an exact COUNT of objects (e.g. "five envelopes") — image
  models miscount. Use "one red among plain ones" or a single hero object.

THE OVERLAY — a short, emotionally charged phrase, 2 to 4 words (per concept):
- Keep it punchy, but it may run a little longer than a bare label (2 to 4 words)
  when that makes it land harder — e.g. "IT'S NOT WORTH IT", "THIS WILL COST YOU",
  "STOP DOING THIS", "YOU'RE LOSING MONEY". Lay it out across one or two lines —
  however reads best — and it MUST read as one natural, coherent phrase, never
  random or nonsense words.
- EMOTION over label. Do NOT just name the topic ("MARRIAGE TRAP", "JOINT RETURN")
  — make the viewer FEEL something. Choose the emotional register that best fits
  THIS concept, and vary it across the three:
    * warning / urgency  — "BEFORE IT'S TOO LATE", "DON'T DO THIS"
    * personal verdict   — "IT'S NOT WORTH IT", "YOU'RE WRONG ABOUT THIS"
    * reveal / curiosity — "WHAT THEY HIDE", "THE REAL COST"
    * loss / stakes      — "YOU'RE LOSING MONEY", "IT COSTS YOU MORE"
  Pick whichever hits hardest for the concept.
- High-stakes, not loud. The phrase should feel CONSEQUENTIAL — imply a real,
  significant stake (money lost, a costly mistake, something genuinely at risk),
  never a mild observation. Raise the STAKES, not the volume: keep it credible and
  adult, never cartoonish, screaming, or over-hyped.
- Anchor when natural. If you can tie the phrase to the real topic (the tax, the
  IRS, the account, the retiree) without killing the emotion, do so — but a pure
  emotional hook is fine when the image and the title already make the subject
  clear. Either way the viewer must feel they have to click.
- Still implies, never explains — do not state the lesson or spoil the answer.
- Never repeats or summarizes the title, and never reuses a word from the title.
- No creator name, no hashtags. Punctuation only if it adds tension.
- Output the words plain, separated by single spaces only — never a slash, pipe,
  dash, bullet, or any other separator between them.

TEXT IS RENDERED SEPARATELY (added to the image after the scene):
- The caption is added to the image separately, so scene_prompt must contain NO
  text at all, and must keep one corner clean for it.
- For each concept choose:
  * subject_side = the side the creator's face/body occupies, "left" or "right".
  * text_zone   = where the caption goes. Choose the placement that makes the BEST
                  composition and sits in the cleanest empty space — any corner, a
                  side, or the top/bottom center. It does NOT have to be a corner.
                  The ONE hard rule: it must never overlap or crowd the face. Put it
                  in the area away from the creator where there is room to breathe.
  * text_on_dark = true if that corner is dark, false if it is light.
  * text_style  = "marker" (default), "block", or "gold-italic". This now only
                  decides whether a thin underline appears (marker = yes). Use
                  "marker" for most concepts.
  * accent     = the color of the SINGLE highlighted payoff word, chosen by the
                  concept's MEANING (not by style):
      - "red"  : warning, danger, loss, mistake, "stop" / "watch out" concepts.
      - "gold" : insider, secret, money, opportunity, "a legal move the rich use".
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
      "overlay": {"words": "2 TO 4 WORDS — a short emotional phrase", "rationale": "...", "score": 0},
      "subject_side": "<left|right>",
      "text_zone": "<top-left|top-right|bottom-left|bottom-right|top-center|bottom-center>",
      "text_on_dark": true,
      "text_style": "<marker|block|gold-italic>",
      "accent": "<red|gold>",
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

// ── render one concept: 1 GPT call (scene + baked-in caption) -> upload ──────
async function renderConcept(runId, concept, refFiles, creatorName) {
  // GPT renders the caption itself: strip the scene's "no text" rule, then append a text directive
  const scene = String(concept.scene_prompt || '').replace(/do not render any text[^.]*\.?/gi, '').trim();
  const identity =
    'IDENTITY LOCK: keep the person\u2019s face, hairline, hair (same amount, length, color and style), ' +
    'glasses, facial hair and apparent age EXACTLY consistent with the supplied reference photos. ' +
    'Do NOT thin, shorten, recolor, add, or remove hair, and do NOT make the person look balder, younger, ' +
    'or older. Also keep his OUTFIT the same as in the reference photos — the same jacket/suit, shirt, ' +
    'color and style of clothing — do not change his clothes or their color. It must be unmistakably the ' +
    'same exact person, dressed as in the references.';
  const quality =
    'QUALITY: sharp focus, professional studio photography quality, crisp fine detail, clean and ' +
    'polished. Keep skin and textures natural and realistic — NOT plastic, waxy, airbrushed, or over-smoothed.';
  const pngB64 = await gptImage(refFiles, `${scene}\n\n${identity}\n\n${quality}\n\n${buildTextDirective(concept)}`, creatorName);
  const finalBuffer = Buffer.from(pngB64, 'base64');

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

// ── tell GPT to render the 2-word caption itself, in the chosen brand style ──
function buildTextDirective(concept) {
  const words = String(concept.overlay?.words || '').replace(/[\/|]+/g, ' ').replace(/\s+/g, ' ').trim().toUpperCase();
  if (!words) return 'Render no text anywhere in the image.';
  // underline swoosh is ALWAYS red, regardless of concept meaning
  const accent = 'bright red (#E11D2A)';
  return [
    'TEXT OVERLAY — render this caption baked into the image in a FIXED, consistent brand style (render it the SAME way every time):',
    `Render the exact caption "${words}" all uppercase, arranged across one or two lines — break the words wherever it reads best and forms a balanced, punchy block — in an ULTRA-HEAVY, extra-bold, wide condensed sans-serif (Anton / Archivo Black style): very thick, fat strokes and broad heavy letterforms that fill the space, with normal comfortable letter spacing (not cramped, not stretched).`,
    'PLACEMENT: position the caption yourself in whichever area of THIS image has the cleanest, largest empty space (e.g. an open dark area or negative space) — wherever it looks best and is most readable. It must never overlap or crowd the face or the hero object; put it where there is room to breathe.',
    'Render EVERY word in clean pure white — no colored words.',
    'Give each letter a VERY THIN, subtle black outline — almost just a soft crisp edge, not a heavy keyline. Do NOT use a thick block outline, a filled box or rectangle behind the letters, or a heavy border. Add ONE small, soft drop shadow directly behind the text for depth — subtle, never a thick glow, halo, or box.',
    `Beneath the final line, ALWAYS add a single red underline — ALMOST straight, with only a very slight, subtle curve — in ${accent}. It must be of EVEN, UNIFORM thickness from end to end: do NOT taper it (not thick in the middle and thin at the tips). A clean, smooth, even stroke, NOT a thick brush smear and NOT a solid bar. This underline must ALWAYS be present.`,
    'Keep the caption COMPACT: it should occupy only about a quarter of the frame, sitting neatly in its area with clear margins from every edge. Large enough to read instantly on mobile, but it must NOT dominate the image, span the full width, or crowd the edges. Crisp, perfectly legible, correctly spelled, with NO extra, missing, or misspelled words. Keep it fully clear of the person\u2019s face and body. This is the ONLY text anywhere in the image.',
  ].join(' ');
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

  // Normalize whatever GPT returned (webp / odd PNG / etc.) into a clean PNG
  // that @napi-rs/canvas can definitely decode. If it isn't an image at all,
  // the decode throws and we get a clear reason instead of "Invalid SVG".
  const raw = Buffer.from(b64, 'base64');
  const magic = raw.slice(0, 8).toString('hex');
  console.log('[gptImage] in:', raw.length, 'bytes, magic:', magic);
  let pngBuf;
  try {
    pngBuf = await new Transformer(raw).png();
  } catch (e) {
    const head = raw.slice(0, 120).toString('utf8').replace(/\s+/g, ' ');
    throw new Error(`GPT Image: could not decode response (magic=${magic}, ${raw.length} bytes): ${e?.message || e} | head="${head}"`);
  }
  return pngBuf.toString('base64');
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
