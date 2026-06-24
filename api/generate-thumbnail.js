// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (GPT Image edition)
//
// Trigger:  POST { script_run_id }
//   1. Loads the script run, creates a `thumbnail_runs` row (status: pending)
//   2. Responds IMMEDIATELY with the new row id
//   3. In the background (waitUntil): Claude art director -> 3 concepts ->
//      GPT Image (1 call per concept, creator faces as references, 2 words baked in)
//      -> crop to 16:9 -> upload -> row complete
//
// The frontend polls `thumbnail_runs` for status === 'complete'.
//
// ── Prerequisites ────────────────────────────────────────────────────────────
//   npm i @anthropic-ai/sdk openai sharp @supabase/supabase-js @vercel/functions
//
//   Env vars (Vercel project settings, all server-side):
//     ANTHROPIC_API_KEY
//     OPENAI_API_KEY
//     SUPABASE_URL
//     SUPABASE_SERVICE_ROLE_KEY        (service/secret key — bypasses RLS, reads storage)
//
//   Storage buckets:
//     creator-photos   (PRIVATE)  ->  creator-photos/<client_slug>/photo1.jpg ... (3-6 per creator)
//     thumbnails       (PUBLIC)   ->  finished thumbnails land here
//
//   GPT Image has no native 16:9 — we request 1536x1024 and center-crop to 16:9.
//   maxDuration 300 needs Vercel Pro.
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from '@anthropic-ai/sdk';
import OpenAI, { toFile } from 'openai';
import sharp from 'sharp';
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
const ART_DIRECTOR_MODEL = 'claude-opus-4-8';   // the brain (stays on Claude)
const IMAGE_MODEL = 'gpt-image-1';               // newest is 'gpt-image-2' (May 2026, better) — use it if your account has API access; gpt-image-1 is the reliable GA fallback
const IMAGE_SIZE = '1536x1024';                  // landscape; cropped to 16:9 after
const IMAGE_QUALITY = 'high';                    // 'high' | 'medium' | 'low'
const REF_BUCKET = 'creator-photos';
const OUT_BUCKET = 'thumbnails';

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
2. Emotion runs HOT — it earns the click. Push the creator's expression to a
   strong, dramatic, high-arousal register: intense locked stare, jaw-set
   alarm, genuine shock, urgent warning. Keep it believable for a credible
   finance authority — steer only ~5% away from the literal cartoon tells
   (perfectly symmetrical bug-eyes, open-mouth scream, eyebrows pinned at max).
   Intensity stays near the top. Do NOT make it calm or flat.
3. Abstract -> concrete. Finance ideas become ONE physical visual metaphor the
   creator reacts to.
4. Contrast + depth. Sharp subject, desaturated/blurred background, warm rim
   light, complementary colors that pop in a feed.
5. Mobile-first. Must read at 120px wide.
6. Curiosity gap. Image + 2-word overlay open a loop the TITLE closes. Tension,
   never a summary.
7. Honor creator_visual_notes if provided.

ANTI-CLICHÉ (do this FIRST):
- List the 4-6 most overused thumbnail images for THIS exact topic — the ones a
  viewer has already seen on 50 other finance videos.
- You may NOT use any of them as a primary metaphor.
- The three concepts must be genuinely different from each other (different
  metaphor AND different composition — not three renders of one idea).
- At least ONE concept must take an oblique, unexpected angle competitors would
  not think of. Reward yourself for the idea that is NOT the first thing that
  came to mind.

THE 2-WORD OVERLAY (per concept):
- Exactly two words. Short, punchy.
- ADDS intrigue — never repeats/summarizes the title, never reuses a title word.
- No creator name, no hashtags. Punctuation only if it adds tension.
- It implies, threatens, or teases. It does not explain.
- The two words are baked directly into the scene_prompt (see below).

AVOID (AI-slop tells): cluttered scenes, multiple focal points, generic stock
look, over-saturation, plastic skin, gibberish text, extra logos/watermarks,
any text beyond the two chosen words.

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
      "composition": "<focal point, rule-of-thirds, fg/bg, where the 2-word text sits>",
      "color_and_lighting": "...",
      "overlay": {"words": "TWO WORDS", "rationale": "...", "score": 0},
      "freshness_score": 0,
      "click_score": 0,
      "scene_prompt": "<ONE paragraph for the image model: 16:9 photorealistic YouTube thumbnail; describe the creator WITH an explicit instruction to keep face and identity exactly consistent with the supplied reference images, same person; the metaphor; the hot-but-credible emotion; composition; lighting. THEN bake in the exact two overlay words as large heavy condensed bold sans-serif text, bright color with a thick contrasting outline, placed in the negative space, rendered cleanly and legibly with exact correct spelling. End with 'Do not add any other text, logo, or watermark. One clear focal point, readable as a small mobile thumbnail.'>"
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

  // Confirm these column names match your script_runs table.
  // We derive the "hook" from the opening of full_script (no separate hook column).
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

    // a) creator context
    const { data: client } = await supabase
      .from('client_profiles')
      .select('client_name, thumbnail_visual_notes')
      .eq('client_slug', script.client_slug)
      .single();

    const creatorName = client?.client_name || script.client_slug;
    const visualNotes = client?.thumbnail_visual_notes || '';
    const hook = (script.full_script || '').slice(0, 800);

    // b) art director (the brain) -> 3 concepts
    const brief = await runArtDirector({
      creator_name: creatorName,
      creator_visual_notes: visualNotes,
      video_title: script.video_title || '',
      hook,
      main_idea: script.main_idea || '',
    });

    const concepts = (brief.concepts || []).filter((c) => c?.scene_prompt).slice(0, 3);
    if (!concepts.length) throw new Error('Art director returned no usable concepts');

    // c) creator reference photos (face lock) as OpenAI files
    const refFiles = await loadReferencePhotos(script.client_slug);

    // d) render each concept in parallel (1 GPT call -> crop 16:9 -> upload)
    const images = await Promise.all(concepts.map((c) => renderConcept(runId, c, refFiles)));

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

// ── render one concept: single GPT Image call -> crop -> upload ──────────────
async function renderConcept(runId, concept, refFiles) {
  const b64 = refFiles.length
    ? // with creator references -> images.edit (multi-image input)
      (await openai.images.edit({
        model: IMAGE_MODEL,
        image: refFiles,
        prompt: concept.scene_prompt,
        size: IMAGE_SIZE,
        quality: IMAGE_QUALITY,
      })).data?.[0]?.b64_json
    : // no references yet -> plain generate (no face lock)
      (await openai.images.generate({
        model: IMAGE_MODEL,
        prompt: concept.scene_prompt,
        size: IMAGE_SIZE,
        quality: IMAGE_QUALITY,
      })).data?.[0]?.b64_json;

  if (!b64) throw new Error('GPT Image returned no image (possible content refusal)');

  // crop center to 16:9
  const cropped = await cropTo16x9(Buffer.from(b64, 'base64'));

  const path = `${runId}/${concept.id}.png`;
  const { error: upErr } = await supabase.storage
    .from(OUT_BUCKET)
    .upload(path, cropped, { contentType: 'image/png', upsert: true });
  if (upErr) throw new Error(`Upload failed: ${upErr.message}`);

  const { data: pub } = supabase.storage.from(OUT_BUCKET).getPublicUrl(path);

  return {
    concept_id: concept.id,
    url: pub.publicUrl,
    overlay_words: concept.overlay?.words || '',
    angle: concept.angle || '',
    score: null, // filled later by CleanCut
  };
}

// ── center-crop a buffer to 16:9, return a PNG buffer ────────────────────────
async function cropTo16x9(buffer) {
  const img = sharp(buffer);
  const meta = await img.metadata();
  const targetH = Math.round((meta.width * 9) / 16);
  const top = Math.max(0, Math.round((meta.height - targetH) / 2));
  return img
    .extract({ left: 0, top, width: meta.width, height: Math.min(targetH, meta.height) })
    .png()
    .toBuffer();
}

// ── load 3-6 creator reference photos from private storage as OpenAI files ───
async function loadReferencePhotos(clientSlug) {
  const { data: files, error } = await supabase.storage.from(REF_BUCKET).list(clientSlug);
  if (error || !files?.length) return []; // no photos yet -> still renders, just no face lock
  const usable = files.filter((f) => /\.(jpe?g|png|webp)$/i.test(f.name)).slice(0, 6);

  return Promise.all(
    usable.map(async (f) => {
      const { data, error: dErr } = await supabase.storage
        .from(REF_BUCKET)
        .download(`${clientSlug}/${f.name}`);
      if (dErr) throw new Error(`Reference photo download failed: ${dErr.message}`);
      const buf = Buffer.from(await data.arrayBuffer());
      const lower = f.name.toLowerCase();
      const type = lower.endsWith('.png')
        ? 'image/png'
        : lower.endsWith('.webp')
        ? 'image/webp'
        : 'image/jpeg';
      return toFile(buf, f.name, { type });
    })
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────
function update(runId, patch) {
  return supabase.from('thumbnail_runs').update(patch).eq('id', runId);
}
function stripFences(s) {
  return s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
}
