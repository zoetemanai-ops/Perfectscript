// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (Nano Banana Pro edition)
//
// Trigger:  POST { script_run_id }
//   1. Loads the script run, creates a `thumbnail_runs` row (status: pending)
//   2. Responds IMMEDIATELY with the new row id
//   3. In the background (waitUntil): Claude art director -> 3 concepts ->
//      Nano Banana Pro (scene render + 2-word text pass) -> upload -> row complete
//
// The frontend polls `thumbnail_runs` for status === 'complete'.
//
// ── Prerequisites ────────────────────────────────────────────────────────────
//   npm i @anthropic-ai/sdk @google/genai @supabase/supabase-js @vercel/functions
//
//   Env vars (Vercel project settings, all server-side):
//     ANTHROPIC_API_KEY
//     GEMINI_API_KEY
//     SUPABASE_URL
//     SUPABASE_SERVICE_ROLE_KEY        (service/secret key — bypasses RLS, reads storage)
//
//   Storage buckets:
//     creator-photos   (PRIVATE)  ->  creator-photos/<client_slug>/photo1.jpg ... (3-6 per creator)
//     thumbnails       (PUBLIC)   ->  finished thumbnails land here
//
//   maxDuration 300 needs Vercel Pro.
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI, Modality } from '@google/genai';
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
const ART_DIRECTOR_MODEL = 'claude-opus-4-8';     // the brain
const IMAGE_MODEL = 'gemini-3-pro-image-preview';  // Nano Banana Pro
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

TEXT IS A SEPARATE PASS:
- scene_prompt describes the image with NO overlay text in it (scene + face
  lock only). The 2 words are applied afterward.
- text_pass is the instruction for the second step that adds the 2 words.

AVOID (AI-slop tells): cluttered scenes, multiple focal points, generic stock
look, over-saturation, plastic skin, gibberish text, extra logos/watermarks,
any text in the scene beyond what text_pass adds.

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
      "composition": "<focal point, rule-of-thirds, fg/bg, where the negative space for text sits>",
      "color_and_lighting": "...",
      "overlay": {"words": "TWO WORDS", "rationale": "...", "score": 0},
      "freshness_score": 0,
      "click_score": 0,
      "scene_prompt": "<ONE paragraph, text-FREE, for Nano Banana Pro: 16:9 photorealistic thumbnail; describe the creator WITH an explicit instruction to keep face/identity exactly consistent with the supplied reference images, same person; the metaphor; the hot-but-credible emotion; composition; lighting; end with 'Leave the upper area clear for a text overlay. Do not add any text, logo, or watermark. One clear focal point, readable as a small mobile thumbnail.'>",
      "text_pass": "<instruction for the second NBP edit call: add the exact 2 words, heavy bold condensed sans, color, thick outline, placement, 'render legibly with exact spelling, no other text added.'>"
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

    // c) creator reference photos (face lock)
    const refParts = await loadReferencePhotos(script.client_slug);

    // d) render each concept in parallel (scene render -> text pass -> upload)
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

// ── render one concept (2 sequential Banana calls) ───────────────────────────
async function renderConcept(runId, concept, refParts) {
  // 1) scene — text-free, creator faces locked via reference images
  const sceneB64 = await bananaImage([...refParts, { text: concept.scene_prompt }]);

  // 2) add the 2 words as an edit pass on the scene
  const finalB64 = await bananaImage([
    { inlineData: { mimeType: 'image/png', data: sceneB64 } },
    { text: concept.text_pass },
  ]);

  const path = `${runId}/${concept.id}.png`;
  const { error: upErr } = await supabase.storage
    .from(OUT_BUCKET)
    .upload(path, Buffer.from(finalB64, 'base64'), { contentType: 'image/png', upsert: true });
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

// ── Nano Banana Pro -> base64 PNG ────────────────────────────────────────────
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

// ── load 3-6 creator reference photos from private storage ───────────────────
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
      const b64 = Buffer.from(await data.arrayBuffer()).toString('base64');
      const lower = f.name.toLowerCase();
      const mime = lower.endsWith('.png')
        ? 'image/png'
        : lower.endsWith('.webp')
        ? 'image/webp'
        : 'image/jpeg';
      return { inlineData: { mimeType: mime, data: b64 } };
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
