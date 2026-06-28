// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (GPT Image 2)
//
// Trigger:  POST { script_run_id }
//   Background (waitUntil): Claude art director -> 3 concepts -> GPT Image 2
//   renders each concept in ONE call (scene + baked-in caption, creator NEVER
//   named) -> upload -> complete. The caption is rendered by the image model
//   itself via buildTextDirective; there is no separate canvas text layer.
//
// ── Prerequisites ────────────────────────────────────────────────────────────
//   npm i @anthropic-ai/sdk openai @napi-rs/image @supabase/supabase-js @vercel/functions
//   Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
//   Buckets: creator-photos (PRIVATE, <client_slug>/*.jpg), thumbnails (PUBLIC)
//   maxDuration 300 needs Vercel Pro.
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from '@anthropic-ai/sdk';
import OpenAI, { toFile } from 'openai';
import { Transformer } from '@napi-rs/image';
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
1. ONE idea, ONE dominant focal point per concept; the eye lands in under 0.3s,
   and that focal point is the creator's face. (Opposites-split may use two
   contrasting background worlds, but the face still dominates and the two halves
   stay clearly subordinate, never competing focal points.)
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
3. Topic-legible ALWAYS; a concrete metaphor when the archetype calls for one.
   Whatever the archetype, a cold viewer who knows nothing about the video must be
   able to guess the SUBJECT (that this is about taxes, retirement, your money, the
   IRS, an account, etc.) from the image alone in under 0.3s. In metaphor-driven
   concepts, finance ideas become ONE physical visual metaphor the creator reacts
   to: not just any concrete object. A premium but subject-blind prop (a generic
   vault, a plain door, a blank envelope) FAILS this even when it looks great: if
   the object would fit fifty unrelated finance videos, it is too abstract. Tie the
   metaphor to the ACTUAL mechanism in the script (the specific tax, the account,
   the deadline, the retiree), then give THAT one fresh twist. Held-document carries
   the subject through a short real label on a prop instead of a metaphor, and that
   is fine.
4. Natural, realistic light and real depth — it must look like a genuine
   PHOTOGRAPH, not a stylized or graded render. Light the creator cleanly and
   believably so the face reads instantly on mobile, with real separation between
   subject and background. The background brightness and mood are FREE and should
   VARY across the three concepts: bright, neutral, or darker, whichever genuinely
   fits — there is no single default, so do not make every thumbnail dark, and do
   not wash every thumbnail in the same warm golden grade. Warm, dark or colored
   light is welcome when something real in the scene motivates it (an actual lamp,
   a window, a screen) or when a concept is clearly stronger that way; just don't
   apply it on autopilot to all three. "Premium" here means clean, sharp and real,
   NOT over-graded or plastic. The creator must read as a real, credible authority
   captured on a real camera.
5. Mobile-first. Must read at 120px wide.
6. Curiosity gap. Image + 2-word overlay open a loop the TITLE closes. Tension,
   never a summary.
7. creator_visual_notes = the creator's REAL set. If creator_visual_notes is
   provided, it describes the creator's actual recording office/studio (backdrop,
   furniture, shelves, books, plants, lighting). Whenever a concept is set in the
   creator's own office or studio, reproduce THAT set as closely as possible so the
   thumbnail matches their real channel look. For concepts deliberately set
   elsewhere (a different location the metaphor needs), you do not have to use it,
   but still respect any brand colors or styling it mentions. If it is empty, design
   the environment freely.
8. Creator framing (default). Frame the creator LARGE and close — a tight
   head-and-shoulders crop where the face is the single biggest element in the
   frame, filling a generous share of the height (faces get ~42% of viewing time,
   so make it count). Shoot from a slight LOW angle looking up, which makes the creator read as powerful and
   larger-than-life. They are the primary magnet — leave room for the visual
   metaphor, but never shrink the face. They look directly into the lens, locking
   eyes with the viewer. Break eye contact ONLY when a concept is clearly stronger
   with an averted gaze (e.g. looking toward the thing the metaphor depicts).
   Never default to a small, passive, side-of-frame subject. (Opposites-split may
   place the creator off-center beside the split, but the face must still be LARGE
   and dominant, never shrunk.) Keep the face clearly
   visible, well-lit and roughly front-facing or three-quarter — avoid extreme
   profile, far-away, heavily shadowed, or partially hidden faces, which break
   the likeness.
9. Real numbers only, and never as a giant standalone graphic. If the video_title,
   hook or main_idea gives a strong concrete figure (a real dollar amount,
   percentage, age, year, or count such as "2026"), it MAY appear naturally as part
   of the scene — for example printed on the held document, a screen, or a sign —
   when it strengthens the concept. It must be a real figure pulled from the script,
   never invented, rounded into a fake stat, or added just to have a number. Do NOT
   build a thumbnail around one enormous floating 3D number; keep any figure small,
   real, and physically part of the scene. If there is no strong figure, skip it.

TITLE + THUMBNAIL = ONE HOOK:
On YouTube the viewer always sees the thumbnail and the title TOGETHER — they
are one weapon, not two. Do NOT let the thumbnail resolve the curiosity on its
own. Design each concept as the OTHER HALF of the video_title: the title states
one thing, the thumbnail shows what the title leaves unsaid, and only together
do they form an unanswered question that demands the click. Use the title as an
active partner — complete it, contradict it, or raise the stakes on it — never
merely avoid repeating its words.

STYLE LIBRARY — STEP 1, design one concept in EACH of the THREE archetypes:
This factory has THREE thumbnail archetypes. They are EQUALS: none is the "house
style" and none is a default. Every run uses all three, ONE concept per archetype,
so a run always shows three genuinely different formats.

OVERRIDE RULE: each archetype's recipe below sets its own composition, background
and text handling. Where a recipe conflicts with a general principle or with the
ANTI-CLICHÉ scene preference, the ARCHETYPE wins for that concept. Everything else
always applies to every archetype: hot-but-credible emotion, the creator's face
large and identity-locked, topic-legibility in under 0.3s, mobile readability at
120px, title+thumbnail = one hook, natural realistic photography, and no tired
executions.

1) METAPHOR-PORTRAIT
   ONE topic-true physical metaphor the creator reacts to, inside a believable
   real environment with natural depth (study / office / room / desk). Light it
   like a clean, well-lit real photograph with real subject/background separation.
   Creator chest-up, eye contact. No text inside the scene; leave clean, even-toned
   negative space for the caption.

2) OPPOSITES-SPLIT
   The background splits into TWO contrasting, topic-true worlds that show the
   stakes (e.g. a struggling side vs a prosperous side, before vs after, what you
   lose vs what you keep). Creator centered or slightly to one side, reacting to the
   divide. This archetype BENDS "one focal point": the split is deliberate, but the
   creator's face must stay the single dominant element and each half must be
   instantly readable and clearly subordinate, never two competing focal points.
   Distinguish the two halves by real, believable means (different setting, props,
   or natural light level), not by drenching one side in golden glow. Both worlds
   concrete and graspable in 0.3s. No text in the scene; leave space for the caption.

3) HELD-DOCUMENT (exposé look)
   The creator holds a real prop toward the lens — a form, letter, folder, card or
   sign — and reacts with conviction. The prop carries ONE very short label, printed
   like a real letterhead or header on the document at NATURAL document scale (not as
   huge hero text): 1 to 3 words MAX naming the real mechanism from the script (e.g.
   "TAX DEDUCTION NOTICE", "IRS NOTICE", "BOX 3"). That label is the ONLY legible
   in-scene text. Do NOT print a large year or big number on the paper; if a year or
   figure appears at all it must be small, inside the letterhead or a date field,
   never large hero text. Any other body text on the prop must be small, blurred,
   unreadable filler, never legible sentences (the image model garbles long text).
   Keep the face large and the label crisp and correctly spelled. The paper must look
   REAL: genuinely gripped by the fingers with a slight bend or curl, a soft shadow
   where the hand holds it and where it falls against the body, visible paper texture
   and a subtle sheen under the room light — never a flat, pasted-on white rectangle.

In each concept's "angle", say how that archetype is made to fit THIS script.

ANTI-CLICHÉ (run this for EVERY concept, after choosing its archetype):
- List the 4-6 most overused thumbnail EXECUTIONS for THIS exact topic — the tired
  literal renders a viewer has already seen on 50 other finance videos (e.g. a
  plain calculator on a desk, a generic 1040 form, a piggy bank, stacked coins).
- You may NOT reuse those tired executions as-is. But do NOT flee the topic
  entirely: a tax video SHOULD still read as being about tax, a retirement video
  about retirement. The goal is a topic-TRUE object given ONE fresh, unexpected
  turn — never a premium but unrelated object (vault, door, blank envelope) that
  only escapes the cliché list by saying nothing about the subject. Reinvent the
  topic, do not abandon it.
- The three concepts must be genuinely different from each other. HARD RULE: they
  MUST use three DIFFERENT archetypes (see STYLE LIBRARY) and must also differ in
  background and composition, so a run never looks like three versions of one
  image. Within whatever archetype mix you pick, do NOT repeat the same underlying
  metaphor family — e.g. do not submit two "open vs closing portal" ideas (a door
  AND a vault AND an envelope are the same idea three times). If two concepts feel
  like variants, replace one with a different archetype.
- Fresh with ONE surprising twist, but graspable in 0.3 seconds. Take a
  recognizable object or situation and give it one unexpected turn — not the 50th
  generic "person frowning at a chart", but also NOT an abstract, cryptic puzzle
  or an obscure visual riddle the viewer has to decode. If a normal finance
  viewer wouldn't get it almost instantly, simplify it.
- FAVOR a complete, believable SCENE over a person-plus-object on a bare backdrop
  (this is the metaphor-portrait baseline; opposites-split and held-document set
  their own backgrounds per the STYLE LIBRARY). Place the creator inside a real
  environment with depth — at a desk or table, in a study, office, or room — with
  foreground, midground and a background that has lamps, shelves, furniture or
  texture. Light it naturally and cleanly; it must not be a flat, empty void, and
  it must not be the same dark or golden look every time — vary the brightness and
  mood across the three. A cleaner background is fine when a single hero object
  carries the shot, but lean toward the richer, fuller scene.
- REAL, INTERESTING props. Whatever object carries the metaphor must look like a
  genuine physical thing photographed in the room: real materials with real texture,
  weight, edge wear and a natural contact shadow — never a generic, plasticky or
  crude AI object, and never a flat, boring default (a plain blank sheet, a generic
  paper stack). Make the prop SPECIFIC and a little unexpected: a real, detailed,
  characterful object the eye wants to look at, with believable small details that
  sell it as photographed, not generated.
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

TEXT IN THE IMAGE:
- The 2-4 word caption is rendered onto the image by the image model in a FIXED
  brand style (you do NOT choose its font or color). Your job is composition: in
  scene_prompt, leave one area of clean, even-toned, low-detail negative space,
  away from the face, where that caption can sit and read clearly.
- The other two archetypes carry NO text in the scene. The ONLY exception is
  held-document: one short label on the prop (and, if the script gives one, a small
  real figure printed on that same prop). Describe that single short element in
  scene_prompt and keep it short and correctly spelled, never full sentences or
  paragraphs of legible text.

AVOID (AI-slop tells): cluttered scenes, multiple competing focal points, generic
stock look, over-saturation, plastic skin, gibberish or misspelled text, extra
logos/watermarks, and any legible text beyond the caption plus (for held-document
only) the one short label/figure on the prop.

OUTPUT — return ONLY valid JSON, no preamble:
{
  "topic_read": "<one line: what actually makes this clickable>",
  "cliches_banned": ["<overused execution>", "<overused execution>"],
  "concepts": [
    {
      "id": "A",
      "archetype": "<metaphor-portrait|opposites-split|held-document>",
      "angle": "<the distinct direction in a phrase, and why this archetype fits>",
      "visual_metaphor": "...",
      "subject_direction": "<expression (hot but credible), pose, gesture, framing, placement>",
      "composition": "<focal point, rule-of-thirds, fg/bg, and where the clean negative space for the caption sits>",
      "color_and_lighting": "...",
      "overlay": {"words": "2 TO 4 WORDS — a short emotional phrase", "rationale": "...", "score": 0},
      "freshness_score": 0,
      "click_score": 0,
      "scene_prompt": "<ONE paragraph for the image model: 16:9 photorealistic thumbnail that looks like a real photo, not a render; describe the creator WITH an explicit instruction to keep face and identity exactly consistent with the supplied reference images, same person; the archetype scene and/or metaphor; the hot-but-credible emotion; framing chest-up with eye contact by default; composition; natural realistic lighting (varied brightness/mood across the three concepts, not the same dark or golden look every time). Leave clean, even-toned, low-detail negative space away from the face for the caption that gets added on top. ONLY for held-document: describe ONE short label on the prop (1-3 words), plus a small real figure on it if the script gives one. End with 'One clear focal point, readable as a small mobile thumbnail.'>"
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
    'PHOTOGRAPHIC REALISM (this must look like a real photo taken on a real camera, NOT a CGI render or an AI image): ' +
    'shot on a full-frame camera with an 85mm lens at a wide aperture (~f1.8), giving a genuinely SHALLOW depth of field — ' +
    'only the face is tack-sharp while the background falls into soft, natural bokeh (do NOT keep the whole frame equally sharp). ' +
    'Add the subtle imperfections of a real photograph: fine, natural film grain / sensor noise throughout, real skin texture ' +
    'with visible pores, fine lines and slight unevenness (NEVER plastic, waxy, airbrushed, over-smoothed or glossy CGI skin), ' +
    'a faint hint of chromatic aberration at the edges, and natural, slightly uneven directional lighting with real falloff and ' +
    'soft, believable catchlights in the eyes (not glassy or over-bright). ' +
    'ONE REAL CAPTURE, as if a real photographer came to the creator\u2019s actual office and shot this in one frame: the person, the background ' +
    'and every prop were photographed together in the same room, lit by the SAME light from the same direction, sharing one consistent white ' +
    'balance, exposure and color grade — so nothing looks cut out, pasted, stickered, or composited. Every object and the person cast a real, soft ' +
    'CONTACT SHADOW where they meet a surface, hand or wall; foreground, subject and background sit in the same believable space and atmosphere, ' +
    'with no too-clean cut-out edges and no element that looks floated on top. ' +
    'Keep it clean, sharp and natural like a real editorial portrait photograph, ' +
    'NOT a heavily stylized, over-graded or CGI look (the scene lighting itself is set by the scene description); ' +
    'it must read as captured, not generated: avoid a flawless, over-clean, perfectly symmetrical studio look. ' +
    'The photographic grain and softness apply to the scene only; the caption stays crisp and clean.';
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
    archetype: concept.archetype || '',
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
    'Keep the caption COMPACT: it should occupy only about a quarter of the frame, sitting neatly in its area with clear margins from every edge. Large enough to read instantly on mobile, but it must NOT dominate the image, span the full width, or crowd the edges. Crisp, perfectly legible, correctly spelled, with NO extra, missing, or misspelled words. Keep it fully clear of the person\u2019s face and body. Apart from this caption, the ONLY other text permitted is a SINGLE short label on a prop if the scene description explicitly calls for one (e.g. "TAX DEDUCTION NOTICE", "IRS NOTICE", "BOX 3", 1 to 3 words) — render it at a natural, modest size like real printed text on the document, NOT as large hero text, and never a big standalone year or number. Render it cleanly, legibly and correctly spelled where the scene places it; do NOT add full sentences, paragraphs, large floating numbers, or any other text. No other text, letters, words, logos, or watermarks anywhere.',
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
  // that @napi-rs/image can definitely decode. If it isn't an image at all,
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
function update(runId, patch) {
  return supabase.from('thumbnail_runs').update(patch).eq('id', runId);
}
function stripFences(s) {
  return s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
}
