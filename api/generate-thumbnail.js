// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (GPT Image 2)
//
// Trigger:  POST { script_run_id }
//   Background (waitUntil): Claude art director -> 3 concepts -> GPT Image 2
//   renders each scene TEXT-FREE (creator NEVER named) -> the 2-4 word caption is
//   composited afterwards in the exact brand font via @napi-rs/canvas (pixel-
//   perfect typography, zero misspellings, optional highlight word in red or in a
//   red block) -> upload -> complete. The art director picks a caption_zone; a
//   detail-score check relocates the caption if the render put the face or hero
//   object there. bottom-right is never used (YouTube's duration badge).
//
// ── Prerequisites ────────────────────────────────────────────────────────────
//   npm i @anthropic-ai/sdk openai @napi-rs/image @napi-rs/canvas @supabase/supabase-js @vercel/functions
//   Font: api/fonts/LibreFranklin-Black.ttf (referenced via new URL(), so
//   Vercel's file tracer bundles it automatically with the function).
//   Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
//   Buckets: creator-photos (PRIVATE, <client_slug>/*.jpg), thumbnails (PUBLIC)
//   maxDuration 300 needs Vercel Pro.
// ─────────────────────────────────────────────────────────────────────────────

import Anthropic from '@anthropic-ai/sdk';
import OpenAI, { toFile } from 'openai';
import { Transformer } from '@napi-rs/image';
import { createCanvas, loadImage, GlobalFonts } from '@napi-rs/canvas';
import { fileURLToPath } from 'node:url';
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

const ART_DIRECTOR_SYSTEM = `SYSTEM — "Perfect Thumbnail · Art Director v3"

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
   Match the expression to THIS concept's register: a loss, threat or warning concept
   needs a serious, hard or grave face — NO smile or pleased look; a subtle smirk
   fits ONLY the insider-edge register, never a loss or seizure scene.
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
   the deadline, the retiree), then give THAT one fresh twist. Opposites-split,
   reaction-to-object and on-location can carry the subject through contrasting
   worlds, a real object, or a real place instead of a single metaphor, and
   evidence-closeup carries it through a real marked document; all of that is fine.
4. Natural, realistic light and real depth — it must look like a genuine
   PHOTOGRAPH, not a stylized or graded render. Light the creator cleanly and
   believably so the face reads instantly on mobile, with real separation between
   subject and background. The background brightness and mood are FREE and should
   VARY across the three concepts: bright, neutral, or darker, whichever genuinely
   fits — there is no single default, so do not apply the same look on autopilot
   to all three. Warm, dark or colored light is welcome when something real in the
   scene motivates it (an actual lamp, a window, a screen) or when a concept is
   clearly stronger that way. "Premium" here means clean, sharp and real,
   NOT over-graded or plastic. The creator must read as a real, credible authority
   captured on a real camera.
5. Mobile-first. Must read at 120px wide.
6. Curiosity gap. Image + 2-4 word overlay open a loop the TITLE closes. Tension,
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
   of the scene — for example on a real document, a screen, or a sign already in the
   shot — when it strengthens the concept. It must be a real figure pulled from the script,
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

STYLE LIBRARY — the pool is FOUR archetypes; TWO are mandatory every run:
This factory has FOUR thumbnail archetypes. Every run MUST include exactly these
two, each as its own concept: opposites-split and reaction-to-object. The THIRD
concept is your FREE pick — score evidence-closeup and on-location for THIS
script and build a concept in whichever fits best. Result: 3 concepts across three
different archetypes (the two mandatory ones + your best free pick). Mandatory does
NOT mean generic: make each archetype genuinely earn its place for THIS specific
script, never a bolted-on template.

OVERRIDE RULE: each archetype's recipe below sets its own composition, background
and text handling. Where a recipe conflicts with a general principle or with the
ANTI-CLICHÉ scene preference, the ARCHETYPE wins for that concept. Everything else
always applies to every archetype: hot-but-credible emotion, the creator's face
large and identity-locked, topic-legibility in under 0.3s, mobile readability at
120px, title+thumbnail = one hook, natural realistic photography, and no tired
executions.

1) EVIDENCE CLOSE-UP
   Creator large beside ONE real, instantly recognizable document or screen that
   carries the PROOF — a tax bill, bank statement, official letter, contract, a
   retirement-account or brokerage screen — with EXACTLY ONE detail marked: a
   hand-drawn red circle or a marker highlight around one real figure (a dollar
   amount, percentage, age or year) pulled from the script. "Look what's sitting
   HERE" is the hook: the marked figure opens the loop the title closes. Rules:
   - The marked figure is the ONLY legible text on the document (2-10 characters,
     e.g. "$48,000", "37%"), rendered large, crisp and correctly spelled inside
     the mark; ALL other print on the page stays soft-focus, blurred and
     illegible — never readable sentences or paragraphs.
   - The figure must be REAL, pulled from video_title / hook / main_idea
     (Principle 9 applies in full); if the script offers no strong concrete
     figure, this archetype does not fit — score it low.
   - The document reads as what it is through LAYOUT and physical cues (official
     letterhead shape, a table grid, an envelope it came from, a screen UI), not
     through readable words.
   - The creator reacts to the marked detail — pointing at it, holding the page,
     or locking eyes with the viewer while presenting it; face stays dominant per
     Principle 8, the document angled toward the lens so the mark reads at 120px.

2) OPPOSITES-SPLIT
   The background splits into TWO contrasting, topic-true worlds that show the
   stakes. CORE LENS — the GAP: the two sides are the SAME starting point that
   split, not just a grim world beside a winning one — the same subject (a person,
   business, deal or asset) ending opposite ways because of ONE decision the viewer
   can't yet see, so the image itself asks "what did one do that the other didn't?"
   Make the two sides DIRECTLY comparable: the SAME kind of thing in opposite states
   (one cheque full vs shredded, one door open vs slammed shut, a desk thriving vs
   stripped bare), placed side by side so the eye reads one against the other at a
   glance — not two unrelated props sharing a room. Give each side ONE clear subject
   in its two states (the same house intact vs seized), never a pile-up of extra
   symbols on one side — a villa AND a safe AND gold is two cues too many; one clean
   subject reads faster.
   Go HARD on the contrast — this should hit, not whisper: a grim, harsh,
   high-stakes side versus a prosperous, winning side (e.g. a run-down derelict
   house vs a luxury villa, an empty/foreclosed space vs a thriving one, a pile of
   red "FINAL NOTICE" letters vs a full ledger, dark storm light vs warm success).
   Carry the drama entirely through PLACES, OBJECTS, light and color — NOT through a
   depicted suffering person. Never show an identifiable human as a victim, in
   distress, poverty, or pain; the only person in the frame is the creator, framed
   as the authority pointing at the divide, never as the one being harmed. Creator
   positioned so the central divide stays fully visible — straddling it or just to
   one side — reacting to it, face still dominant. This archetype BENDS
   "one focal point": the split is deliberate, but the creator's face must stay the
   single dominant element and each half must be instantly readable and clearly
   subordinate, never two competing focal points. Distinguish the two halves by real,
   believable means (different setting, props, or natural light level). The divide
   sits on the vertical CENTER line of the frame, splitting it into two roughly equal
   left/right halves — never pushed off to one side. It may be a hard edge OR a
   softer organic transition (an almost liquid seam where the two worlds meet); soft
   is fine, but it must stay CENTERED and instantly readable at 0.3s, never a vague
   smear, fog, floating particles, or murky AI sludge. Both worlds concrete and
   graspable in 0.3s. BOTH halves must be equally photographic: the grim side is a
   REAL photograph of decay — same texture, grain, camera and light logic as the
   winning side — never a darker, vaguer illustration or an "AI drawing" (a common
   split failure where the good side looks shot and the bad side looks sketched).

3) REACTION-TO-OBJECT
   Creator framed large with a strong, intense reaction to ONE real object that
   carries the stakes — placed beside them or held near the chest, lit as part of
   the same scene with real materials, weight and a real contact shadow. CRITICAL
   BALANCE — it must be surprising AND completely real, never AI gimmickry:
   CORE LENS — the REVERSAL: the strongest version flips the object's meaning —
   something that normally signals threat, cost or bad news, caught doing the
   OPPOSITE (paying out, opening, protecting, working FOR the viewer). "Wait, that's
   backwards" is the hook. Stage it as a visible MOMENT, not a static pose: the
   object mid-action — being pulled from the envelope, tipping, spilling, caught as
   it falls, snatched — at the instant its meaning flips, with the creator reacting.
   A calmly held or neatly displayed object kills the tension; capture the verb, not
   the noun. "Ordinary and recognizable" describes the OBJECT, never the SCENE: you
   have wide latitude to stage something dramatic and unexpected, and the ONLY hard
   limit is realism — every element must be a real thing you could photograph. Real
   objects may collide dramatically but must NOT turn surreal: never give the object
   a face, teeth, or a life of its own, and never let it melt (e.g. a tax envelope
   drawn as a monster). Surprise through the situation, not an impossible object.
   The destructive beat below is the fallback when the script offers no natural
   reversal.
   - The OBJECT itself is ordinary, everyday and instantly recognizable: something
     that genuinely exists and a viewer knows on sight (an IRS envelope, a real
     cheque, cash, a bank card, a key, a padlock, a passport, a phone screen). Do
     NOT invent a novelty object, a fantasy gadget, or an engraved/branded plaque or
     sign — those read as AI slop.
   - The object must MEAN something, not just NAME the topic. It has to play a real
     role in the story and carry the stakes through what it is and what happens to
     it (a shredded cheque = money gone). A prop that merely has the subject written
     on it (a plaque or label reading the topic) is banned — that is a caption, not a
     metaphor. And never translate an abstract word LITERALLY: a "gap", "flaw" or
     "hole" in a plan is NOT a literal hole cut in a document — show its real-world
     CONSEQUENCE (a summons coming through, a creditor reaching in), not the word
     made solid.
   - The SURPRISE comes from what is HAPPENING to that ordinary object, or its
     state: it is being shredded, torn in half, burning at one edge, locked, cut
     with scissors, cracked, stamped — a real moment that opens a curiosity loop and
     makes the stakes felt. Real object, unexpected situation.
   The power is the COMBINATION of the creator's hot expression and that single
   object mid-moment. One object, one face, one reaction — no clutter. Do NOT use a
   plain sheet of paper or a document as the hero object.

4) ON-LOCATION
   Take the creator OUT of the studio into a meaningful real-world PLACE tied to the
   topic — e.g. a bank lobby, a vault room, an empty or foreclosed building, a
   construction site, in front of a mansion, a government/IRS-style building, a
   trading floor. The location carries the context and breaks the studio look.
   Photograph it as a real environment with depth and the natural light of that
   place; the creator stays large and dominant in the foreground, reacting. This
   archetype does NOT use the creator's studio set, so ignore creator_visual_notes
   for this one.

In each concept's "angle", say how that archetype is made to fit THIS script.

ANTI-CLICHÉ (run this for EVERY concept, after choosing its archetype):
- List the 4-6 most overused thumbnail EXECUTIONS for THIS exact topic — the tired
  literal renders a viewer has already seen on 50 other finance videos (e.g. a
  plain calculator on a desk, a generic 1040 form, a piggy bank, stacked coins).
- You may NOT reuse those tired executions as-is — but do NOT overcorrect by
  fleeing into a "premium" but subject-blind object (a generic vault, plain door,
  blank envelope). Give a topic-TRUE object ONE fresh, unexpected turn: reinvent
  the topic, do not abandon it.
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
  (each archetype sets its own background per the STYLE LIBRARY). Place the creator
  inside a real environment with depth — at a desk or table, in a study, office, or
  room — with foreground, midground and a background that has lamps, shelves,
  furniture or texture. Light it naturally and cleanly; it must not be a flat, empty
  void. A cleaner background is fine when a single
  hero object carries the shot, but lean toward the richer, fuller scene.
- REAL, RECOGNIZABLE props. Any object carrying the metaphor must be an ordinary,
  instantly recognizable real thing, photographed with real materials and a natural
  contact shadow — never a plasticky AI object, an invented novelty, an
  engraved/branded plaque, or a flat default (a blank sheet, a plain paper stack).
- Keep the environment SUBORDINATE: softly lit / gently blurred so the face and
  hero object stay sharp and dominant in front. Avoid props the viewer cannot
  place (cracked rock or tree stumps, drifting fog, floating embers/particles,
  glowing cracks, swirling debris).
- Avoid depending on an exact COUNT of objects (e.g. "five envelopes") — image
  models miscount. Use "one red among plain ones" or a single hero object.

THE OVERLAY — a short, emotionally charged phrase, 2 to 4 words (per concept):
- Keep it punchy, but it may run a little longer than a bare label (2 to 4 words)
  when that makes it land harder — e.g. "IT'S NOT WORTH IT", "THIS WILL COST YOU",
  "STOP DOING THIS", "YOU'RE LOSING MONEY". Line-breaking, size and typography are
  handled automatically after render — you only choose the WORDS, and they MUST
  read as one natural, coherent phrase, never random or nonsense words.
- EMOTION over label, but always CRYSTAL CLEAR. Do NOT just name the topic
  ("MARRIAGE TRAP", "JOINT RETURN") and do NOT write a flat, informational
  description of the situation ("YOU CAN STILL CLAIM", "RETIREE TAX BREAK") — those
  state a fact instead of landing a punch. Make the viewer FEEL something with a
  direct, personal line that hits a nerve or a desire: "IT'S YOUR MONEY", "THEY
  STEAL FROM YOU", "YOU'RE LOSING MONEY", "DON'T LET THEM", "IT'S STILL YOURS". It
  must stay plain and instantly readable — emotional, NOT vague, cryptic, or clever
  for its own sake. EMOTION TEST before you finalize: would this make a stranger
  feel something (anger, fear of loss, "wait, that's mine")? If it merely tells them
  what the video is about, rewrite it until it stings. Speak to the viewer directly
  ("you / your / they") whenever it fits. Choose the register that best fits THIS
  concept, and vary it across the three:
    * possession / theft  — "IT'S YOUR MONEY", "THEY STEAL FROM YOU"
    * warning / urgency  — "BEFORE IT'S TOO LATE", "DON'T DO THIS"
    * personal verdict   — "IT'S NOT WORTH IT", "YOU'RE WRONG ABOUT THIS"
    * reveal / curiosity — "WHAT THEY HIDE", "THE REAL COST"
    * loss / stakes      — "YOU'RE LOSING MONEY", "IT COSTS YOU MORE"
  Pick whichever hits hardest for the concept. This is a REQUIREMENT, not a
  preference: every overlay MUST land on at least one of three — a stake for
  someone the viewer cares about, a loss that is still coming, or a false sense of
  safety punctured. Naming the concept or the angle does NOT count: "ONE GAP
  DECIDES", "THE HOLE THEY MISSED", "ONE FLAW DECIDES" describe the video instead of
  hitting a nerve, so they FAIL this rule. Still vary the register across the three.
- High-stakes, not loud. The phrase should feel CONSEQUENTIAL — imply a real,
  significant stake (money lost, a costly mistake, something genuinely at risk),
  never a mild observation. Raise the STAKES, not the volume: keep it credible and
  adult, never cartoonish, screaming, or over-hyped.
- Anchor when natural. If you can tie the phrase to the real topic (the tax, the
  IRS, the account, the retiree) without killing the emotion, do so — but a pure
  emotional hook is fine when the image and the title already make the subject
  clear. Either way the viewer must feel they have to click.
- Still implies, never explains — do not state the lesson or spoil the answer.
- Never repeats or summarizes the title, and never reuses a word from it — and go
  further: the overlay must pull a DIFFERENT axis than the title already gives, not a
  reworded version of its promise. Add a WHEN ("before you earn"), a CONSEQUENCE
  ("one filing decides"), a THREAT or LOSS ("you're losing money"), or a WHO/WHOSE
  — something the title leaves unsaid. If the title already states the payoff, open
  a NEW loop; do not restate it in punchier words.
- No creator name, no hashtags. Punctuation only if it adds tension.
- Output the words plain, separated by single spaces only — never a slash, pipe,
  dash, bullet, or any other separator between them.
- HIGHLIGHT (per concept): pick exactly ONE word from the overlay that carries the
  stake or the emotion (BROKE, STEAL, LOSING, YOUR, TRAP) — never an article,
  preposition, or the bare topic noun — and set highlight_style:
    * "none"  — all white (safe default when the image itself is already loud)
    * "color" — that one word in the brand red, the rest white
    * "block" — that one word in a solid red block with white text (news-banner
      punch; the strongest — use it when one word IS the whole message, not by
      default)
  Vary the highlight styles across the three concepts; at most ONE "block" per run.

TEXT IN THE IMAGE:
- The caption is NOT rendered by the image model. It is composited afterwards in a
  fixed brand font, inside the caption_zone you choose per concept. Your job is
  composition: pick the caption_zone (top-banner, top-left, top-right, mid-left,
  mid-right, or bottom-left — NEVER bottom-right, YouTube's duration badge covers
  it) on the OPPOSITE side of, or clearly above, wherever the face and the hero
  object sit, and make scene_prompt keep that zone CALM — free of the face, the
  hero object, clutter and busy detail — so large white text reads instantly. It
  does NOT have to be flat, empty or even-toned: contrast may come from color or
  depth (a cool backlight glow, a dark but softly lit wall, gentle bokeh), and the
  scene's natural light and atmosphere should keep running through it. The caption
  block is LARGE (roughly a third of the frame, full-width for top-banner), so the
  calm area must be generous — the caption must NEVER touch or cover the creator's
  face or the hero object.
- top-banner = a clean horizontal band across the entire top of the frame, like a
  news banner. When you choose it, frame the creator with clear HEADROOM: head and
  hair stay fully below the top ~30% of the frame.
- NO text appears in the scene itself. The only exceptions: (a) small, incidental
  text that naturally lives on a real object (a short header or sign on a prop,
  1 to 3 words, tiny and real-looking, never hero-sized, never sentences — the
  image model garbles long text), and (b) the single large marked figure in an
  evidence-closeup concept.

AVOID (AI-slop tells): cluttered scenes, multiple competing focal points, generic
stock look, over-saturation, plastic skin, gibberish or misspelled text, extra
logos/watermarks, and any legible text beyond, at most, one tiny incidental
real-world label on a prop (or the single marked figure in evidence-closeup).

STEP 3 — SELF-AUDIT & FIX (do this silently, before writing the JSON):
For EACH concept, score two axes 0-10 and FIX any that fall short before output:
- overlay_punch (= overlay.score): does the caption clear the OVERLAY rules above
  — hitting at least one lever, not just describing the image or naming the
  mechanic? Captions like "THEY GET THROUGH" or "NOTHING BEHIND IT" describe what is
  shown and score LOW; rewrite until it hits a nerve.
- expression_match: does the creator's expression fit THIS concept's register (no
  smile or pleased look on a loss / threat / seizure concept)? If not, correct it in
  BOTH subject_direction and scene_prompt.
Any concept below 7 on either axis MUST be rewritten, not shipped as-is. Do not lower
the bar to pass — raise the concept. Keep the pinned archetypes: rewrite the
concept, never swap it.

OUTPUT — return ONLY valid JSON, no preamble:
{
  "topic_read": "<one line: what actually makes this clickable>",
  "cliches_banned": ["<overused execution>", "<overused execution>"],
  "archetype_fit": {
    "evidence-closeup": "<0-10 how well this fits THIS script + one-line why>",
    "opposites-split": "<0-10 + why>",
    "reaction-to-object": "<0-10 + why>",
    "on-location": "<0-10 + why>"
  },
  "archetypes_chosen": ["opposites-split", "reaction-to-object", "<free pick: evidence-closeup OR on-location, whichever scores higher>"],
  "concepts": [
    {
      "id": "A",
      "archetype": "<evidence-closeup|opposites-split|reaction-to-object|on-location>",
      "caption_zone": "<top-banner|top-left|top-right|mid-left|mid-right|bottom-left — where the composited caption sits; never where the face or hero object is, never bottom-right>",
      "angle": "<the distinct direction in a phrase, and why this archetype fits>",
      "visual_metaphor": "...",
      "subject_direction": "<expression (hot but credible), pose, gesture, framing, placement>",
      "composition": "<focal point, rule-of-thirds, fg/bg, and where the calm caption area sits>",
      "color_and_lighting": "...",
      "overlay": {"words": "2 TO 4 WORDS — a short emotional phrase", "highlight_word": "<exactly ONE word from words that carries the stake/emotion — or null>", "highlight_style": "<none|color|block>", "rationale": "...", "score": "<0-10 overlay_punch, >=7 after self-audit>"},
      "expression_match": "<0-10, >=7 after self-audit>",
      "freshness_score": 0,
      "click_score": 0,
      "scene_prompt": "<ONE paragraph for the image model: 16:9 photorealistic thumbnail that looks like a real photo, not a render; describe the creator WITH an explicit instruction to keep face and identity exactly consistent with the supplied reference images, same person; the archetype scene and/or metaphor; the hot-but-credible emotion; framing chest-up with eye contact by default; composition; natural realistic lighting. Keep the chosen caption_zone a generous, CALM area away from the face and hero object — no clutter or busy detail, but it keeps the scene's natural light, color and depth (a backlight glow or softly lit background is welcome there, it need not be flat or empty); the caption is composited there afterwards, so do NOT describe any caption or overlay text. If a real prop in the scene naturally carries a tiny header/sign, you may describe it (1-3 words, small and incidental, never hero-sized). End with 'One clear focal point, readable as a small mobile thumbnail.'>"
    },
    { "id": "B" },
    { "id": "C" }
  ],
  "recommended": "<A|B|C>",
  "why_recommended": "<one line>"
}

archetypes_chosen is fixed: opposites-split and reaction-to-object are always two of
the three; pick the third by fit score (the higher of evidence-closeup vs
on-location). Be decisive — no hedging. The recommended concept maximizes click_score
while keeping freshness_score >= 7. Every concept you emit must ALREADY pass the
STEP 3 self-audit at 7+ on both overlay_punch and expression_match — rewrite
before output, never ship a sub-7 concept.`;

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
    const settled = await Promise.allSettled(
      concepts.map((c) => renderConcept(runId, c, refFiles, creatorName))
    );

    const images = settled
      .filter((s) => s.status === 'fulfilled')
      .map((s) => s.value);

    const failures = settled
      .map((s, i) => (s.status === 'rejected'
        ? { id: concepts[i].id, error: String(s.reason?.message || s.reason) }
        : null))
      .filter(Boolean);
    failures.forEach((f) => console.error(`[generate-thumbnail] concept ${f.id} failed:`, f.error));

    if (!images.length) {
      throw new Error(`All concepts failed: ${failures.map((f) => f.error).join(' | ')}`);
    }

    const recommendedId = images.some((im) => im.concept_id === brief.recommended)
      ? brief.recommended
      : images[0].concept_id;

    await update(runId, {
      status: 'complete',
      art_director_json: brief,
      images,
      recommended_concept: recommendedId,
      error: failures.length ? `Partial: ${failures.map((f) => `${f.id} ${f.error}`).join(' | ')}` : null,
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

// ── render one concept: text-free GPT render -> caption composited -> upload ─
async function renderConcept(runId, concept, refFiles, creatorName) {
  const scene = String(concept.scene_prompt || '').trim();
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
    'it must read as captured, not generated: avoid a flawless, over-clean, perfectly symmetrical studio look.';
  const pngB64 = await gptImage(refFiles, `${scene}\n\n${identity}\n\n${quality}\n\n${buildNoTextDirective(concept)}`, creatorName);
  const sceneBuffer = Buffer.from(pngB64, 'base64');
  const finalBuffer = await composeCaption(sceneBuffer, concept);

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
    highlight_word: concept.overlay?.highlight_word || null,
    highlight_style: concept.overlay?.highlight_style || 'none',
    caption_zone: concept.caption_zone || null,
    archetype: concept.archetype || '',
    angle: concept.angle || '',
  };
}

// ── caption compositor (canvas text layer) ───────────────────────────────────
const CAPTION_FONT_PATH = fileURLToPath(new URL('./fonts/LibreFranklin-Black.ttf', import.meta.url));
const CAPTION_FONT_FAMILY = 'PT Caption';
const CAPTION_ACCENT = '#E11D2A';       // highlight color / block color
const CAPTION_BLOCK_ROTATION_DEG = 0;   // slight tilt (1-2) for younger brands; 0 = authority

let fontRegistered = false;
function ensureCaptionFont() {
  if (!fontRegistered) {
    GlobalFonts.registerFromPath(CAPTION_FONT_PATH, CAPTION_FONT_FAMILY);
    fontRegistered = true;
  }
}

// bottom-right is intentionally absent: YouTube's duration badge covers it.
const CAPTION_ZONES = {
  'top-banner':  { x: 0.06, y: 0.050, w: 0.88, align: 'center', maxFont: 0.200, minFont: 0.110 },
  'top-left':    { x: 0.05, y: 0.060, w: 0.55, align: 'left',   maxFont: 0.170, minFont: 0.095 },
  'top-right':   { x: 0.40, y: 0.060, w: 0.55, align: 'right',  maxFont: 0.170, minFont: 0.095 },
  'mid-left':    { x: 0.05, y: 0.340, w: 0.46, align: 'left',   maxFont: 0.160, minFont: 0.090 },
  'mid-right':   { x: 0.49, y: 0.340, w: 0.46, align: 'right',  maxFont: 0.160, minFont: 0.090 },
  'bottom-left': { x: 0.05, y: 0.600, w: 0.52, align: 'left',   maxFont: 0.160, minFont: 0.090 },
};

// Detail score of a zone on the actual rendered image (std dev of luminance).
// High detail = face / hero object / busy texture -> the caption must not go there.
function zoneDetailScore(ctx, zone, W, H) {
  const x = Math.round(zone.x * W);
  const y = Math.round(zone.y * H);
  const w = Math.round(zone.w * W);
  const h = Math.round(Math.min(0.30 * H, H - zone.y * H));
  const data = ctx.getImageData(x, y, w, h).data;
  let sum = 0, sumSq = 0, n = 0;
  for (let i = 0; i < data.length; i += 16) { // sample every 4th pixel
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sum += lum; sumSq += lum * lum; n++;
  }
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sumSq / n - mean * mean));
}

// Pick the safest zone: the art director's choice unless the render put detail
// (face, object) there and another allowed zone is clearly cleaner.
function pickZone(ctx, requested, W, H) {
  const req = CAPTION_ZONES[requested] ? requested : 'top-banner';
  const scored = Object.entries(CAPTION_ZONES).map(([name, z]) => ({
    name,
    score: zoneDetailScore(ctx, z, W, H) - (name === req ? 20 : 0), // preference bonus
  }));
  scored.sort((a, b) => a.score - b.score);
  return scored[0].name;
}

// Best split of 2-4 words into 1 or 2 lines = the one that yields the largest font.
// A top-banner is always ONE line (the news-banner look); it never grows a second
// line downward into face territory.
function bestLayout(ctx, words, zoneWidthPx, maxFontPx, singleLine) {
  const widthAt100 = (line) => {
    ctx.font = `100px "${CAPTION_FONT_FAMILY}"`;
    return ctx.measureText(line).width;
  };
  const candidates = [[words.join(' ')]];
  if (!singleLine) {
    for (let i = 1; i < words.length; i++) {
      candidates.push([words.slice(0, i).join(' '), words.slice(i).join(' ')]);
    }
  }
  let best = null;
  for (const lines of candidates) {
    const widest = Math.max(...lines.map(widthAt100));
    const font = Math.min(maxFontPx, (zoneWidthPx / widest) * 100);
    if (!best || font > best.font + 0.5) best = { lines, font };
  }
  return best;
}

// Draw one line word-by-word so a single word can be recolored or boxed.
function drawCaptionLine(ctx, lineWords, xStart, baselineY, fontSize, style, highlightWord) {
  ctx.font = `${fontSize}px "${CAPTION_FONT_FAMILY}"`;
  const spaceW = ctx.measureText('\u00A0').width;
  let x = xStart;
  for (const word of lineWords) {
    const m = ctx.measureText(word);
    const isHl = highlightWord && word === highlightWord;

    if (isHl && style === 'block') {
      const padX = fontSize * 0.16;
      const padY = fontSize * 0.10;
      const asc = m.actualBoundingBoxAscent;
      const desc = m.actualBoundingBoxDescent;
      ctx.save();
      if (CAPTION_BLOCK_ROTATION_DEG) {
        const cx = x + m.width / 2, cy = baselineY - asc / 2;
        ctx.translate(cx, cy);
        ctx.rotate((CAPTION_BLOCK_ROTATION_DEG * Math.PI) / 180);
        ctx.translate(-cx, -cy);
      }
      ctx.shadowColor = 'rgba(0,0,0,0.45)';
      ctx.shadowBlur = fontSize * 0.08;
      ctx.shadowOffsetY = fontSize * 0.04;
      ctx.fillStyle = CAPTION_ACCENT;
      ctx.fillRect(x - padX, baselineY - asc - padY, m.width + padX * 2, asc + desc + padY * 2);
      ctx.shadowColor = 'transparent';
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(word, x, baselineY);
      ctx.restore();
    } else {
      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.55)';
      ctx.shadowBlur = fontSize * 0.10;
      ctx.shadowOffsetY = fontSize * 0.05;
      ctx.fillStyle = isHl && style === 'color' ? CAPTION_ACCENT : '#FFFFFF';
      ctx.fillText(word, x, baselineY);
      ctx.restore();
    }
    x += m.width + spaceW;
  }
}

// Main entry: base PNG buffer + concept -> PNG buffer with the caption composited.
async function composeCaption(baseBuffer, concept) {
  ensureCaptionFont();

  const words = String(concept.overlay?.words || '')
    .replace(/[\/|]+/g, ' ').replace(/\s+/g, ' ').trim().toUpperCase()
    .split(' ').filter(Boolean).slice(0, 4);
  if (!words.length) return baseBuffer;

  let highlightWord = String(concept.overlay?.highlight_word || '').trim().toUpperCase() || null;
  if (highlightWord && !words.includes(highlightWord)) highlightWord = null;
  let style = String(concept.overlay?.highlight_style || 'none').toLowerCase();
  if (!['none', 'color', 'block'].includes(style) || !highlightWord) style = 'none';

  const img = await loadImage(baseBuffer);
  const W = img.width, H = img.height;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const zoneName = pickZone(ctx, concept.caption_zone, W, H);
  if (concept.caption_zone && zoneName !== concept.caption_zone) {
    console.log(`[composeCaption] zone '${concept.caption_zone}' too busy -> relocated to '${zoneName}'`);
  }
  const zone = CAPTION_ZONES[zoneName];
  const zoneWidthPx = zone.w * W;

  const layout = bestLayout(ctx, words, zoneWidthPx, zone.maxFont * H, zoneName === 'top-banner');
  const fontSize = layout.font; // never force it larger: that would overflow the zone
  ctx.font = `${fontSize}px "${CAPTION_FONT_FAMILY}"`;
  ctx.textBaseline = 'alphabetic';

  const lineGap = fontSize * 0.18;
  let cursorY = zone.y * H;

  for (const line of layout.lines) {
    const lineWords = line.split(' ');
    const m = ctx.measureText(line);
    const asc = m.actualBoundingBoxAscent;
    let xStart = zone.x * W;
    if (zone.align === 'center') xStart = zone.x * W + (zoneWidthPx - m.width) / 2;
    if (zone.align === 'right') xStart = zone.x * W + (zoneWidthPx - m.width);
    drawCaptionLine(ctx, lineWords, xStart, cursorY + asc, fontSize, style, highlightWord);
    cursorY += asc + m.actualBoundingBoxDescent + lineGap;
  }

  return canvas.encode('png');
}

// ── keep the render text-free: the caption is composited afterwards ──────────
function buildNoTextDirective(concept) {
  const zone = CAPTION_ZONES[concept.caption_zone] ? concept.caption_zone : 'top-banner';
  const zoneHint = zone === 'top-banner'
    ? 'a clean horizontal band across the entire top of the frame (clear headroom: head and hair stay fully below the top 30% of the frame)'
    : `the ${zone.replace('-', ' ')} area of the frame`;
  return [
    'NO CAPTION TEXT: render NO caption, headline, overlay text, subtitles, logos, or watermarks anywhere in the image.',
    `Keep ${zoneHint} as a generous, CALM area: the face, the hero object, and any clutter or busy detail stay fully out of it, because a large caption is composited there afterwards — but let the scene's natural light, color and depth continue through it (a backlight glow or a dark, softly lit background is fine; it must not become a flat, empty, toneless patch).`,
    'The ONLY text permitted is text the scene description above explicitly calls for: a single tiny incidental real-world label on a prop (1 to 3 words, small and natural, never hero-sized) or one single short marked figure on a document. Render such text correctly spelled at its natural size, and render absolutely no other letters, words, or numbers anywhere.',
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
