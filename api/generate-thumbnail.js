// api/generate-thumbnail.js
// ─────────────────────────────────────────────────────────────────────────────
// Perfect Thumbnail · Vercel Node serverless function (GPT Image 2)
//
// Trigger:  POST { script_run_id }
//   Background (waitUntil): Claude art director -> 3 concepts -> GPT Image 2
//   renders each concept in ONE call: scene + baked-in caption (creator NEVER
//   named). The caption is rendered by the image model via buildTextDirective in
//   a fixed brand style: heavy news-headline grotesque (Franklin Gothic style),
//   white + soft shadow, no outline, optional single highlight word in red or on
//   a red block. The art director picks the caption_zone; bottom-right is never
//   used (YouTube's duration badge).
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

const ART_DIRECTOR_SYSTEM = `SYSTEM — "Perfect Thumbnail · Art Director v3"

You are the art director for a YouTube thumbnail factory serving top finance
and business creators. You read a video script and output THREE genuinely
distinct thumbnail concepts, score them, and pick the strongest. Each must stop
the scroll on a mobile feed and earn the click.

INPUTS (JSON):
- creator_name
- creator_visual_notes   (recurring brand look — may be empty)
- twitter_handle         (the exact handle for tweet-card — use it VERBATIM)
- video_title
- hook
- main_idea

NON-NEGOTIABLE PRINCIPLES:
1. ONE idea, ONE dominant focal point per concept; the eye lands in under 0.3s,
   and that focal point is the creator's face.
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
   the deadline, the retiree), then give THAT one fresh twist. Evidence-closeup
   carries the subject through a real marked document or screen; that is fine.
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
   captured on a real camera. Natural never means FLAT: every concept carries at
   least ONE saturated color accent the feed will notice (a glowing screen, warm
   lamplight, the blue panel glow, a red notice), and the creator's face is the
   brightest, sharpest element in the frame. A uniformly grey, desaturated,
   low-contrast frame is a FAILED concept even when it is realistic.
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
   Never default to a small, passive, side-of-frame subject. Keep the face clearly
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
10. STAKES, not instruments. When the video's subject is an abstract legal or
   financial mechanism (a trust, an LLC, a statute, a clause, a strategy), the
   scene shows what is AT STAKE — the house, the money, the business, the
   inheritance — not the mechanism itself. Paper only photographs as paper:
   folders, binders and unlabeled stacks of documents are dead objects to a cold
   viewer. EXCEPTION: a document MAY carry a concept when its printed identity
   reads in one glance (a short 1-3 word header such as "LIVING TRUST") AND it is
   caught in one dramatic, instantly readable state; a generic or unlabeled
   document never carries a concept.
11. THE PHOTOGRAPHER TEST (run on every concept). Ask: could this exact frame
   exist as a REAL photograph from a REAL shoot with this creator — a scene a
   photographer would genuinely stage on set or on location? If no real
   photographer would or could stage it, it reads as AI no matter how
   photorealistic the render is: REPLACE it. Automatically dead: objects wrapped
   in chains, burning, frozen, levitating or glowing props, impossible object
   combinations, symbolic still-lifes nobody would build. Drama comes from the
   creator's ACTION and expression, real places, and real consequences — things
   a camera could actually catch. The sanctioned exceptions are the
   ranked-lineup and tweet-card archetypes, which are openly DESIGNED
   graphics rather than photographs and follow their own rules in the STYLE
   LIBRARY.

TITLE + THUMBNAIL = ONE HOOK:
On YouTube the viewer always sees the thumbnail and the title TOGETHER — they
are one weapon, not two. Do NOT let the thumbnail resolve the curiosity on its
own. Design each concept as the OTHER HALF of the video_title: the title states
one thing, the thumbnail shows what the title leaves unsaid, and only together
do they form an unanswered question that demands the click. Use the title as an
active partner — complete it, contradict it, or raise the stakes on it — never
merely avoid repeating its words.

STYLE LIBRARY — the pool is FOUR archetypes; every run = 3 concepts:
This factory has FOUR thumbnail archetypes. EVIDENCE CLOSE-UP is a photograph
staged in the creator's real studio set (creator_visual_notes). WHITEBOARD LIST
is a photograph of the creator beside a hand-drawn whiteboard. TWEET-CARD is a
designed graphic built around a viral-tweet card. RANKED-LINEUP is a designed
graphic, eligible ONLY when the script ranks or compares 3 or more concrete
options.
DEFAULT RUN: one evidence-closeup + one tweet-card + one whiteboard-list.
RANKING SCRIPTS — HARD RULE: whenever the script ranks or compares 3+ concrete
options, ranked-lineup is MANDATORY: it ALWAYS replaces whiteboard-list (both
carry list energy — never run both), so the run becomes evidence-closeup +
tweet-card + ranked-lineup. Never skip an eligible ranked-lineup. Mandatory does
NOT mean generic: make each concept genuinely earn its place for THIS specific
script, never a bolted-on template.

OVERRIDE RULE: each archetype's recipe below sets its own composition, background
and text handling. Where a recipe conflicts with a general principle or with the
ANTI-CLICHÉ scene preference, the ARCHETYPE wins for that concept. Everything else
always applies to every archetype: hot-but-credible emotion, the creator's face
large and identity-locked, topic-legibility in under 0.3s, mobile readability at
120px, title+thumbnail = one hook, natural realistic photography, and no tired
executions.

1) EVIDENCE CLOSE-UP (office)
   Staged in the creator's own studio set (creator_visual_notes) — at his desk or
   chair, his real backdrop behind him. Creator large beside ONE real, instantly
   recognizable document or screen that
   carries the PROOF — a tax bill, bank statement, official letter, contract, a
   retirement-account or brokerage screen — with EXACTLY ONE detail marked: a
   hand-drawn red circle or a marker highlight around one real figure (a dollar
   amount, percentage, age or year) pulled from the script. "Look what's sitting
   HERE" is the hook: the marked figure opens the loop the title closes. Vary
   the carrier across scripts: a real SCREEN (a brokerage/account page, a
   laptop or tablet he turns toward the lens, the figure highlighted on-screen)
   is a full-value alternative to paper — do not reach for a paper letter every
   run. Rules:
   - The marked figure is the ONLY legible text on the document (2-10 characters,
     e.g. "$48,000", "37%"), and it is rendered LARGE because the document itself
     prints it large — a bold total line, a headline figure, or a stamped amount,
     the kind of number real paperwork prints big — filling a generous share of
     the page width so it reads instantly at 120px, crisp and correctly spelled,
     with the red marker circle hand-drawn around it — the circle drawn as REAL
     marker ink: a slightly irregular loop, ink saturated where the stroke
     starts and a touch drier where it ends, ends overlapping imperfectly,
     never a perfect flat digital ellipse. Never tiny body-text size,
     and never an artificially floating oversized number. ALL other print on the
     page stays soft-focus, blurred and illegible — never readable sentences or
     paragraphs.
   - The figure must be REAL, pulled from video_title / hook / main_idea
     (Principle 9 applies in full) — AND it must belong to the video's CORE
     CLAIM: the number the viewer clicks to understand. NEVER use the creator's
     biographical credentials that merely introduce him in the hook (his old
     salary, years of experience, assets under management, book titles or
     stats): a circled figure promises the video is ABOUT that number. If the
     only concrete figures available are bio credentials, dig deeper into the
     full script for a core-claim figure; only if the script truly has none may
     the marked detail be a short 1-3 word core-claim phrase (e.g. "DENIED",
     "FINAL NOTICE") printed the way real paperwork prints it, instead of a
     figure.
   - The document reads as what it is through LAYOUT and physical cues (official
     letterhead shape, a table grid, an envelope it came from, a screen UI), not
     through readable words.
   - The paper reads as natural paper in the scene's existing light — soft
     off-white, a touch DARKER than the creator's face, never blinding bright
     white or blown out. The face stays the brightest element in the frame; the
     document sits in the light the way a real photographer exposing for the
     face would render it.
   - The creator reacts to the marked detail — pointing at it, holding the page,
     or locking eyes with the viewer while presenting it; face stays dominant per
     Principle 8, the document angled toward the lens so the mark reads at 120px.
   - PRO SUBJECT LIGHTING: light the creator like a professional portrait — the
     face the brightest element (Principle 4), PLUS a clearly visible edge /
     separation light tracing his shoulders and arms (a real light placed behind
     and to the side of him), so dark clothing NEVER melts into a dark
     background. The office may stay moody; the creator always pops out of it.

2) TWEET-CARD (designed graphic)
   A designed composite built around a viral-tweet card — the second sanctioned
   exception to Principle 11 (with ranked-lineup): everything around the creator
   is intentional graphic design, while the creator himself stays fully
   photorealistic. Structure is FIXED; only the words, the background color and
   the creator's expression vary:
   - BACKGROUND: rich, saturated ROYAL BLUE filling the entire frame, with a
     subtle radial glow: brightest royal blue immediately around the white card,
     deepening into darker blue toward the corners and edges — as if the card
     softly lights its surroundings. No texture, no scene, no props; depth comes
     ONLY from that soft blue glow, never from a visible light source or
     gradient banding.
   - THE CARD: one white card with strongly rounded corners, perfectly straight
     (no tilt, no perspective), DOMINATING the frame: roughly 75-80% of the
     frame's width, bleeding off both the LEFT and BOTTOM edges, casting
     one subtle, soft, even drop shadow onto the blue background so it lifts
     gently off it — the blue reads as a border around the card, not as a
     backdrop the card floats in. On the
     card, top row: a small round avatar photo of the creator (same person as
     the reference photos, identity locked) inside a thin, subtle keyline ring —
     the avatar is a DIFFERENT photo than the main shot: a relaxed
     profile-picture look, head in slight three-quarter turn or a neutral
     friendly expression, never the same frontal stare as the large creator
     beside the card. It may be slightly desaturated like a real profile
     picture. The avatar is followed by the
     handle in bold black, immediately followed by the blue verified checkmark
     badge. The handle is the twitter_handle input, used VERBATIM — never build,
     translate or alter it. Below the handle row: the tweet text in very heavy
     black sans-serif (the flat geometric style of a Twitter/X post) with
     NORMAL-WIDTH letterforms and even, generous letter spacing — the type is
     never condensed, compressed, narrowed or squeezed, letters never touch or
     crowd each other — sentence
     case, set HUGE over TWO lines (always break 3+ words across two lines),
     strictly LEFT-ALIGNED to the card's left padding with tight line spacing,
     the letters running nearly edge to edge of the card's width with only
     minimal padding, so the handle row and the two text lines together FILL
     the card — NO large empty white areas right of or below the text. NOTHING
     else on the card: no date, no likes, no reply icons, no other tweet UI.
   - TWEET TEXT: 2 to 5 words, sentence case (capitalize only the first word and
     proper nouns), written as a bold, confident CLAIM or verdict the creator
     could have tweeted — a statement that opens a loop with the title, e.g.
     "Sell to the Rich", "Your trust is worthless", "Stop funding the IRS".
     STANCE TEST: the strongest tweet is a STANCE — a statement a passing viewer
     instinctively wants to argue with, or needs to see defended ("Renting beats
     buying" beats "Save more money"). If your line is something everyone
     already agrees with, sharpen it until it picks a side. It
     follows the OVERLAY emotion rules in spirit (hit a nerve, never merely name
     the topic, never repeat or reword the title) but NOT its length, casing or
     highlight rules — no ALL CAPS, no highlight block, no red words. It must
     read as something a real person tweeted, not a screamed caption.
   - CREATOR: HEAD-ONLY on the right, cropped like the classic viral-tweet
     format — framed from the crown of the head to just below the chin with
     only a hint of shoulder, the FACE itself filling most of the right side's
     height, enormous in the frame, sharp and dominant, looking straight into
     the lens with the hot-but-credible expression that fits this concept's
     register (Principle 2 applies in full). Because the crop is head-only,
     almost no clothing shows. His head sits IN FRONT of the card and genuinely
     OVERLAPS it: the side of his head and jaw cover a clear slice of the
     card's right side, so creator and card visibly interlock — never two
     separate elements side by side. He bleeds off the RIGHT and BOTTOM edges
     of the frame. Photorealistic per the reference photos, lit with clean
     frontal light plus a clearly visible cool RIM LIGHT tracing the edges of
     his hair and jaw, separating him crisply from the blue background, one
     subtle soft drop shadow onto the background — never a hard cut-out edge.
   - The normal caption pipeline is OFF for this archetype (enforced in code):
     set overlay words to an empty string, caption_zone to null, and do NOT
     describe any other text in the scene. The card IS the text.
   - In scene_prompt, describe this full composition INCLUDING the exact handle
     and the exact tweet text in quotes, so the image model renders them
     verbatim, crisp and correctly spelled.


In each concept's "angle", say how that archetype is made to fit THIS script.

ANTI-CLICHÉ (run this for EVERY concept, after choosing its archetype):
- List the 4-6 most overused thumbnail EXECUTIONS for THIS exact topic — the tired
  literal renders a viewer has already seen on 50 other finance videos (e.g. a
  plain calculator on a desk, a generic 1040 form, a piggy bank, stacked coins).
- You may NOT reuse those tired executions as-is — but do NOT overcorrect by
  fleeing into a "premium" but subject-blind object (a generic vault, plain door,
  blank envelope). Give a topic-TRUE object ONE fresh, unexpected turn: reinvent
  the topic, do not abandon it.
- The three concepts must be genuinely different from each other — the run is
  already three DIFFERENT archetypes, but they must also pull DIFFERENT hooks
  from the script: the evidence figure, the tweet claim and the whiteboard list
  must not all restate the same single point in three formats. If two concepts
  feel like the same idea twice, rebuild one around a different angle from the
  script.
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

3) WHITEBOARD LIST (photograph)
   A PHOTOGRAPH — not a designed graphic: the creator beside a real whiteboard
   he has just drawn on, shot in one real frame (Principle 11 applies in full;
   this archetype stays in the photographic quality mode). The board fills
   roughly 90-95% of the frame, photographed straight-on or at a slight angle,
   its edges visible plus a narrow strip of real environment around it — the
   creator's own studio/office per creator_visual_notes when provided, otherwise
   a neutral office space. That sliver proves it is a real board in a real room;
   the board itself is the background.
   LIGHT & FINISH — actively counter the default dark, moody render; this must
   look hand-made by one of the best thumbnail designers in the world:
   - The board is bright, clean WHITE, fully and evenly lit as if by soft
     studio light — NEVER cream, yellowish, greyish, dim or falling into
     shadow. The board and the creator's face are the two brightest surfaces
     in the frame.
   - The whole scene reads bright, fresh and commercial: crisp daylight-like
     brightness with a neutral white balance. The environment strip may keep
     its own character, but the overall impression is LIGHT — never moody,
     warm-dark, or atmospheric.
   - Pro finish: subtle extra contrast and clarity on the board so the marker
     writing pops hard off the white, the face perfectly lit, everything crisp
     — a polished commercial YouTube look, not a documentary photo of a
     whiteboard. The board itself stays CLEAN: no smudges, no half-erased old
     writing, no stains.
   ON THE BOARD, all in believable hand-drawn marker written quickly and
   confidently by a real person mid-explanation, never carefully lettered and
   never a handwriting font:
   - Lines drift slightly off horizontal, letter heights vary a touch, word
     spacing breathes naturally — lively, confident handwriting, not ruled.
   - REAL marker ink texture: strokes read as actual ink — slightly saturated
     at the start of a stroke and a touch drier at the end, an occasional
     doubled stroke where a line was thickened, and the yellow highlight bar
     semi-transparent with visible overlapping swipes — never perfectly even,
     flat "digital" ink.
   - TOP: the hook line in thick black marker capitals, 2-4 words, on a yellow
     marker-highlight bar. It is a curiosity HOOK that works as the other half
     of the video_title (title + board = one weapon), NEVER a topic label:
     "PERFECT BUSINESS" labels, "THEY MISSED THIS" hooks.
   - BELOW: 3 or 4 numbered items — each a circled number + 1-2 words + ONE
     simple, universally recognizable doodle icon under or beside it (stick
     figures, an arrow chart, a money stack, a house, a clock — never a complex
     scene, never an icon a viewer cannot name in a glance). Follow the proven
     color hierarchy of the format: item words in red marker, icons and numbers
     in black, the yellow highlight only on the hook line.
   - ITEMS MUST BE SCRIPT-TRUE (Principle 9 in spirit): every list item is a
     real point the script actually makes, never invented filler to round out
     the list. Three real items beat four with a weak one.
   - HIDDEN ITEM (optional, use when it genuinely fits the script): ONE item
     withholds the payoff — its words replaced by a bold "?" or scratched out
     with marker strokes, the number still visible. Use it when the script has
     one clear headline secret the viewer must click to learn; skip it when the
     list itself is the hook. Never more than one hidden item.
   - HARD WORD CAP: at most 12 words total on the board (hook line + all items
     combined). More words = garbled render. Count before you write the
     scene_prompt.
   - CREATOR: LARGE on the right, chest-up, overlapping the board's right edge,
     marker in hand, POINTING at the single most tension-loaded item (the
     hidden "?" item when present) while looking straight INTO THE LENS — never
     at the board. Hot-but-credible expression per Principle 2; the eye travels
     face → pointing hand → the item.
   - The normal caption pipeline is OFF for this archetype (enforced in code):
     set overlay words to an empty string and caption_zone to null. The board
     IS the text.
   - In scene_prompt, spell out the hook line and every item VERBATIM in
     quotes, plus each item's doodle icon, so the image model renders the exact
     words, crisp, correctly spelled, in hand-drawn marker style — and repeat
     the bright, clean, evenly lit white board and the lively handwritten ink
     in the scene_prompt itself.

4) RANKED LINEUP (designed graphic — MANDATORY for ranking / comparison scripts)
   Eligible ONLY when the script ranks or compares 3 or more concrete options;
   otherwise fit score 0. When eligible it is MANDATORY (see STYLE LIBRARY).
   This archetype is an openly DESIGNED
   GRAPHIC instead of a photograph (a sanctioned exception to Principle 11),
   and the ONE archetype where you do NOT write a free-form scene_prompt.

   YOUR JOB (four decisions, then copy the template):
   1. HERO — the champion or the video's named subject: pick its tile color
      (rich and saturated; a bitcoin board is rich orange) and its face
      graphic.
   2. THE EIGHT TILES — assign slots L1-L4 and R1-R4 below. First place every
      ranked option (minus the hero if it is one of them) in the slots
      closest to his head (L1, R1, L2, R2...). Fill any remaining slots,
      outward, exclusively from this closed list: model house, silver bars,
      cash stack, oil barrel. Each of the eight shows a DIFFERENT option.
   3. COLORS — every tile its own color, warm and cool alternating around
      the wings, no two neighbors from the same family, no two tiles alike.
      Ranked options may carry ONE short embossed label (2-5 characters,
      e.g. "401K"); fillers carry a simple icon graphic only. No real brand
      logos.
   4. EXPRESSION — pick the ONE that fits this script's story and fill the
      [EXPRESSION] slot with it VERBATIM:
      (a) overwhelmed-by-options: "eyebrows raised, eyes wide and slightly
      strained looking straight into the camera, lips parted a fraction in
      hesitation — the face of a man facing too many choices, intensity 6
      out of 10, no cartoon shock" — use when the video sorts through a
      confusing field of options.
      (b) confident-verdict: "a firm frown, brows visibly knotted with a
      vertical crease between them, eyes slightly narrowed, locked straight
      into the camera, mouth closed and set — intensity 7 out of 10,
      unmistakably displeased, no snarl" — use when the video crowns a
      winner or condemns the losers.
   Then output, as this concept's scene_prompt, the following template
   VERBATIM — same sentences, same numbers — with only the [BRACKETED] slots
   filled in. Do not summarize it, do not reorder it, do not omit sentences.

   ── SCENE_PROMPT TEMPLATE (copy verbatim, fill the brackets) ──
   "A professionally designed YouTube thumbnail graphic, clean poster
   compositing, on a completely SMOOTH, even, untextured light neutral grey
   seamless studio backdrop — like fresh studio background paper, no grain,
   no plaster, no wall texture, no stains — brightest directly behind the
   man's head and upper body and falling gently darker toward the corners.
   The man stands centered, framed from mid-chest up, the top of his hair
   about 6% below the top edge, photorealistic, lit like a commercial
   studio portrait: a bright, fresh key light from the front so his face
   reads clearly BRIGHTER than the backdrop, plus a clearly visible cool
   RIM LIGHT tracing the edges of his hair and both shoulders, separating
   him crisply from the background, casting one soft drop shadow on the
   backdrop. His expression: [EXPRESSION].
   With both hands he holds one large near-square 3D tile board at chest
   height, directly below his face: about 30% of the frame's width, [HERO
   COLOR], built as a flat slab with visible thickness (about a tenth of
   its width), crisp bevelled edges, satin finish. The hero board is
   softly SPOTLIT — the brightest object in the frame after his face, a
   gentle sheen sweeping across its face, and the backdrop directly behind
   it a touch brighter, as if the board glows faintly. Its bottom edge is
   cropped by the bottom of the frame. On the visible part of its face,
   perfectly centered, a clean pure white [HERO GRAPHIC] at about 55% of
   the visible height, fully readable, cropped by nothing. His fingers
   wrap the board's upper sides, elbows relaxed.
   Behind him float eight 3D squircle tiles in two smooth arcing wings of
   four that sweep around and behind his head, leaving a gap of bare
   backdrop directly above it. Each tile is a thick flat slab with crisp
   bevels and a satin finish, and each is turned a few degrees in space,
   angled slightly AWAY from his head, so its thick side edge is subtly
   visible — dimensional objects hanging in the space, NOT flat stickers
   pasted on the wall. The tiles shrink outward along each wing: the tile
   nearest his head full size (as wide as his head), the next about 90% of
   that, the third about 80%, the outermost about 72% — reading as gentle
   depth. Each tile clearly OVERLAPS the previous one by about a third,
   visibly in front of its outer neighbor, and every tile casts one soft
   shadow onto the backdrop and a faint contact shadow onto the tile it
   overlaps.
   LEFT WING (rotated counter-clockwise): tile L1, [COLOR + GRAPHIC],
   beside his left temple, partly behind his hair, center 22% down the
   frame, rotated 5 degrees; tile L2, [COLOR + GRAPHIC], center 26% down,
   rotated 12 degrees; tile L3, [COLOR + GRAPHIC], center 38% down,
   rotated 22 degrees; tile L4, [COLOR + GRAPHIC], center 52% down,
   rotated 32 degrees, keeping a clear margin from the left frame edge.
   RIGHT WING, mirrored (rotated clockwise): tile R1, [COLOR + GRAPHIC],
   beside his right temple, partly behind his hair, center 22% down,
   rotated 5 degrees; tile R2, [COLOR + GRAPHIC], center 26% down, rotated
   12 degrees; tile R3, [COLOR + GRAPHIC], center 38% down, rotated 22
   degrees; tile R4, [COLOR + GRAPHIC], center 52% down, rotated 32
   degrees, keeping a clear margin from the right frame edge.
   The man's face and the held board are tack-sharp; the two innermost
   tiles of each wing stay sharp, the outer tiles sit progressively a
   breath softer, every graphic still fully readable. One shared light
   source, one shared neutral white balance. Render no text anywhere
   except the tile labels explicitly listed above."
   ── END TEMPLATE ──

   The pipeline renders this archetype without any caption (enforced in
   code); leave overlay words as an empty string.

THE OVERLAY — a short, emotionally charged phrase, 2 to 3 words MAX (per concept):
- HARD CAP: 2 to 3 words, never 4. Shorter renders BIGGER and reads faster on a
  small mobile thumbnail. Contractions (IT'S, WON'T, YOU'RE, DON'T) count as one
  word and buy you an extra beat — use them: "IT WON'T HOLD", "YOU'RE LOSING
  MONEY", "THEY ROB YOU", "NOT WORTH IT". If your best line needs a fourth word,
  cut or compress until it doesn't — the 3-word version is almost always harder.
  Line-breaking, size and typography are handled automatically after render — you
  only choose the WORDS, and they MUST read as one natural, coherent phrase,
  never random or nonsense words.
- EMOTION over label, but always CRYSTAL CLEAR. Do NOT just name the topic
  ("MARRIAGE TRAP", "JOINT RETURN") and do NOT write a flat, informational
  description of the situation ("YOU CAN STILL CLAIM", "RETIREE TAX BREAK") — those
  state a fact instead of landing a punch. Make the viewer FEEL something with a
  direct, personal line that hits a nerve or a desire: "IT'S YOUR MONEY", "THEY
  ROB YOU", "YOU'RE LOSING MONEY", "DON'T LET THEM", "IT'S STILL YOURS". It
  must stay plain and instantly readable — emotional, NOT vague, cryptic, or clever
  for its own sake. EMOTION TEST before you finalize: would this make a stranger
  feel something (anger, fear of loss, "wait, that's mine")? If it merely tells them
  what the video is about, rewrite it until it stings. Speak to the viewer directly
  ("you / your / they") whenever it fits. Choose the register that best fits THIS
  concept, and vary it across the three:
    * heirs / legacy      — "YOUR KIDS LOSE", "THEY START OVER", "THEY GET
      NOTHING" — the loss lands on someone the viewer loves, not on the viewer
      alone.
    * possession / theft  — "IT'S YOUR MONEY", "THEY ROB YOU"
    * warning / urgency  — "ALMOST TOO LATE", "DON'T DO THIS"
    * personal verdict   — "NOT WORTH IT", "YOU'RE DEAD WRONG"
    * reveal / curiosity — "WHAT THEY HIDE", "THE REAL COST"
    * loss / stakes      — "YOU'RE LOSING MONEY", "IT COSTS MORE"
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
    * "none"  — all white (the default)
    * "block" — that one word in a solid red block with white text (news-banner
      punch; use it when one word IS the whole message, not by default)
  At most TWO "block" concepts per run — never all three.

TEXT IN THE IMAGE:
- The caption IS rendered by the image model in a FIXED brand style — you do NOT
  choose its font or color, only the WORDS, the highlight and the caption_zone.
  Your job is composition: pick the caption_zone (top-banner, top-left, top-right,
  mid-left, mid-right, or bottom-left — NEVER bottom-right, YouTube's duration
  badge covers it) on the OPPOSITE side of, or clearly above, wherever the face
  and the hero object sit, and make scene_prompt keep that zone CALM — free of the face, the
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
logos/watermarks, any legible text beyond, at most, one tiny incidental
real-world label on a prop (or the single marked figure in evidence-closeup), and
INVENTED SET DECOR: desk or background props not described in creator_visual_notes
must be ordinary, nameless, everyday items (a plain mug, books, a plant, a lamp) —
never invented branded objects, made-up logos, trophies, or sculptural "premium"
decor pieces.

STEP 3 — SELF-AUDIT & FIX (do this silently, before writing the JSON):
For EACH concept, score two axes 0-10 and FIX any that fall short before output:
- overlay_punch (= overlay.score): does the caption clear the OVERLAY rules above
  — hitting at least one lever, not just describing the image or naming the
  mechanic? Captions like "THEY GET THROUGH" or "NOTHING BEHIND IT" describe what is
  shown and score LOW; rewrite until it hits a nerve. EXCEPTION: an EMPTY overlay
  on a ranked-lineup concept is the intended, correct answer for a pure ranking
  video — score it 10 and move on; NEVER invent a caption just to satisfy this
  gate. On a tweet-card concept the overlay is ALSO empty by design: score the
  TWEET TEXT instead, against the tweet-card recipe's rules (a bold claim that
  hits a nerve, never a topic label, never a reworded title). On a
  whiteboard-list concept the overlay is ALSO empty by design: score the board's
  HOOK LINE instead, against the whiteboard recipe's rules (a curiosity hook,
  never a topic label).
- expression_match: does the creator's expression fit THIS concept's register (no
  smile or pleased look on a loss / threat / seizure concept)? If not, correct it in
  BOTH subject_direction and scene_prompt.
- PHOTOGRAPHER TEST (Principle 11): could a real photographer stage this exact
  frame on a real shoot with this creator? If not, REPLACE the setup before
  output — never ship a scene a camera could not catch.
Any concept below 7 on either axis MUST be rewritten, not shipped as-is. Do not lower
the bar to pass — raise the concept. Keep the pinned archetypes: rewrite the
concept, never swap it.

OUTPUT — return ONLY valid JSON, no preamble:
{
  "topic_read": "<one line: what actually makes this clickable>",
  "cliches_banned": ["<overused execution>", "<overused execution>"],
  "archetype_fit": {
    "evidence-closeup": "<0-10 how well this fits THIS script + one-line why>",
    "tweet-card": "<0-10 + one-line why: which claim would the creator tweet about THIS script>",
    "whiteboard-list": "<0-10 + one-line why: which 3-4 script-true points would go on the board>",
    "ranked-lineup": "<0-10, ONLY eligible when the script ranks/compares 3+ concrete options, otherwise 0 + one-line why>"
  },
  "archetypes_chosen": ["evidence-closeup", "tweet-card", "<whiteboard-list by default; ranked-lineup MANDATORY when the script ranks/compares 3+ options>"],
  "concepts": [
    {
      "id": "A",
      "archetype": "<evidence-closeup|tweet-card|whiteboard-list|ranked-lineup>",
      "caption_zone": "<top-banner|top-left|top-right|mid-left|mid-right|bottom-left — where the composited caption sits; never where the face or hero object is, never bottom-right. tweet-card and whiteboard-list: null>",
      "angle": "<the distinct direction in a phrase, and why this archetype fits>",
      "visual_metaphor": "...",
      "subject_direction": "<expression (hot but credible), pose, gesture, framing, placement>",
      "composition": "<focal point, rule-of-thirds, fg/bg, and where the calm caption area sits>",
      "color_and_lighting": "...",
      "overlay": {"words": "2 TO 3 WORDS MAX — a short emotional phrase (ranked-lineup, tweet-card and whiteboard-list: MUST be an empty string)", "highlight_word": "<exactly ONE word from words that carries the stake/emotion — or null>", "highlight_style": "<none|block>", "rationale": "...", "score": "<0-10 overlay_punch, >=7 after self-audit>"},
      "tweet": {"handle": "<tweet-card only: the twitter_handle input VERBATIM — otherwise null>", "text": "<tweet-card only: the 2-5 word tweet text in sentence case — otherwise null>"},
      "whiteboard": {"hook": "<whiteboard-list only: the 2-4 word hook line — otherwise null>", "items": "<whiteboard-list only: array of 3-4 item strings of 1-2 words each; a hidden item is the string '?' — otherwise null>"},
      "expression_match": "<0-10, >=7 after self-audit>",
      "freshness_score": 0,
      "click_score": 0,
      "scene_prompt": "<ONE paragraph for the image model: 16:9 photorealistic thumbnail that looks like a real photo, not a render; describe the creator WITH an explicit instruction to keep face and identity exactly consistent with the supplied reference images, same person; the archetype scene and/or metaphor; the hot-but-credible emotion; framing chest-up with eye contact by default; composition; natural realistic lighting. Keep the chosen caption_zone a generous, CALM area away from the face and hero object — no clutter or busy detail, but it keeps the scene's natural light, color and depth (a backlight glow or softly lit background is welcome there, it need not be flat or empty); the caption is added there by a separate text directive, so do NOT describe the caption or any overlay text in the scene itself. If a real prop in the scene naturally carries a tiny header/sign, you may describe it (1-3 words, small and incidental, never hero-sized). End with 'One clear focal point, readable as a small mobile thumbnail.'>"
    },
    { "id": "B" },
    { "id": "C" }
  ],
  "recommended": "<A|B|C>",
  "why_recommended": "<one line>"
}

archetypes_chosen: the default run is evidence-closeup + tweet-card +
whiteboard-list. HARD RULE: when the script ranks or compares 3+ concrete
options, ranked-lineup is MANDATORY and replaces whiteboard-list. Be decisive — no hedging. The recommended concept maximizes click_score
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
      .select('client_name, thumbnail_visual_notes, twitter_handle')
      .eq('client_slug', script.client_slug)
      .single();

    const creatorName = client?.client_name || script.client_slug;
    const visualNotes = client?.thumbnail_visual_notes || '';
    // twitter_handle column wins; fallback: creator name stripped of spaces and punctuation ("Ryan D. Lee" -> "@RyanDLee")
    const rawHandle = String(client?.twitter_handle || '').trim().replace(/^@/, '');
    const fallbackHandle = String(creatorName).replace(/[^A-Za-z0-9_]/g, '');
    const twitterHandle = '@' + (rawHandle.replace(/[^A-Za-z0-9_]/g, '') || fallbackHandle);
    const hook = (script.full_script || '').slice(0, 800);

    const brief = await runArtDirector({
      creator_name: creatorName,
      creator_visual_notes: visualNotes,
      twitter_handle: twitterHandle,
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
    max_tokens: 12000,
    system: ART_DIRECTOR_SYSTEM,
    messages: [{ role: 'user', content: JSON.stringify(input, null, 2) }],
  });
  if (msg.stop_reason === 'max_tokens') {
    throw new Error('Art director response hit max_tokens and was truncated — raise max_tokens.');
  }
  const text = msg.content.map((b) => (b.type === 'text' ? b.text : '')).join('').trim();
  return JSON.parse(stripFences(text));
}

// ── render one concept: text-free GPT render -> caption composited -> upload ─
async function renderConcept(runId, concept, refFiles, creatorName) {
  const scene = String(concept.scene_prompt || '').trim();
  const outfitClause =
    'Also keep his OUTFIT the same as in the reference photos — the same jacket/suit, shirt, ' +
    'color and style of clothing — do not change his clothes or their color. ';
  const identity =
    'IDENTITY LOCK: keep the person\u2019s face, hairline, hair (same amount, length, color and style), ' +
    'glasses, facial hair and apparent age EXACTLY consistent with the supplied reference photos. ' +
    'Preserve the person\u2019s REAL skin character exactly as the reference photos show it — the same wrinkles, ' +
    'lines, pores, unevenness, blemishes and natural redness — do NOT beautify, smooth, even out, or de-age the skin in any way. ' +
    'Do NOT thin, shorten, recolor, add, or remove hair, and do NOT make the person look balder, younger, ' +
    'or older. ' + outfitClause + 'It must be unmistakably the ' +
    'same exact person. IDENTITY means face and hair (clothing per the outfit instruction above) — NOT the ' +
    'expression: take the facial EXPRESSION exclusively from the scene description above, never from the ' +
    'reference photos. If the references show him smiling or pleasant, do NOT carry that smile over; on a ' +
    'loss, threat or warning concept his face must be serious, hard and grave exactly as the scene describes.';
  const isLineup = (concept.archetype || '') === 'ranked-lineup';
  const isTweet = (concept.archetype || '') === 'tweet-card';
  const quality = isTweet
    ? 'DESIGNED GRAPHIC MODE: this thumbnail is a clean, professionally DESIGNED graphic composition, not a photograph of a real scene. ' +
      'The creator himself stays fully PHOTOREALISTIC — real skin texture with visible pores, real hair, bright even frontal light on the face with minimal shadow plus a clearly visible cool RIM LIGHT tracing the edges of his hair and jaw, separating him crisply from the blue background, exactly the person in the reference photos — composited large like a premium poster subject, casting ONE subtle soft drop shadow onto the background, his color grade HARMONIZED with the blue background — one shared, natural white balance — so he sits IN the design instead of cut against it, never a hard cut-out edge. ' +
      'Everything around him is intentional graphic design: a rich saturated royal-blue background with a subtle radial glow exactly as the scene describes — brightest around the white card, deepening toward the corners, no texture and no banding — and one white rounded-corner tweet card rendered as crisp, perfectly flat vector-clean UI — clean left-aligned typography with tight line spacing, even spacing, correctly spelled text exactly as quoted in the scene, the card casting one subtle soft drop shadow onto the blue background. ' +
      'Clean, balanced, deliberate — it must read as the work of a top thumbnail designer, never as a faked photograph and never as messy AI compositing.'
    : isLineup
    ? 'DESIGNED GRAPHIC MODE: this thumbnail is a clean, professionally DESIGNED graphic composition, not a photograph of a real scene. ' +
      'The creator himself stays fully PHOTOREALISTIC — real skin texture with visible pores, real hair, lit like a commercial studio portrait with a bright fresh frontal key light so the face reads clearly brighter than the backdrop, exactly the person in the reference photos — composited large like a premium poster subject, with a CLEARLY VISIBLE cool RIM LIGHT tracing his shoulders and hair, separating him crisply from the background, and his color grade HARMONIZED with the light grey backdrop — one shared, natural white balance across subject and backdrop — so he sits IN the space instead of cut against it. ' +
      'Everything around him is intentional graphic design: a completely smooth, even, UNTEXTURED seamless studio-backdrop gradient exactly as the scene describes — no grain, no plaster, no wall texture — brightest directly behind his head and shoulders and falling gently darker toward the corners, ' +
      'and floating 3D tiles with real thickness, slight spatial angles, consistent key lighting and soft shadows cast onto the backdrop and onto each other where they overlap. ' +
      'Clean, balanced, deliberate — it must read as the work of a top thumbnail designer, never as a faked photograph and never as messy AI compositing.'
    : 'PHOTOGRAPHIC REALISM (this must look like a real photo taken on a real camera, NOT a CGI render or an AI image): ' +
    'shot on a full-frame camera with an 85mm lens at a wide aperture (~f1.8), giving a genuinely SHALLOW depth of field — ' +
    'only the face is tack-sharp while the background falls into soft, natural bokeh (do NOT keep the whole frame equally sharp). ' +
    'Add the subtle imperfections of a real photograph: fine, natural film grain / sensor noise throughout, real skin texture ' +
    'with visible pores, fine lines and slight unevenness (NEVER plastic, waxy, airbrushed, over-smoothed or glossy CGI skin), ' +
    'a faint hint of chromatic aberration at the edges, and natural, slightly uneven directional lighting with real falloff and ' +
    'soft, believable catchlights in the eyes (not glassy or over-bright). ' +
    'Give the subject professional SEPARATION: a believable edge / separation light traces the person\u2019s shoulders and arms (as from a real light behind and to the side), so dark clothing never melts into a dark background — the person always reads crisply in front of the room. ' +
    'ONE REAL CAPTURE, as if a real photographer came to the creator\u2019s actual office and shot this in one frame: the person, the background ' +
    'and every prop were photographed together in the same room, lit by the SAME light from the same direction, sharing one consistent white ' +
    'balance, exposure and color grade — so nothing looks cut out, pasted, stickered, or composited. Every object and the person cast a real, soft ' +
    'CONTACT SHADOW where they meet a surface, hand or wall; foreground, subject and background sit in the same believable space and atmosphere, ' +
    'with no too-clean cut-out edges and no element that looks floated on top. ' +
    'Keep it clean, sharp and natural like a real editorial portrait photograph, ' +
    'NOT a heavily stylized, over-graded or CGI look (the scene lighting itself is set by the scene description); ' +
    'it must read as captured, not generated: avoid a flawless, over-clean, perfectly symmetrical studio look. ' +
    'ANY paper or document in the scene — held, torn, shredded, or lying on the desk — reads as natural, real paper in the scene\u2019s existing light: ' +
    'soft off-white or lightly warm-toned, a touch DARKER than the person\u2019s face, with subtle fiber texture, natural bends and soft shadows in the sheet, ' +
    'and rough, fibrous edges where it is torn — never blinding pure white, never blown out, never stiff and perfectly flat. The face stays the brightest element in the frame. ' +
    'The caption text stays sharp and fully legible, but shares the photograph\u2019s grain and white balance so it reads as part of the image, never as a pasted sticker.';
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
    highlight_word: concept.overlay?.highlight_word || null,
    highlight_style: concept.overlay?.highlight_style || 'none',
    caption_zone: concept.caption_zone || null,
    archetype: concept.archetype || '',
    angle: concept.angle || '',
  };
}

// ── tell GPT to render the caption itself, in the decided brand style ────────
const ZONE_HINTS = {
  'top-banner': 'as a clean horizontal banner across the top of the frame',
  'top-left': 'in the upper-left area of the frame',
  'top-right': 'in the upper-right area of the frame',
  'mid-left': 'in the left half of the frame, vertically centered',
  'mid-right': 'in the right half of the frame, vertically centered',
  'bottom-left': 'in the lower-left area of the frame',
};

function buildTextDirective(concept) {
  // ranked-lineup NEVER carries a caption — enforced here, not left to the art director
  if ((concept.archetype || '') === 'ranked-lineup') {
    return 'Render absolutely NO text, letters, words, captions, or numbers anywhere in the image, except the short embossed labels the scene description explicitly places on the option tiles.';
  }
  // whiteboard-list: the board IS the text — no caption pipeline, enforced here
  if ((concept.archetype || '') === 'whiteboard-list') {
    const hook = String(concept.whiteboard?.hook || '').trim();
    const items = Array.isArray(concept.whiteboard?.items)
      ? concept.whiteboard.items.map((s) => String(s).trim()).filter(Boolean)
      : [];
    const itemList = items.map((it, i) => (it === '?' ? `item ${i + 1}: a bold hand-drawn "?"` : `item ${i + 1}: "${it}"`)).join(', ');
    return (
      'WHITEBOARD TEXT — the ONLY text in the image is hand-drawn marker writing on the whiteboard, exactly as the scene describes: ' +
      (hook ? `the hook line "${hook}" in thick black marker capitals on a yellow marker-highlight bar, ` : 'the hook line in thick black marker capitals on a yellow marker-highlight bar, ') +
      (itemList ? `and the numbered items — ${itemList} — each with its circled number and simple doodle icon per the scene. ` : 'and the numbered items with their circled numbers and simple doodle icons per the scene. ') +
      'All writing reads as REAL handwriting written quickly and confidently by a real person — lines drifting slightly off horizontal, letter heights varying a touch, natural word spacing, real marker ink texture (slightly saturated stroke starts, drier stroke ends, the yellow highlight semi-transparent with visible overlapping swipes) — never a handwriting font, never perfectly even digital ink. Yet every word stays crisp, fully legible at a small mobile size, and correctly spelled with NO extra, missing, or misspelled words or letters. ' +
      'Render absolutely NO other text, letters, words, captions, numbers, logos, or watermarks anywhere else in the image.'
    );
  }
  // tweet-card: the card IS the text — no caption pipeline, enforced here
  if ((concept.archetype || '') === 'tweet-card') {
    const handle = String(concept.tweet?.handle || '').trim();
    const tweetText = String(concept.tweet?.text || '').trim();
    return (
      'TWEET CARD TEXT — the ONLY text in the image lives on the white tweet card: ' +
      (handle ? `the handle "${handle}" in bold black followed by the blue verified checkmark, ` : 'the handle in bold black followed by the blue verified checkmark, ') +
      (tweetText ? `and below it the tweet text "${tweetText}" ` : 'and below it the tweet text ') +
      'in a very heavy black flat sans-serif with NORMAL-WIDTH letterforms and even, generous letter spacing — never condensed, compressed, narrowed or squeezed, letters never touching or crowding each other — sentence case exactly as quoted, set HUGE over two lines running nearly edge to edge of the card width, crisp and perfectly legible at a small mobile size, correctly spelled with NO extra, missing, or misspelled words or letters. ' +
      'Render absolutely NO other text, letters, words, captions, numbers, logos, or watermarks anywhere else in the image.'
    );
  }
  const words = String(concept.overlay?.words || '').replace(/[\/|]+/g, ' ').replace(/\s+/g, ' ').trim().toUpperCase();
  if (!words) return 'Render no text anywhere in the image.';

  let highlight = String(concept.overlay?.highlight_word || '').trim().toUpperCase();
  if (highlight && !words.split(' ').includes(highlight)) highlight = '';
  let style = String(concept.overlay?.highlight_style || 'none').toLowerCase();
  if (!highlight || style !== 'block') style = 'none';
  const zone = ZONE_HINTS[concept.caption_zone] ? concept.caption_zone : 'top-banner';
  const isLineup = (concept.archetype || '') === 'ranked-lineup';

  const highlightRule =
    style === 'block'
      ? `Render every word in clean pure white EXCEPT "${highlight}": place that ONE word on a solid bright red (#E11D2A) rectangular block with clean straight edges and even padding, the word itself in white on the block — like a hard news banner. No other boxes, no other colors.`
      : 'Render EVERY word in clean pure white — no colored words, no colored letters anywhere in the caption.';

  return [
    'TEXT OVERLAY — render this caption baked into the image in a FIXED, consistent brand style (render it the SAME way every time):',
    `Render the exact caption "${words}" all uppercase, arranged across one or two lines — break the words wherever it reads best and forms a balanced, punchy block — in a HEAVY, extra-bold sans-serif with NORMAL-WIDTH letterforms, in the style of a hard news headline (Libre Franklin Black / Helvetica Black): thick, confident strokes and clean, simple letter shapes with even letter spacing. The letters must NEVER touch, overlap, merge, or squeeze together, and the type is never condensed, compressed, narrowed, or stretched — every single letter fully formed, correctly shaped, clearly separated from its neighbors.`,
    `PLACEMENT: put the caption ${ZONE_HINTS[zone]}, in the calm area the scene keeps clear there. It must NEVER overlap, touch, or crowd the creator's face or the hero object, and it never sits in the bottom-right corner (YouTube's duration badge covers that).`,
    highlightRule,
    'NO outline or keyline around the letters, NO box or rectangle behind the full caption, no glow, no halo, no underline.',
    isLineup
      ? 'FINISH — this caption is part of a designed graphic composition: render it as clean, crisp, perfectly flat graphic text in the same design language as the rest of the composition, evenly lit, with ONE soft drop shadow cast onto the backdrop. No photo grain, no perspective warp, no bending, never faded or washed out.'
      : 'FINISH — the caption must look professionally set INTO the photograph, never like a sticker floating on top: keep the text facing the camera perfectly flat (no perspective warp, no bending, no 3D), but let it fully share the photograph\u2019s finish — the SAME film grain and sensor noise running visibly through the letterforms at the same intensity as the scene, the same white balance and exposure (a white that clearly belongs to this scene\u2019s light, never a sterile digital #FFFFFF), the letter edges rendered with the same optical softness as the rest of the photo (never razor-sharp vector edges against a soft image), and ONE soft, natural drop shadow that grounds it. Finish it the way a professional thumbnail designer would: the white picks up a subtle tint of the scene\u2019s ambient light, the background directly behind the caption may be gently and locally deepened for contrast (a subtle designer\u2019s gradient, never a visible box or band), and where the creator\u2019s shoulder, arm or hair naturally reaches the caption area, the person may slightly OVERLAP and cut in front of the caption\u2019s nearest edge, so the text visibly sits INSIDE the scene\u2019s depth between background and subject — the FACE itself always stays fully clear of the text. It must remain bold, high-contrast and instantly legible at a small mobile size — integrated, never faded, washed out, or blended INTO transparency.',
    'Size the caption LARGE and consistent: it fills its area confidently — roughly a third of the frame, and as a top banner it runs nearly full-width with a HUGE cap height (each line roughly a sixth of the frame height), hugging the top of the frame with only a small margin — big enough to punch and read instantly on a small mobile thumbnail. When a red highlight block sits on the top line of a top banner, the block may extend to and bleed off the top edge of the frame, like a news banner cropped by the frame. Keep this scale the SAME across all three concepts: never let one come out noticeably smaller or more timid than the others.',
    'Crisp, perfectly legible, correctly spelled, with NO extra, missing, or misspelled words. Apart from this caption, the ONLY other text permitted is what the scene description explicitly calls for: a single tiny incidental real-world label on a real object (1 to 3 words, small and natural, never hero-sized) or one short marked figure on a document. No other text, letters, words, logos, or watermarks anywhere.',
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
  // Shield @handles: mask them before name-replacement, restore after — a handle
  // derived from the creator's name must never be rewritten by the sanitizer.
  const handles = [];
  p = p.replace(/@[A-Za-z0-9_]+/g, (m) => {
    handles.push(m);
    return `\u0000H${handles.length - 1}\u0000`;
  });
  const parts = String(creatorName || '').trim().split(/\s+/).filter((s) => s.length > 2);
  const variants = [creatorName, ...parts].filter((s) => s && s.length > 2);
  for (const v of variants) {
    p = p.replace(new RegExp(`\\b${escapeRegExp(v)}\\b`, 'gi'), 'the person in the reference photo');
  }
  p = p.replace(/\u0000H(\d+)\u0000/g, (_, i) => handles[Number(i)] || '');
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
