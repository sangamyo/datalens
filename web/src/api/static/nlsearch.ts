// TypeScript port of the rule-based parser in api/app/nlsearch.py (`parse_rules`, `resolve_category`) for the
// static demo build. Same regexes, same order of passes, same output: keep the two in sync.
import type { QCStatus } from '../types'

/** SampleFilter exactly as the API serialises it (every key present, null when unset). */
export interface ParsedFilter {
  dataset_id: number | null
  qc_status: QCStatus[] | null
  category: string | null
  lang: string | null
  min_tokens: number | null
  max_tokens: number | null
  text_contains: string | null
  failed_check: string | null
}

const STATUS_ORDER: QCStatus[] = ['pending', 'pass', 'warn', 'fail']

const SHORT_ANSWER_MAX_TOKENS = 40
const LONG_SAMPLE_MIN_TOKENS = 400
const WORDS_TO_TOKENS = 1.3
const CHARS_PER_TOKEN = 4
const APPROX_TOLERANCE = 0.2 // "about 200 tokens" -> 160..240

const CATEGORIES = [
  'open_qa',
  'closed_qa',
  'general_qa',
  'classification',
  'brainstorming',
  'information_extraction',
  'summarization',
  'creative_writing',
  'generation',
  'rewrite',
  'chat',
  'coding',
]

const CATEGORY_VARIANTS: Record<string, string[]> = {
  open_qa: ['open_qa', 'open qa', 'open-qa', 'openqa'],
  closed_qa: ['closed_qa', 'closed qa', 'closed-qa', 'closedqa'],
  general_qa: ['general_qa', 'general qa', 'general-qa'],
  classification: ['classification', 'classify'],
  brainstorming: ['brainstorming', 'brainstorm'],
  information_extraction: ['information_extraction', 'information extraction', 'extract', 'extraction'],
  summarization: ['summarization', 'summarisation', 'summarize', 'summarise', 'summary'],
  creative_writing: ['creative_writing', 'creative writing', 'creative'],
  generation: ['generation', 'generate'],
  rewrite: ['rewrite', 'rewriting'],
  chat: ['chat', 'conversation'],
  coding: ['coding', 'code', 'programming'],
}

const normCat = (s: string) => s.trim().toLowerCase().replace(/[\s_-]+/g, ' ')

/** Map a canonical category (e.g. "brainstorming") to the spelling a dataset actually uses. */
export function resolveCategory(category: string, available: Iterable<string | null>): string {
  const avail = [...available].filter((a): a is string => !!a)
  const want = normCat(category)
  for (const a of avail) if (normCat(a) === want) return a
  const variants = new Set((CATEGORY_VARIANTS[want.replace(/ /g, '_')] ?? []).map(normCat))
  for (const a of avail) if (variants.has(normCat(a))) return a
  return category
}

// ---------------------------------------------------------------------------------------------
// Rule-based parser
// ---------------------------------------------------------------------------------------------

const W = String.raw`(?<![\w-])` // left word boundary that also treats '-' as part of a word
const E = String.raw`(?![\w-])`

const NUM = String.raw`(\d+(?:[.,]\d+)?)\s*(k)?`
const UNIT = String.raw`(tokens?|toks?|words?|characters?|chars?)`

const MIN_CMP =
  String.raw`longer\s+than|more\s+than|greater\s+than|bigger\s+than|larger\s+than|over|above|exceeding|exceeds|` +
  String.raw`at\s+least|no\s+less\s+than|no\s+fewer\s+than|no\s+shorter\s+than|minimum(?:\s+of)?|min\.?|>=|=>|>`
const MAX_CMP =
  String.raw`shorter\s+than|less\s+than|fewer\s+than|under|below|within|at\s+most|no\s+more\s+than|` +
  String.raw`no\s+longer\s+than|up\s+to|maximum(?:\s+of)?|max\.?|<=|=<|<`
const LENGTH_CMP = String.raw`longer\s+than|shorter\s+than|no\s+longer\s+than|no\s+shorter\s+than`

const BETWEEN_RE = new RegExp(
  String.raw`(?:between|from)\s+${NUM}\s*${UNIT}?\s*(?:and|to|-)\s*${NUM}\s*${UNIT}(?![a-z])` +
    String.raw`|(?<![\w.])${NUM}\s*${UNIT}?\s*(?:-|to)\s*${NUM}\s*${UNIT}(?![a-z])`,
  'gi',
)
const CMP_RE = new RegExp(String.raw`(?<![a-z])(${MIN_CMP}|${MAX_CMP})\s*${NUM}\s*${UNIT}?(?![a-z])`, 'gi')
const APPROX_RE = new RegExp(
  String.raw`(?<![a-z])(?:about|around|roughly|approximately|approx\.?|circa|~)\s*${NUM}\s*${UNIT}(?![a-z])`,
  'gi',
)
const PLUS_RE = new RegExp(String.raw`(?<![\w.])${NUM}\s*\+\s*${UNIT}(?![a-z])`, 'gi') // "500+ tokens"
const POSTFIX_RE = new RegExp(
  String.raw`(?<![\w.])${NUM}\s*${UNIT}\s*(\+|or\s+(?:more|longer|above|over|greater)|plus` +
    String.raw`|or\s+(?:less|fewer|shorter|under|below))(?![a-z])`,
  'gi',
)
const MIN_CMP_RE = new RegExp(String.raw`^(?:${MIN_CMP})$`, 'i')
const LENGTH_CMP_RE = new RegExp(String.raw`^(?:${LENGTH_CMP})$`, 'i')

const SHORT_RE = new RegExp(
  String.raw`\b(?:short|brief|terse|concise)(?=\s+(?:[a-z_-]+\s+){0,2}?(?:answers?|responses?|replies|reply|outputs?|completions?|samples?|` +
    String.raw`examples?|ones|rows?|entries|items?)\b)`,
  'i',
)
const LONG_RE = new RegExp(
  String.raw`\b(?:long|lengthy|verbose)(?=\s+(?:[a-z_-]+\s+){0,2}?(?:answers?|responses?|replies|reply|outputs?|completions?|samples?|` +
    String.raw`examples?|ones|rows?|entries|items?|prompts?|instructions?|texts?)\b)`,
  'i',
)

// A check phrase may be wrapped in "failed the ... check" / "... issues": that wrapping belongs to
// the check, not to the qc_status.
const CHECK_PREFIX =
  String.raw`(?:\bfail(?:s|ed|ing)?\s+(?:(?:the|on|for|by|at)\s+)+` +
  String.raw`|\b(?:flagged|bad|poor|has|have|having|with|contain(?:s|ing)?|show(?:s|ing)?)` +
  String.raw`\s+(?:(?:the|on|for|by|as|an?|any|some)\s+)*)?`
const CHECK_SUFFIX = String.raw`(?:\s+(?:issues?|problems?|errors?|warnings?|flags?|checks?|failures?|detected|found|hits?))*`

// (pattern, check_name) in priority order: more specific phrases first; matched spans are
// consumed so e.g. "repeated characters" never also counts as "repeated" (exact_duplicate).
const CHECK_PATTERNS: [string, string][] = [
  [
    String.raw`near[_\s-]*dup(?:e|es|licates?|licated)?s?|nearly\s+(?:identical|the\s+same)|almost\s+(?:identical|the\s+same)|` +
      String.raw`(?:very\s+)?similar(?:\s+(?:samples?|prompts?|ones|examples?))?|paraphras(?:ed|es)\s+dup(?:e|es|licates?)s?|` +
      String.raw`fuzzy\s+dup(?:e|es|licates?)s?|semantic\s+dup(?:e|es|licates?)s?|paraphrased|` +
      String.raw`(?:almost|near|nearly)\s+duplicated?`,
    'near_duplicate',
  ],
  [
    String.raw`formatting|format(?:ted)?\s+(?:badly|poorly|wrong(?:ly)?)|badly\s+formatted|poorly\s+formatted|mis-?formatted|` +
      String.raw`malformed|broken\s+(?:code|markdown|formatting|json|html)|unclosed\s+(?:code\s+)?(?:blocks?|fences?)|` +
      String.raw`unbalanced\s+(?:code\s+)?(?:blocks?|fences?|brackets?)|truncat(?:ed|ion)|cut\s+off|cut-off|` +
      String.raw`incomplete(?:\s+(?:answers?|responses?|sentences?))?|repeated\s+(?:characters?|chars?|letters?|symbols?|punctuation)|` +
      String.raw`repetitive(?:\s+(?:characters?|text))?|garbled|weird\s+characters?|mojibake`,
    'formatting',
  ],
  [
    String.raw`exact[_\s-]*dup(?:e|es|licates?|licated)?s?|dup(?:e|es)|dup(?:licate|licates|licated|lication)s?|` +
      String.raw`repeated(?:\s+(?:samples?|prompts?|rows?|entries|examples?))?|repeats|copies|copied|identical(?:\s+(?:samples?|prompts?|rows?))?`,
    'exact_duplicate',
  ],
  [
    String.raw`pii|personal(?:ly)?\s+(?:data|info(?:rmation)?|details|identifiable(?:\s+information)?)|private\s+(?:data|info(?:rmation)?)|` +
      String.raw`e-?mails?(?:\s+addresses?)?|email\s+addresses|phone(?:\s+numbers?)?|telephone(?:\s+numbers?)?|` +
      String.raw`sensitive(?:\s+(?:data|info(?:rmation)?|content))?|secrets?|api\s+keys?|passwords?|credentials?|` +
      String.raw`credit[\s-]*cards?(?:\s+numbers?)?|ssns?|social\s+security(?:\s+numbers?)?|ip\s+addresses?|addresses`,
    'pii',
  ],
  [
    String.raw`non[\s_-]*english|not\s+(?:in\s+)?english|foreign(?:\s+languages?)?|other\s+languages?|another\s+language|` +
      String.raw`different\s+languages?|multilingual|hindi|spanish|french|german|chinese|japanese|korean|arabic|russian|` +
      String.raw`portuguese|italian|dutch|turkish|vietnamese|indonesian|bengali|urdu|tamil|telugu|marathi`,
    'non_english',
  ],
  [
    String.raw`refus(?:als?|ing|es|ed|e)|refusal[_\s-]*boilerplate|as\s+an\s+ai(?:\s+(?:language\s+)?model)?|` +
      String.raw`(?:ai\s+)?boilerplate|canned(?:\s+(?:answers?|responses?|replies))?|i\s+can(?:'|no)?t\s+help|i\s+cannot|` +
      String.raw`declin(?:es|ed|ing)|disclaimers?|apologi[sz](?:es|ing|e)|sorry`,
    'refusal_boilerplate',
  ],
  [
    String.raw`empty(?:[_\s-]*or[_\s-]*short)?|blank|too\s+short|very\s+short|extremely\s+short|one[\s-]+word(?:\s+(?:answers?|responses?))?|` +
      String.raw`single[\s-]+word(?:\s+(?:answers?|responses?))?|missing\s+(?:answers?|responses?)|no\s+(?:answers?|responses?)|` +
      String.raw`trivial(?:\s+(?:answers?|responses?))?`,
    'empty_or_short',
  ],
  [
    String.raw`length[_\s-]*outliers?|outliers?|too\s+long|very\s+long|extremely\s+long|overly\s+long|` +
      String.raw`unusual(?:ly)?\s+(?:length|long|sized?)|abnormal(?:ly)?\s+(?:length|long)|extreme\s+lengths?|weird\s+lengths?|` +
      String.raw`odd\s+lengths?`,
    'length_outlier',
  ],
]
const CHECK_RES = CHECK_PATTERNS.map(
  ([p, name]) => [new RegExp(`${CHECK_PREFIX}${W}(?:${p})${E}${CHECK_SUFFIX}`, 'gi'), name] as const,
)

const STATUS_PATTERNS: [string, QCStatus[]][] = [
  // Negations / specific phrases first; their spans are consumed before the generic words run.
  [
    String.raw`\b(?:not|never|didn'?t|did\s+not|haven'?t|hasn'?t|have\s+not|has\s+not)\s+(?:been\s+|yet\s+)?` +
      String.raw`(?:qc'?d|checked|scored|evaluated|analy[sz]ed|inspected|reviewed|run)(?:\s+yet)?\b`,
    ['pending'],
  ],
  [String.raw`\b(?:not|didn'?t|did\s+not|doesn'?t|does\s+not|don'?t)\s+pass(?:ed|ing)?\b|\bnon[\s-]?passing\b`, ['warn', 'fail']],
  [String.raw`\b(?:no|without|zero)\s+(?:problems?|issues?|warnings?|errors?|failures?|flags?)\b`, ['pass']],
  [String.raw`\b(?:only\s+warn(?:ings?|s)?|warn(?:ings?|s)?\s+only|just\s+warn(?:ings?|s)?)\b`, ['warn']],
  [String.raw`\bneeds?\s+(?:a\s+)?(?:review|attention|fixing|cleaning|cleanup)\b|\bto\s+review\b`, ['warn', 'fail']],
  [String.raw`\b(?:pending|unchecked|unscored|unprocessed|unreviewed|awaiting\s+qc|qc\s+pending|in\s+progress)\b`, ['pending']],
  [String.raw`\b(?:high|good|top)[-\s]quality\b`, ['pass']],
  [String.raw`\b(?:low|poor|bad)[-\s]quality\b`, ['warn', 'fail']],
  [
    String.raw`\b(?:warn(?:ings?|s|ed)?|problems?|problematic|issues?|flagged|suspicious|questionable|errors?|` +
      String.raw`anomal(?:y|ies|ous)|noisy|dirty)\b`,
    ['warn', 'fail'],
  ],
  [String.raw`\b(?:fail(?:s|ed|ing|ures?)?|broken|bad|worst|garbage|junk)\b`, ['fail']],
  [String.raw`\b(?:pass(?:es|ed|ing)?|clean|good|ok|okay|fine|valid|perfect)\b`, ['pass']],
]
const STATUS_RES = STATUS_PATTERNS.map(([p, s]) => [new RegExp(p, 'gi'), s] as const)

// (pattern, canonical category)
const CATEGORY_PATTERNS: [string, string][] = [
  [String.raw`open[\s_-]*qa|open[\s-]+ended(?:\s+(?:questions?|qa))?|open\s+questions?`, 'open_qa'],
  [String.raw`closed[\s_-]*qa|closed[\s-]+book(?:\s+(?:questions?|qa))?|reading\s+comprehension|closed\s+questions?`, 'closed_qa'],
  [String.raw`general[\s_-]*qa|general\s+(?:knowledge\s+)?questions?`, 'general_qa'],
  [String.raw`information[\s_-]+extraction|extraction|extract(?:ing|s)?`, 'information_extraction'],
  [String.raw`classification|classif(?:y|ying|ied)|categori[sz]ation`, 'classification'],
  [String.raw`brainstorm(?:ing|s)?|ideas?`, 'brainstorming'],
  [String.raw`summar(?:y|ies|i[sz]ation|i[sz]e|i[sz]ing|i[sz]ations)`, 'summarization'],
  [String.raw`creative[\s_-]*writing|creative|stor(?:y|ies)|poems?|poetry|fiction|haikus?|songs?|lyrics`, 'creative_writing'],
  [String.raw`generation|generat(?:e|ing)`, 'generation'],
  [String.raw`rewrit(?:e|es|ing|ten)|rephras(?:e|ing)`, 'rewrite'],
  [String.raw`chats?|conversations?|conversational|dialog(?:ue)?s?|multi[\s-]?turn`, 'chat'],
  [String.raw`coding|code|programming|programs?`, 'coding'],
]
const CATEGORY_RES = CATEGORY_PATTERNS.map(([p, c]) => [new RegExp(`${W}(?:${p})${E}`, 'gi'), c] as const)
const CANONICAL_CATEGORY_RE = new RegExp(String.raw`(?<![\w])(` + CATEGORIES.join('|') + String.raw`)(?![\w])`, 'i')

const ENGLISH_RE = /(?<![\w-])(?:in\s+)?english(?:[\s-]+only)?(?![\w-])|\bonly\s+english\b/i

const QUOTE_RE = /"([^"]+)"|“([^”]+)”|'([^']{2,})'(?!\w)|`([^`]+)`/
const TEXT_TRIGGER =
  String.raw`mention(?:s|ing|ed)?|contain(?:s|ing)?|includ(?:es|ing)|referenc(?:es|ing)|` +
  String.raw`(?:that|which|who)\s+(?:talk|talks|are|is)\s+about|talk(?:s|ing)?\s+about|about|regarding|` +
  String.raw`related\s+to|on\s+the\s+topic\s+of|with\s+the\s+(?:word|phrase|term|text)|with\s+(?:word|phrase|term|text)|` +
  String.raw`using\s+the\s+(?:word|phrase|term)|that\s+say|saying|says|matching|with\s+keyword`
const TEXT_RE = new RegExp(String.raw`\b(?:${TEXT_TRIGGER})\s+(?:the\s+(?:word|phrase|term|topic)\s+)?(?<x>.+)$`, 'dgi')
const TEXT_TRIGGER_END_RE = new RegExp(String.raw`\b(?:${TEXT_TRIGGER})(?=\s{2,}|\s*$)`, 'gi')
const TEXT_STOP = new RegExp(
  String.raw`\s+(?:in|with|from|that|which|where|and|or|but|having|whose|longer|shorter|over|under|above|below|` +
    String.raw`more|less|fewer|between|at|than|for|of\s+category|category|status|tokens?|words?|marked|flagged|failed|` +
    String.raw`failing|passing|passed|only|written|not)\b.*$`,
  'i',
)
const LEADING_JUNK = /^(?:the|a|an|some|any)\s+/i
const FILLER_WORDS = new Set([
  'samples', 'sample', 'examples', 'example', 'rows', 'row', 'entries', 'entry', 'items', 'item', 'ones', 'data',
  'show', 'find', 'get', 'list', 'give', 'me', 'all', 'the', 'a', 'an', 'any', 'some', 'with', 'that', 'which',
  'are', 'is', 'in', 'of', 'and', 'or', 'please', 'only', 'answers', 'answer', 'responses', 'response', 'prompts',
  'prompt', 'instructions', 'instruction', 'dataset', 'records', 'record', 'things', 'stuff', 'everything',
  'it', 'them', 'those', 'these', 'this', 'qc', 'check', 'checks', 'status', 'category',
])

/** Lower-cased working copy of the query whose matched spans get blanked out. */
class Text {
  s: string
  constructor(s: string) {
    this.s = s
  }

  blank(a: number, b: number) {
    this.s = this.s.slice(0, a) + ' '.repeat(b - a) + this.s.slice(b)
  }

  consume(m: RegExpMatchArray) {
    this.blank(m.index!, m.index! + m[0].length)
  }
}

/** Python's round(): half to even. */
function pyRound(x: number): number {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

function num(value: string, k: string | undefined): number {
  const parts = value.split(',')
  const n = parts.length === 2 && parts[1].length === 3 ? Number(value.replace(/,/g, '')) : Number(value.replace(/,/g, '.'))
  return k ? n * 1000 : n
}

function toTokens(n: number, unit: string | undefined): number {
  const u = (unit || 'tokens').toLowerCase()
  if (u.startsWith('word')) return Math.max(0, pyRound(n * WORDS_TO_TOKENS))
  if (u.startsWith('char')) return Math.max(0, pyRound(n / CHARS_PER_TOKEN))
  return Math.max(0, pyRound(n))
}

const STRICT_CMP = new Set([
  'longer than', 'more than', 'greater than', 'bigger than', 'larger than', 'over', 'above', 'exceeding', 'exceeds', '>',
  'shorter than', 'less than', 'fewer than', 'under', 'below', '<',
])

function parseTokens(t: Text): [number | null, number | null] {
  let lo: number | null = null
  let hi: number | null = null

  for (const m of [...t.s.matchAll(BETWEEN_RE)]) {
    const [a, ak, au, b, bk, bu] = m[1] !== undefined ? m.slice(1, 7) : m.slice(7, 13)
    const unit = bu || au
    ;[lo, hi] = [toTokens(num(a, ak), au || unit), toTokens(num(b, bk), unit)].sort((x, y) => x - y)
    t.consume(m)
  }

  for (const m of [...t.s.matchAll(APPROX_RE)]) {
    const center = toTokens(num(m[1], m[2]), m[3])
    lo = Math.floor(center * (1 - APPROX_TOLERANCE))
    hi = Math.ceil(center * (1 + APPROX_TOLERANCE))
    t.consume(m)
  }

  for (const m of [...t.s.matchAll(PLUS_RE)]) {
    lo = toTokens(num(m[1], m[2]), m[3])
    t.consume(m)
  }

  for (const m of [...t.s.matchAll(POSTFIX_RE)]) {
    const n = toTokens(num(m[1], m[2]), m[3])
    if (/less|fewer|shorter|under|below/i.test(m[4])) hi = n
    else lo = n
    t.consume(m)
  }

  for (const m of [...t.s.matchAll(CMP_RE)]) {
    const [, cmp, value, k, unit] = m
    const cmpNorm = cmp.trim().toLowerCase().replace(/\s+/g, ' ')
    // a bare number ("over 3") only counts as a length after "longer/shorter than" or with
    // a "1k" suffix; otherwise it may mean something else entirely
    if (unit === undefined && !LENGTH_CMP_RE.test(cmpNorm) && !k) continue
    const n = toTokens(num(value, k), unit)
    const strict = STRICT_CMP.has(cmpNorm)
    if (MIN_CMP_RE.test(cmpNorm)) lo = strict ? n + 1 : n
    else hi = strict ? Math.max(0, n - 1) : n
    t.consume(m)
  }
  return [lo, hi]
}

const NEGATED_RE = /(?:\b(?:without|no|zero|free\s+of|excluding|exclude|except|not|non)\s*-?\s*(?:(?:any|the|a)\s+)?)$/i

/** First check mentioned (by position). Negated mentions ("without PII") are consumed and ignored. */
function parseChecks(t: Text): string | null {
  const found: [number, string][] = []
  for (const [rx, name] of CHECK_RES) {
    for (const m of [...t.s.matchAll(rx)]) {
      const neg = NEGATED_RE.exec(t.s.slice(0, m.index))
      t.consume(m)
      if (neg) {
        t.blank(neg.index, neg.index + neg[0].length)
        continue
      }
      found.push([m.index!, name])
    }
  }
  return minFound(found)
}

/** Python's min() over (position, name) tuples. */
function minFound(found: [number, string][]): string | null {
  if (!found.length) return null
  return found.reduce((a, b) => (b[0] < a[0] || (b[0] === a[0] && b[1] < a[1]) ? b : a))[1]
}

function parseStatus(t: Text): QCStatus[] | null {
  const statuses = new Set<QCStatus>()
  for (const [rx, sts] of STATUS_RES) {
    for (const m of [...t.s.matchAll(rx)]) {
      sts.forEach((s) => statuses.add(s))
      t.consume(m)
    }
  }
  const out = STATUS_ORDER.filter((s) => statuses.has(s))
  return out.length ? out : null
}

function parseCategory(t: Text): string | null {
  const m = CANONICAL_CATEGORY_RE.exec(t.s)
  if (m) {
    t.consume(m)
    return m[1].toLowerCase()
  }
  const found: [number, string][] = []
  for (const [rx, cat] of CATEGORY_RES) {
    for (const mm of [...t.s.matchAll(rx)]) {
      found.push([mm.index!, cat])
      t.consume(mm)
    }
  }
  return minFound(found)
}

/** True if `phrase` is only status/check/category/length words (plus filler), not free text. */
function isReserved(phrase: string): boolean {
  const p = phrase.trim().toLowerCase()
  if (!p) return true
  if (/^[\d.,]+\s*k?\s*(?:tokens?|words?|chars?|characters?)?$/.test(p)) return true
  const probe = new Text(p)
  parseChecks(probe)
  parseStatus(probe)
  parseCategory(probe)
  return (probe.s.match(/[a-z0-9']+/g) ?? []).every((w) => FILLER_WORDS.has(w))
}

const stripChars = (s: string, chars: string) => {
  let a = 0
  let b = s.length
  while (a < b && chars.includes(s[a])) a++
  while (b > a && chars.includes(s[b - 1])) b--
  return s.slice(a, b)
}

/** 'mentioning python', 'about tax law' -> the topic words (max 4), unless they are reserved. */
function parseTextPhrase(t: Text, original: string): string | null {
  const src = original.length === t.s.length ? original : t.s
  for (const m of [...t.s.matchAll(TEXT_RE)]) {
    let start = m.indices!.groups!.x[0]
    let seg = t.s.slice(start)
    start += seg.length - seg.trimStart().length
    seg = t.s.slice(start)
    const gap = /\s{2,}/.exec(seg) // a blanked-out (already parsed) region ends the phrase
    let end = start + (gap ? gap.index : seg.trimEnd().length)
    let raw = src.slice(start, end).replace(TEXT_STOP, '')
    end = start + raw.length
    raw = stripChars(raw.replace(LEADING_JUNK, ''), ' \t.,;:!?')
    const phrase = raw.split(/\s+/).filter(Boolean).slice(0, 4).join(' ')
    if (!phrase || isReserved(phrase)) continue
    t.blank(m.index!, end)
    return phrase
  }
  return null
}

/** Deterministic keyword/regex parser. Always succeeds (possibly with an empty filter). */
export function parseRules(query: string, datasetId: number | null = null): ParsedFilter {
  let q = query.split(/\s+/).filter(Boolean).join(' ')
  let textContains: string | null = null

  // 1. quoted phrases are literal text searches
  const qm = QUOTE_RE.exec(q)
  if (qm) {
    textContains = qm.slice(1).find((g) => g !== undefined)!.trim() || null
    q = q.slice(0, qm.index) + ' '.repeat(qm[0].length) + q.slice(qm.index + qm[0].length)
    q = q.replace(new RegExp(QUOTE_RE.source, 'g'), (s) => ' '.repeat(s.length))
  }

  const original = q
  const t = new Text(q.toLowerCase())

  // 2. explicit token / word / char bounds
  let [minTokens, maxTokens] = parseTokens(t)

  // 3. free-text topic ("mentioning python", "about the french revolution")
  if (textContains === null) textContains = parseTextPhrase(t, original)
  else for (const trig of [...t.s.matchAll(TEXT_TRIGGER_END_RE)]) t.consume(trig)

  // 4. quality checks (consumes "repeated characters", "too long", "failed the pii check", ...)
  const failedCheck = parseChecks(t)

  // 5. "short answers" / "long samples"
  let m: RegExpExecArray | null
  if (maxTokens === null && (m = SHORT_RE.exec(t.s))) {
    maxTokens = SHORT_ANSWER_MAX_TOKENS
    t.consume(m)
  }
  if (minTokens === null && (m = LONG_RE.exec(t.s))) {
    minTokens = LONG_SAMPLE_MIN_TOKENS
    t.consume(m)
  }

  // 6. qc status
  const qcStatus = parseStatus(t)

  // 7. category
  const category = parseCategory(t)

  // 8. language
  let lang: string | null = null
  if (failedCheck === 'non_english') lang = 'other'
  else if ((m = ENGLISH_RE.exec(t.s))) {
    lang = 'en'
    t.consume(m)
  }

  if (minTokens !== null && maxTokens !== null && minTokens > maxTokens) [minTokens, maxTokens] = [maxTokens, minTokens]

  return {
    dataset_id: datasetId,
    qc_status: qcStatus,
    category,
    lang,
    min_tokens: minTokens,
    max_tokens: maxTokens,
    text_contains: textContains,
    failed_check: failedCheck,
  }
}
