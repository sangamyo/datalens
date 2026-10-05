export const PII_KIND_LABELS: Record<string, string> = {
  email: 'Email',
  phone: 'Phone',
  ip_address: 'IP address',
  credit_card: 'Credit card',
  aadhaar: 'Aadhaar',
  secret: 'Secret / API key',
  ssn: 'SSN',
}

export const piiKindLabel = (k: string) => PII_KIND_LABELS[k] ?? k.replace(/_/g, ' ')

const KNOWN_KINDS = new Set(Object.keys(PII_KIND_LABELS))
export const piiKindClass = (k: string) => `pii--${KNOWN_KINDS.has(k) ? k : 'other'}`
