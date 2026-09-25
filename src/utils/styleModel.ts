// "Appliquer mon style": types returned by src-tauri/src/style_model.rs and error mapping.
// The backend returns codes (`STYLE_…`); the UI shows i18n texts only (UX spec MAX-19, P4).
import type { TFunction } from 'i18next';

export const STYLE_PROVENANCE_KEY = 'styleProvenance';

export interface StyleModelInfo {
  modelId: string;
  kind: string;
  fallback: boolean;
  trainingSamples: number;
  minimumForLearnedModel: number;
}

export interface StyleMergeOutcome {
  applied: string[];
  manual: string[];
  skipped: boolean;
}

export interface StyleEditorProposal {
  model: StyleModelInfo;
  patch: Record<string, unknown>;
  outcome: StyleMergeOutcome;
}

export interface StyleApplySummary {
  model: StyleModelInfo;
  updated: number;
  skipped: number;
  failed: number;
  errors: string[];
}

export type StyleErrorKind = 'noModel' | 'modelOutdated' | 'modelDamaged' | 'generic';

const DAMAGED_CODES = new Set([
  'STYLE_MODEL_MANIFEST_INVALID',
  'STYLE_MODEL_ARTIFACT_MISSING',
  'STYLE_MODEL_ARTIFACT_HASH_MISMATCH',
  'STYLE_MODEL_LOAD_FAILED',
]);

export const styleErrorCode = (error: unknown): string =>
  String(error ?? '')
    .split(':')[0]
    .trim();

export const classifyStyleError = (error: unknown): StyleErrorKind => {
  const code = styleErrorCode(error);
  if (code === 'STYLE_MODEL_UNAVAILABLE') return 'noModel';
  if (code === 'STYLE_MODEL_CONTRACT_MISMATCH') return 'modelOutdated';
  if (DAMAGED_CODES.has(code)) return 'modelDamaged';
  return 'generic';
};

export const styleErrorMessage = (t: TFunction, error: unknown): string => {
  switch (classifyStyleError(error)) {
    case 'noModel':
      return `${t('style.noModel')}. ${t('style.noModelHint')} ${t('style.trainCliHint')}`;
    case 'modelOutdated':
      return t('style.errors.modelOutdated');
    case 'modelDamaged':
      return t('style.errors.modelDamaged');
    default:
      return t('style.errors.generic');
  }
};

export const styleFallbackNotice = (t: TFunction, model: StyleModelInfo): string | null =>
  model.fallback
    ? t('style.fallbackNotice', { count: model.trainingSamples, recommended: model.minimumForLearnedModel })
    : null;

export const styleBatchMessages = (t: TFunction, summary: StyleApplySummary): string[] => {
  const messages = [t('style.batch.done', { count: summary.updated })];
  if (summary.skipped > 0) messages.push(t('style.batch.skipped', { count: summary.skipped }));
  if (summary.failed > 0) messages.push(t('style.batch.failed', { count: summary.failed }));
  return messages;
};
