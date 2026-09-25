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

// Asynchronous responses (editor proposal, batch reload) must only land on the photo and the
// settings they were computed from. Path and object identity are not enough: reopening a photo
// reuses its cached adjustments object and undo restores an earlier one, so each view also
// carries a revision that grows on every change and never comes back (see revisionCounter).
export interface ActivePhotoSnapshot<A> {
  path: string | null;
  adjustments: A;
  revision: number;
}

// Counts the changes of the values picked by `select` in a store (zustand `subscribe`).
export const revisionCounter = <S>(
  subscribe: (listener: (state: S, previous: S) => void) => unknown,
  select: (state: S) => readonly unknown[],
): (() => number) => {
  let revision = 0;
  subscribe((state, previous) => {
    const before = select(previous);
    if (select(state).some((value, index) => !Object.is(value, before[index]))) revision += 1;
  });
  return () => revision;
};

export interface ActivePhotoSource<A> {
  current: () => ActivePhotoSnapshot<A>;
  apply: (adjustments: A) => void;
}

export const isSnapshotCurrent = <A>(snapshot: ActivePhotoSnapshot<A>, current: ActivePhotoSnapshot<A>): boolean =>
  snapshot.path !== null &&
  snapshot.path === current.path &&
  snapshot.revision === current.revision &&
  snapshot.adjustments === current.adjustments;

export type StyleApplyResult = { status: 'applied'; proposal: StyleEditorProposal } | { status: 'stale' | 'noPhoto' };

// Editor: compute the proposal for the active photo, then merge its patch only if neither the
// photo nor its settings changed while the model was running.
export const applyStyleToActivePhoto = async <A extends object>(
  source: ActivePhotoSource<A>,
  compute: (snapshot: ActivePhotoSnapshot<A>) => Promise<StyleEditorProposal>,
): Promise<StyleApplyResult> => {
  const snapshot = source.current();
  if (snapshot.path === null) return { status: 'noPhoto' };
  const proposal = await compute(snapshot);
  if (!isSnapshotCurrent(snapshot, source.current())) return { status: 'stale' };
  source.apply({ ...snapshot.adjustments, ...proposal.patch });
  return { status: 'applied', proposal };
};

// Library batch: reload the sidecar of each view (editor, library preview) only if it still shows
// the photo it showed at launch, that photo was in the batch, and its settings were not changed
// meanwhile (checked again once the sidecar is read). Returns the reloaded paths.
export const reloadActivePhotos = async <A>(
  views: { launch: ActivePhotoSnapshot<A>; source: ActivePhotoSource<A> }[],
  batch: string[],
  load: (path: string) => Promise<A | null>,
): Promise<string[]> => {
  const reloaded: string[] = [];
  for (const { launch, source } of views) {
    if (launch.path === null || !batch.includes(launch.path)) continue;
    if (!isSnapshotCurrent(launch, source.current())) continue;
    const adjustments = await load(launch.path);
    if (adjustments === null || !isSnapshotCurrent(launch, source.current())) continue;
    source.apply(adjustments);
    reloaded.push(launch.path);
  }
  return reloaded;
};
