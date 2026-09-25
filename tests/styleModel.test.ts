import assert from 'node:assert/strict';
import test from 'node:test';
import type { TFunction } from 'i18next';
import { createStore } from 'zustand/vanilla';
import type { ActivePhotoSnapshot, ActivePhotoSource, StyleEditorProposal } from '../src/utils/styleModel.ts';
import {
  applyStyleToActivePhoto,
  classifyStyleError,
  reloadActivePhotos,
  revisionCounter,
  styleBatchMessages,
  styleErrorMessage,
  styleFallbackNotice,
} from '../src/utils/styleModel.ts';

const t = ((key: string, options?: Record<string, unknown>) =>
  options ? `${key}${JSON.stringify(options)}` : key) as unknown as TFunction;

test('style error codes map to UX error kinds', () => {
  assert.equal(classifyStyleError('STYLE_MODEL_UNAVAILABLE'), 'noModel');
  assert.equal(classifyStyleError('STYLE_MODEL_CONTRACT_MISMATCH'), 'modelOutdated');
  assert.equal(classifyStyleError('STYLE_MODEL_ARTIFACT_HASH_MISMATCH'), 'modelDamaged');
  assert.equal(classifyStyleError('STYLE_MODEL_MANIFEST_INVALID: expected value at line 1'), 'modelDamaged');
  assert.equal(classifyStyleError('STYLE_INFERENCE_FAILED: boom'), 'generic');
  assert.equal(classifyStyleError(undefined), 'generic');
});

test('raw backend strings never reach the message', () => {
  const message = styleErrorMessage(t, 'STYLE_INFERENCE_FAILED: /home/someone/secret.jpg');
  assert.equal(message, 'style.errors.generic');
  assert.match(styleErrorMessage(t, 'STYLE_MODEL_UNAVAILABLE'), /style\.trainCliHint/);
});

test('fallback notice only for fallback models', () => {
  const model = { modelId: 'm', kind: 'cluster_mean', fallback: true, trainingSamples: 30, minimumForLearnedModel: 50 };
  assert.equal(styleFallbackNotice(t, model), 'style.fallbackNotice{"count":30,"recommended":50}');
  assert.equal(styleFallbackNotice(t, { ...model, fallback: false }), null);
});

test('batch summary lists skipped and failed photos only when present', () => {
  const model = { modelId: 'm', kind: 'mlp', fallback: false, trainingSamples: 300, minimumForLearnedModel: 50 };
  assert.deepEqual(styleBatchMessages(t, { model, updated: 52, skipped: 0, failed: 0, errors: [] }), [
    'style.batch.done{"count":52}',
  ]);
  assert.equal(styleBatchMessages(t, { model, updated: 52, skipped: 12, failed: 1, errors: [] }).length, 3);
});

type Settings = Record<string, number>;

// A zustand store view whose settings object is replaced on every change, with the same
// revision counter as the editor and library stores. `open` may reuse a cached settings object
// and `undo` restores the previous object, as useAppNavigation and the editor history do.
const view = (path: string | null, adjustments: Settings) => {
  const store = createStore<{ path: string | null; adjustments: Settings }>(() => ({ path, adjustments }));
  const revision = revisionCounter(store.subscribe, (state) => [state.path, state.adjustments]);
  const history: Settings[] = [];
  const applied: Settings[] = [];
  const source: ActivePhotoSource<Settings> = {
    current: () => ({ ...store.getState(), revision: revision() }),
    apply: (next) => {
      applied.push(next);
      store.setState({ adjustments: next });
    },
  };
  return {
    source,
    applied,
    edit: (patch: Settings) => {
      const { adjustments: previous } = store.getState();
      history.push(previous);
      store.setState({ adjustments: { ...previous, ...patch } });
    },
    undo: () => store.setState({ adjustments: history.pop() }),
    open: (nextPath: string, next: Settings) => {
      history.length = 0;
      store.setState({ path: nextPath, adjustments: next });
    },
    get state() {
      return store.getState();
    },
  };
};

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
};

const proposal = (patch: Settings): StyleEditorProposal => ({
  model: { modelId: 'm', kind: 'ridge', fallback: false, trainingSamples: 300, minimumForLearnedModel: 50 },
  patch,
  outcome: { applied: Object.keys(patch), manual: [], skipped: false },
});

test('editor proposal is applied when nothing changed during inference', async () => {
  const editor = view('a.nef', { exposure: 0, clarity: 7 });
  const result = await applyStyleToActivePhoto(editor.source, async () => proposal({ exposure: 1.2 }));
  assert.equal(result.status, 'applied');
  assert.deepEqual(editor.state.adjustments, { exposure: 1.2, clarity: 7 });
});

test('editor proposal is dropped when a slider moved during inference', async () => {
  const editor = view('a.nef', { exposure: 0, contrast: 0 });
  const pending = deferred<StyleEditorProposal>();
  const running = applyStyleToActivePhoto(editor.source, () => pending.promise);
  editor.edit({ exposure: 0.7 });
  pending.resolve(proposal({ exposure: 1.2, contrast: 15 }));
  assert.equal((await running).status, 'stale');
  assert.deepEqual(editor.applied, []);
  assert.deepEqual(editor.state.adjustments, { exposure: 0.7, contrast: 0 });
});

test('editor proposal is dropped when another photo was opened during inference', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const pending = deferred<StyleEditorProposal>();
  let computedFor: string | null = null;
  const running = applyStyleToActivePhoto(editor.source, (snapshot) => {
    computedFor = snapshot.path;
    return pending.promise;
  });
  editor.open('b.nef', { exposure: -0.3 });
  pending.resolve(proposal({ exposure: 1.2 }));
  assert.equal((await running).status, 'stale');
  assert.equal(computedFor, 'a.nef');
  assert.deepEqual(editor.state, { path: 'b.nef', adjustments: { exposure: -0.3 } });
});

test('editor proposal is dropped when the same photo was reopened during inference', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const pending = deferred<StyleEditorProposal>();
  const running = applyStyleToActivePhoto(editor.source, () => pending.promise);
  editor.open('b.nef', { exposure: 0 });
  editor.open('a.nef', { exposure: 0 });
  pending.resolve(proposal({ exposure: 1.2 }));
  assert.equal((await running).status, 'stale');
  assert.deepEqual(editor.applied, []);
});

test('editor proposal is dropped when the photo was reopened with its cached settings object', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const cachedA = editor.state.adjustments;
  const pending = deferred<StyleEditorProposal>();
  const running = applyStyleToActivePhoto(editor.source, () => pending.promise);
  editor.open('b.nef', { exposure: -0.3 });
  editor.open('a.nef', cachedA);
  assert.equal(editor.state.adjustments, cachedA);
  pending.resolve(proposal({ exposure: 1.2 }));
  assert.equal((await running).status, 'stale');
  assert.deepEqual(editor.applied, []);
  assert.equal(editor.state.adjustments, cachedA);
});

test('editor proposal is dropped after an edit undone back to the same settings object', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const initial = editor.state.adjustments;
  const pending = deferred<StyleEditorProposal>();
  const running = applyStyleToActivePhoto(editor.source, () => pending.promise);
  editor.edit({ exposure: 0.7 });
  editor.undo();
  assert.equal(editor.state.adjustments, initial);
  pending.resolve(proposal({ exposure: 1.2 }));
  assert.equal((await running).status, 'stale');
  assert.deepEqual(editor.applied, []);
});

test('revision ignores store updates that leave the photo and its settings unchanged', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const launch = editor.source.current();
  editor.source.apply(editor.state.adjustments);
  assert.equal(editor.source.current().revision, launch.revision);
});

test('batch reload refreshes the active photo when it is unchanged', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const library = view('b.nef', { exposure: 0 });
  const views = [editor, library].map(({ source }) => ({ launch: source.current(), source }));
  const reloaded = await reloadActivePhotos(views, ['a.nef', 'b.nef'], async (path) => ({
    exposure: path === 'a.nef' ? 1 : 2,
  }));
  assert.deepEqual(reloaded, ['a.nef', 'b.nef']);
  assert.deepEqual(editor.state.adjustments, { exposure: 1 });
  assert.deepEqual(library.state.adjustments, { exposure: 2 });
});

test('batch reload skips photos outside the batch', async () => {
  const editor = view('c.nef', { exposure: 0 });
  const loads: string[] = [];
  const reloaded = await reloadActivePhotos(
    [{ launch: editor.source.current(), source: editor.source }],
    ['a.nef'],
    async (path) => {
      loads.push(path);
      return { exposure: 1 };
    },
  );
  assert.deepEqual(reloaded, []);
  assert.deepEqual(loads, []);
});

test('batch reload does not touch a view that switched photo during the batch', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const launch = editor.source.current();
  // The batch runs; meanwhile the user opens b.nef (also in the batch).
  editor.open('b.nef', { exposure: -0.3 });
  const loads: string[] = [];
  const reloaded = await reloadActivePhotos([{ launch, source: editor.source }], ['a.nef', 'b.nef'], async (path) => {
    loads.push(path);
    return { exposure: 1 };
  });
  assert.deepEqual(reloaded, []);
  assert.deepEqual(loads, []);
  assert.deepEqual(editor.state, { path: 'b.nef', adjustments: { exposure: -0.3 } });
});

test('batch reload does not overwrite settings edited while the sidecar was read', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const launch = editor.source.current();
  const pending = deferred<Settings | null>();
  const running = reloadActivePhotos([{ launch, source: editor.source }], ['a.nef'], () => pending.promise);
  editor.edit({ exposure: 0.4 });
  pending.resolve({ exposure: 1 });
  assert.deepEqual(await running, []);
  assert.deepEqual(editor.state.adjustments, { exposure: 0.4 });
});

test('batch reload does not overwrite settings edited during the batch', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const launch = editor.source.current();
  editor.edit({ contrast: 12 });
  const reloaded = await reloadActivePhotos([{ launch, source: editor.source }], ['a.nef'], async () => ({
    exposure: 1,
  }));
  assert.deepEqual(reloaded, []);
  assert.deepEqual(editor.state.adjustments, { exposure: 0, contrast: 12 });
});

test('batch reload skips views reopened on the launch photo with its cached settings object', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const library = view('a.nef', { exposure: 0 });
  const views = [editor, library].map(({ source }) => ({ launch: source.current(), source }));
  const pending = deferred<void>();
  const loads: string[] = [];
  const running = reloadActivePhotos(views, ['a.nef', 'b.nef'], async (path) => {
    await pending.promise;
    loads.push(path);
    return { exposure: 1 };
  });
  // Editor: reopened while its sidecar is read; library: before its sidecar is read.
  const cachedEditor = editor.state.adjustments;
  editor.open('b.nef', { exposure: -0.3 });
  editor.open('a.nef', cachedEditor);
  const cachedLibrary = library.state.adjustments;
  library.open('b.nef', { exposure: -0.3 });
  library.open('a.nef', cachedLibrary);
  pending.resolve();
  assert.deepEqual(await running, []);
  assert.deepEqual(loads, ['a.nef']);
  assert.deepEqual(editor.applied, []);
  assert.deepEqual(library.applied, []);
  assert.equal(editor.state.adjustments, cachedEditor);
  assert.equal(library.state.adjustments, cachedLibrary);
});

test('batch reload skips a view whose edit was undone back to the same settings object', async () => {
  const editor = view('a.nef', { exposure: 0 });
  const launch = editor.source.current();
  editor.edit({ contrast: 12 });
  editor.undo();
  const reloaded = await reloadActivePhotos([{ launch, source: editor.source }], ['a.nef'], async () => ({
    exposure: 1,
  }));
  assert.deepEqual(reloaded, []);
  assert.deepEqual(editor.applied, []);
});
