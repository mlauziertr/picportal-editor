import assert from 'node:assert/strict';
import test from 'node:test';
import type { CullingSuggestions } from '../src/components/ui/AppProperties.tsx';
import { cullingResultsFor, restoreCullingSession } from '../src/utils/cullingSession.ts';

const suggestions: CullingSuggestions = {
  similarGroups: [],
  selectedImages: [],
  highlightImages: [],
  duplicateImages: [],
  blurryImages: [],
  closedEyeImages: [],
  unratedImages: [],
  results: [{ path: '/shoot/a.nef' } as CullingSuggestions['results'][number]],
  failedPaths: [],
  starAssignments: {},
  colorAssignments: {},
  eyeAnalysisStatus: 'disabled',
  subjectAnalysisStatus: 'disabled',
  folderPath: '/shoot',
};

// UI store right after a webview reload: everything reset.
const resetUi = {
  cullingModalState: {
    isOpen: false,
    suggestions: null,
    progress: null,
    error: null,
    pathsToCull: [],
    folderPath: null,
  },
  cullingResultsState: { isOpen: false, folderPath: null, suggestions: null, persistence: null, selectedPath: null },
};

test('a running analysis reopens its progress view with the backend folder', () => {
  const progress = { current: 40, total: 120, stage: 'Analyzing', stageCode: 'analyzing' as const };
  const restored = restoreCullingSession({ running: true, folderPath: '/shoot', progress, result: null }, resetUi);
  assert.equal(restored?.cullingModalState?.isOpen, true);
  assert.equal(restored?.cullingModalState?.folderPath, '/shoot');
  assert.deepEqual(restored?.cullingModalState?.progress, progress);
});

test('a finished, unreviewed analysis reopens its results for the right folder', () => {
  const restored = restoreCullingSession(
    { running: false, folderPath: '/shoot', progress: null, result: suggestions },
    resetUi,
  );
  assert.equal(restored?.cullingResultsState?.isOpen, true);
  assert.equal(restored?.cullingResultsState?.folderPath, '/shoot');
  assert.equal(restored?.cullingResultsState?.selectedPath, '/shoot/a.nef');
});

test('nothing is restored when idle or when the UI already shows the session', () => {
  assert.equal(
    restoreCullingSession({ running: false, folderPath: null, progress: null, result: null }, resetUi),
    null,
  );
  const showing = {
    ...resetUi,
    cullingModalState: { ...resetUi.cullingModalState, progress: { current: 1, total: 2, stage: '' } },
  };
  assert.equal(
    restoreCullingSession({ running: true, folderPath: '/shoot', progress: null, result: null }, showing),
    null,
  );
});

test('culling-complete takes the folder from the payload, not the reset UI store', () => {
  assert.equal(cullingResultsFor(suggestions, null).folderPath, '/shoot');
  assert.equal(cullingResultsFor({ ...suggestions, folderPath: undefined }, '/fallback').folderPath, '/fallback');
});
