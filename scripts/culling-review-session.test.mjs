import assert from 'node:assert/strict';
import {
  beginCullingInvocation,
  createCullingInvocationId,
  cullingEventMatches,
  hideCullingReviewForEditor,
  restoreCullingReviewOnLibraryReturn,
} from '../src/utils/cullingReviewSession.ts';

const review = {
  isOpen: true,
  suggestions: { similarGroups: [], reviewAlerts: [{ path: '/photos/dance.jpg' }] },
  progress: null,
  error: null,
  pathsToCull: ['/photos/dance.jpg', '/photos/other.jpg'],
};

const hidden = hideCullingReviewForEditor(review);
assert.equal(hidden.isOpen, false);
assert.equal(hidden.hiddenForEditor, true);
assert.equal(hidden.suggestions, review.suggestions);
assert.deepEqual(hidden.pathsToCull, review.pathsToCull);
assert.equal(hidden.progress, null);

const restored = restoreCullingReviewOnLibraryReturn(hidden);
assert.equal(restored.isOpen, true);
assert.equal(restored.hiddenForEditor, false);
assert.equal(restored.suggestions, review.suggestions);
assert.deepEqual(restored.pathsToCull, review.pathsToCull);
assert.equal(restoreCullingReviewOnLibraryReturn({ ...restored, isOpen: false }).isOpen, false);

const dismissed = {
  isOpen: false,
  suggestions: null,
  progress: null,
  error: null,
  pathsToCull: [],
};
assert.equal(restoreCullingReviewOnLibraryReturn(dismissed).isOpen, false);

const stillOpen = restoreCullingReviewOnLibraryReturn(review);
assert.equal(stillOpen.isOpen, true);
assert.equal(stillOpen.suggestions, review.suggestions);

const dismissedWithLateResults = {
  isOpen: false,
  suggestions: review.suggestions,
  progress: null,
  error: null,
  pathsToCull: review.pathsToCull,
  hiddenForEditor: false,
  invocationId: null,
};
assert.equal(restoreCullingReviewOnLibraryReturn(dismissedWithLateResults).isOpen, false);

const firstId = createCullingInvocationId();
const secondId = createCullingInvocationId();
assert.notEqual(firstId, secondId);
assert.notEqual(firstId, 'cull-1');
assert.equal(cullingEventMatches({ invocationId: secondId }, 'cull-1'), false);
const started = beginCullingInvocation({ ...hidden, hiddenForEditor: true }, secondId);
assert.equal(started.invocationId, secondId);
assert.equal(started.hiddenForEditor, false);
assert.equal(started.suggestions, null);
assert.equal(cullingEventMatches(started, secondId), true);
assert.equal(cullingEventMatches(started, firstId), false);
assert.equal(cullingEventMatches({ invocationId: null }, secondId), false);
assert.equal(cullingEventMatches(dismissedWithLateResults, firstId), false);

console.log('culling review session preserves the list across editor return');
