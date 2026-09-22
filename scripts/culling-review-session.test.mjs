import assert from 'node:assert/strict';
import {
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
assert.equal(hidden.suggestions, review.suggestions);
assert.deepEqual(hidden.pathsToCull, review.pathsToCull);
assert.equal(hidden.progress, null);

const restored = restoreCullingReviewOnLibraryReturn(hidden);
assert.equal(restored.isOpen, true);
assert.equal(restored.suggestions, review.suggestions);
assert.deepEqual(restored.pathsToCull, review.pathsToCull);

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

console.log('culling review session preserves the list across editor return');
