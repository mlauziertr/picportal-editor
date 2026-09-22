import assert from 'node:assert/strict';
import {
  beginCullingInvocation,
  completeCullingCancellation,
  createCullingInvocationId,
  cullingAnalysisMessageKey,
  cullingEventMatches,
  emptyCullingResultsHeadline,
  hasCullingResultItems,
  hideCullingReviewForEditor,
  initialCullingRejectPaths,
  populatedCullingResultsTab,
  requestCullingCancellation,
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
assert.deepEqual(started.progress, { current: 0, total: 0, stage: 'Starting...' });
assert.equal(started.isCancelling, false);
assert.equal(started.cancelled, false);
assert.equal(cullingEventMatches(started, secondId), true);
assert.equal(cullingEventMatches(started, firstId), false);
assert.equal(cullingEventMatches({ invocationId: null }, secondId), false);
assert.equal(cullingEventMatches(dismissedWithLateResults, firstId), false);

const cancellationRequested = requestCullingCancellation(started);
assert.equal(cancellationRequested.isCancelling, true);
assert.equal(cancellationRequested.invocationId, secondId);
assert.equal(requestCullingCancellation(cancellationRequested), cancellationRequested);
const cancellationComplete = completeCullingCancellation(cancellationRequested);
assert.equal(cancellationComplete.cancelled, true);
assert.equal(cancellationComplete.isCancelling, false);
assert.equal(cancellationComplete.invocationId, null);
assert.equal(cancellationComplete.progress, null);
const restarted = beginCullingInvocation(cancellationComplete, createCullingInvocationId());
assert.equal(restarted.cancelled, false);
assert.equal(restarted.isCancelling, false);
assert.notEqual(restarted.invocationId, secondId);

const emptyLists = { similarGroups: [], blurryImages: [], reviewAlerts: [], unknownImages: [], failedPaths: [] };
for (const status of [
  'subject-ready-focus-calibration-unavailable',
  'focus-calibration-unavailable',
]) {
  const headline = emptyCullingResultsHeadline(status);
  assert.notEqual(headline, 'noIssuesFound');
  assert.equal(headline, cullingAnalysisMessageKey(status));
  assert.equal(emptyLists.blurryImages.length, 0);
  assert.equal(emptyLists.reviewAlerts.length, 0);
  assert.equal(emptyLists.unknownImages.length, 0);
}
assert.equal(emptyCullingResultsHeadline('ready'), 'noIssuesFound');
assert.equal(emptyCullingResultsHeadline('disabled'), 'noIssuesFound');
assert.equal(cullingAnalysisMessageKey('focus-ready'), 'subjectAnalysisUnavailable');

const unevaluablePass = {
  similarGroups: [],
  blurryImages: [],
  reviewAlerts: [],
  unknownImages: [],
  failedPaths: [],
};
const similarPass = {
  similarGroups: [{ duplicates: [{ path: '/photos/soft.jpg' }] }],
  blurryImages: [],
  reviewAlerts: [],
  unknownImages: [],
  failedPaths: [],
};
const afterUnevaluable = populatedCullingResultsTab(unevaluablePass);
const afterSimilar = populatedCullingResultsTab(similarPass);
assert.equal(afterSimilar, 'similar');
assert.notEqual(afterSimilar, 'unknown');
assert.equal(
  populatedCullingResultsTab({
    similarGroups: [],
    blurryImages: [{ path: '/photos/blur.jpg' }],
    reviewAlerts: [],
    unknownImages: [],
    failedPaths: [],
  }),
  'blurry',
);

const failedOnlyPass = {
  ...emptyLists,
  failedPaths: ['/photos/unreadable.raw'],
};
assert.equal(hasCullingResultItems(failedOnlyPass), true);
assert.equal(populatedCullingResultsTab(failedOnlyPass), 'failed');
assert.deepEqual([...initialCullingRejectPaths(failedOnlyPass, 'standard')], []);

const mixedCoveragePass = {
  similarGroups: [
    {
      duplicates: [
        { path: '/photos/eyes-not-evaluated-duplicate.raw' },
        { path: '/photos/regular-duplicate.raw' },
      ],
    },
  ],
  blurryImages: [
    { path: '/photos/eyes-not-evaluated-blurry.raw' },
    { path: '/photos/regular-blurry.raw' },
  ],
  reviewAlerts: [],
  unknownImages: [
    { path: '/photos/eyes-not-evaluated-duplicate.raw', eyeState: 'not-evaluated' },
    { path: '/photos/eyes-not-evaluated-blurry.raw', eyeState: 'not-evaluated' },
  ],
  failedPaths: ['/photos/unreadable.raw'],
};
const mixedRejects = initialCullingRejectPaths(mixedCoveragePass, 'extreme');
assert.equal(hasCullingResultItems(mixedCoveragePass), true);
assert.equal(mixedRejects.has('/photos/regular-duplicate.raw'), true);
assert.equal(mixedRejects.has('/photos/regular-blurry.raw'), true);
assert.equal(mixedRejects.has('/photos/eyes-not-evaluated-duplicate.raw'), false);
assert.equal(mixedRejects.has('/photos/eyes-not-evaluated-blurry.raw'), false);
assert.equal(mixedRejects.has('/photos/unreadable.raw'), false);

console.log('culling review session preserves explicit review coverage');
