import assert from 'node:assert/strict';
import test from 'node:test';
import {
  isRatingProtectedFromCulling,
  mergeLoadedRating,
  persistBatchWithReconciliation,
  persistColorAssignments,
  persistRatingAssignments,
} from '../src/utils/ratingPersistence.ts';

test('culling preserves manual, unknown, and explicit zero provenance', () => {
  assert.equal(isRatingProtectedFromCulling(true), true);
  assert.equal(isRatingProtectedFromCulling(null), true);
  assert.equal(isRatingProtectedFromCulling(undefined), true);
  assert.equal(isRatingProtectedFromCulling(false), false);
});

test('stale metadata refresh cannot replace a manually cleared rating', () => {
  assert.deepEqual(mergeLoadedRating(0, true, 4, false), { rating: 0, ratingIsManual: true });
  assert.deepEqual(mergeLoadedRating(undefined, null, 4, null), { rating: 4, ratingIsManual: null });
});

test('partial rating failures retain only durable successful groups', async () => {
  const calls: Array<{ paths: string[]; rating: number }> = [];
  const result = await persistRatingAssignments(
    { 'retained.jpg': 5, 'review.jpg': 3, 'second-retained.jpg': 5 },
    async (paths, rating) => {
      calls.push({ paths, rating });
      if (rating === 3) throw new Error('write failed');
    },
  );

  assert.deepEqual(calls, [
    { paths: ['retained.jpg', 'second-retained.jpg'], rating: 5 },
    { paths: ['review.jpg'], rating: 3 },
  ]);
  assert.deepEqual(result.succeeded, { 'retained.jpg': 5, 'second-retained.jpg': 5 });
  assert.equal(result.failures.length, 1);
  assert.deepEqual(result.failures[0]?.paths, ['review.jpg']);
  assert.equal(result.failures[0]?.rating, 3);
});

test('a partial batch write is reconciled with idempotent individual retries', async () => {
  const calls: string[][] = [];
  let batchAttempt = true;
  const result = await persistBatchWithReconciliation(
    ['first.jpg', 'second.jpg'],
    async () => {
      calls.push(['batch']);
      if (batchAttempt) {
        batchAttempt = false;
        throw new Error('partial batch failure');
      }
    },
    async (path) => {
      calls.push([path]);
    },
  );

  assert.deepEqual(calls, [['batch'], ['first.jpg'], ['second.jpg']]);
  assert.deepEqual(result, { succeeded: ['first.jpg', 'second.jpg'], failures: [] });
});

test('mixed individual retries keep successful sidecar writes', async () => {
  const result = await persistBatchWithReconciliation(
    ['saved.jpg', 'failed.jpg'],
    async () => {
      throw new Error('partial batch failure');
    },
    async (path) => {
      if (path === 'failed.jpg') throw new Error('sidecar write failed');
    },
  );

  assert.deepEqual(result.succeeded, ['saved.jpg']);
  assert.equal(result.failures.length, 1);
  assert.equal(result.failures[0]?.path, 'failed.jpg');
});

test('mixed rating retries keep successful writes in the same group', async () => {
  const result = await persistRatingAssignments(
    { 'saved.jpg': 4, 'failed.jpg': 4, 'other-group.jpg': 2 },
    async (paths) => {
      if (paths.length > 1) throw new Error('batch failed');
      if (paths[0] === 'failed.jpg') throw new Error('sidecar write failed');
    },
  );

  assert.deepEqual(result.succeeded, { 'saved.jpg': 4, 'other-group.jpg': 2 });
  assert.equal(result.failures.length, 1);
  assert.deepEqual(result.failures[0]?.paths, ['failed.jpg']);
  assert.equal(result.failures[0]?.rating, 4);
});

test('color assignments group writes and retain partial failures', async () => {
  const calls: Array<{ paths: string[]; color: string | null }> = [];
  const result = await persistColorAssignments(
    { 'selected.jpg': 'green', 'highlight.jpg': 'blue', 'clear.jpg': null, 'second-selected.jpg': 'green' },
    async (paths, color) => {
      calls.push({ paths, color });
      if (color === 'blue') throw new Error('label write failed');
    },
  );

  assert.equal(calls.length, 3);
  assert.deepEqual(result.succeeded, {
    'selected.jpg': 'green',
    'second-selected.jpg': 'green',
    'clear.jpg': null,
  });
  assert.equal(result.failures.length, 1);
  assert.deepEqual(result.failures[0]?.paths, ['highlight.jpg']);
  assert.equal(result.failures[0]?.color, 'blue');
});

test('mixed color retries keep successful writes in the same group', async () => {
  const result = await persistColorAssignments(
    { 'saved.jpg': 'green', 'failed.jpg': 'green', 'other.jpg': 'red' },
    async (paths) => {
      if (paths.length > 1) throw new Error('batch failed');
      if (paths[0] === 'failed.jpg') throw new Error('sidecar write failed');
    },
  );

  assert.deepEqual(result.succeeded, { 'saved.jpg': 'green', 'other.jpg': 'red' });
  assert.equal(result.failures.length, 1);
  assert.deepEqual(result.failures[0]?.paths, ['failed.jpg']);
  assert.equal(result.failures[0]?.color, 'green');
});
