import assert from 'node:assert/strict';
import test from 'node:test';
import {
  persistBatchWithReconciliation,
  persistColorAssignments,
  persistRatingAssignments,
} from '../src/utils/ratingPersistence.ts';

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
  await persistBatchWithReconciliation(
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
