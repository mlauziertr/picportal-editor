import assert from 'node:assert/strict';
import test from 'node:test';
import { persistRatingAssignments } from '../src/utils/ratingPersistence.ts';

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
