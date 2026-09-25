import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildCullingUndoAssignments,
  colorLabelFromTags,
  planCullingApplication,
} from '../src/utils/cullingApplication.ts';

const proposals = {
  results: [
    { path: 'auto.jpg' },
    { path: 'manual-rating.jpg' },
    { path: 'manual-color.jpg' },
    { path: 'lightroom-rating.jpg' },
    { path: 'unchanged.jpg' },
    { path: 'gone.jpg' },
  ],
  starAssignments: {
    'auto.jpg': 4,
    'manual-rating.jpg': 1,
    'manual-color.jpg': 3,
    'lightroom-rating.jpg': 2,
    'unchanged.jpg': 3,
    'gone.jpg': 5,
  },
  colorAssignments: {
    'auto.jpg': 'green',
    'manual-rating.jpg': 'red',
    'manual-color.jpg': 'red',
    'lightroom-rating.jpg': null,
    'unchanged.jpg': null,
    'gone.jpg': 'green',
  },
};

const images = [
  { path: 'auto.jpg', rating: 0, rating_is_manual: false, color_label_is_manual: false, tags: ['keep'] },
  { path: 'manual-rating.jpg', rating: 5, rating_is_manual: true, color_label_is_manual: false, tags: null },
  {
    path: 'manual-color.jpg',
    rating: 2,
    rating_is_manual: false,
    color_label_is_manual: true,
    tags: ['color:blue'],
  },
  // Rating imported from a Lightroom XMP: provenance unknown, so it is protected.
  { path: 'lightroom-rating.jpg', rating: 3, rating_is_manual: null, color_label_is_manual: false, tags: null },
  { path: 'unchanged.jpg', rating: 3, rating_is_manual: false, color_label_is_manual: false, tags: null },
];

test('culling plan writes only automatic values that actually change', () => {
  const plan = planCullingApplication(proposals, images);
  assert.deepEqual(plan.ratings, { 'auto.jpg': 4, 'manual-color.jpg': 3 });
  assert.deepEqual(plan.colors, { 'auto.jpg': 'green', 'manual-rating.jpg': 'red' });
  assert.deepEqual(plan.protectedPaths, ['manual-rating.jpg', 'manual-color.jpg', 'lightroom-rating.jpg']);
  // unchanged.jpg already matches; gone.jpg is no longer in the library.
  assert.equal(plan.photoCount, 3);
  assert.deepEqual(plan.undo, [
    { path: 'auto.jpg', rating: 0, colorLabel: null, ratingIsManual: false, colorLabelIsManual: false },
    { path: 'manual-rating.jpg', rating: 5, colorLabel: null, ratingIsManual: true, colorLabelIsManual: false },
    { path: 'manual-color.jpg', rating: 2, colorLabel: 'blue', ratingIsManual: false, colorLabelIsManual: true },
  ]);
});

test('undo restores previous values only for what was really written', () => {
  const plan = planCullingApplication(proposals, images);
  // The color write for manual-rating.jpg failed: it must not be "restored".
  const undo = buildCullingUndoAssignments(
    plan.undo,
    { 'auto.jpg': 4, 'manual-color.jpg': 3 },
    { 'auto.jpg': 'green' },
  );
  assert.deepEqual(undo.ratings, { 'auto.jpg': 0, 'manual-color.jpg': 2 });
  assert.deepEqual(undo.colors, { 'auto.jpg': null });
});

test('color label is read from color: tags', () => {
  assert.equal(colorLabelFromTags(['a', 'color:red']), 'red');
  assert.equal(colorLabelFromTags(null), null);
});
