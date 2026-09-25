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

// State after applying: the color write for manual-rating.jpg failed.
const writtenRatings = { 'auto.jpg': 4, 'manual-color.jpg': 3 };
const writtenColors: Record<string, string | null> = { 'auto.jpg': 'green' };
const afterApply = images.map((image) =>
  image.path === 'auto.jpg'
    ? { ...image, rating: 4, tags: ['keep', 'color:green'] }
    : image.path === 'manual-color.jpg'
      ? { ...image, rating: 3 }
      : image,
);

test('undo restores previous values only for what was really written', () => {
  const plan = planCullingApplication(proposals, images);
  // manual-rating.jpg's color write failed: it must not be "restored".
  const undo = buildCullingUndoAssignments(plan.undo, writtenRatings, writtenColors, afterApply);
  assert.deepEqual(undo.ratings, { 'auto.jpg': 0, 'manual-color.jpg': 2 });
  assert.deepEqual(undo.colors, { 'auto.jpg': null });
  assert.deepEqual(undo.skippedPaths, []);
});

test('undo leaves photos changed after the application untouched', () => {
  const plan = planCullingApplication(proposals, images);
  // A newer application B rated auto.jpg 2; the user then starred manual-color.jpg by hand.
  const later = afterApply.map((image) =>
    image.path === 'auto.jpg'
      ? { ...image, rating: 2 }
      : image.path === 'manual-color.jpg'
        ? { ...image, rating: 5, rating_is_manual: true }
        : image,
  );
  const undo = buildCullingUndoAssignments(plan.undo, writtenRatings, writtenColors, later);
  assert.deepEqual(undo.ratings, {});
  // auto.jpg still shows the green label A wrote, but one of its values moved on: it is reported.
  assert.deepEqual(undo.colors, { 'auto.jpg': null });
  assert.deepEqual(undo.skippedPaths, ['auto.jpg', 'manual-color.jpg']);
});

test('undo of an older application yields to a newer one with the same values', () => {
  const plan = planCullingApplication(proposals, images);
  // B rewrote the same values on auto.jpg: only the writer record tells them apart.
  const undo = buildCullingUndoAssignments(
    plan.undo,
    writtenRatings,
    writtenColors,
    afterApply,
    (path) => path !== 'auto.jpg',
  );
  assert.deepEqual(undo.ratings, { 'manual-color.jpg': 2 });
  assert.deepEqual(undo.colors, {});
  assert.deepEqual(undo.skippedPaths, ['auto.jpg']);
});

test('color label is read from color: tags', () => {
  assert.equal(colorLabelFromTags(['a', 'color:red']), 'red');
  assert.equal(colorLabelFromTags(null), null);
});
