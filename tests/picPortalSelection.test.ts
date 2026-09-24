import assert from 'node:assert/strict';
import test from 'node:test';
import { picPortalDestinationChanged, retainExplicitGallerySelection } from '../src/utils/picPortalSelection.ts';

test('gallery refresh never implicitly chooses the first available gallery', () => {
  const available = ['gallery-a', 'gallery-b'];

  assert.equal(retainExplicitGallerySelection('', available), '');
  assert.equal(retainExplicitGallerySelection('gallery-b', available), 'gallery-b');
  assert.equal(retainExplicitGallerySelection('removed-gallery', available), '');
});

test('changing account or gallery requires fresh face-analysis consent', () => {
  assert.equal(picPortalDestinationChanged('account-a', 'gallery-a', 'account-a', 'gallery-b'), true);
  assert.equal(picPortalDestinationChanged('account-a', 'gallery-a', 'account-b', 'gallery-a'), true);
  assert.equal(picPortalDestinationChanged('account-a', 'gallery-a', 'account-a', 'gallery-a'), false);
});
