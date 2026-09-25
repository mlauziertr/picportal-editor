import assert from 'node:assert/strict';
import test from 'node:test';
import { isVirtualCopyPath, splitVirtualCopyPath, stripVirtualCopySuffix } from '../src/utils/virtualCopyPath.ts';

test('only a complete virtual-copy suffix is removed from image paths', () => {
  const physicalPath = '/synthetic?vc=abcdef/photos/image.jpg';
  const virtualPath = `${physicalPath}?vc=123abc`;

  assert.deepEqual(splitVirtualCopyPath(virtualPath), { sourcePath: physicalPath, copyId: '123abc' });
  assert.equal(stripVirtualCopySuffix(virtualPath), physicalPath);
  assert.equal(isVirtualCopyPath(virtualPath), true);
});

test('question marks in folders and ordinary filename queries are preserved', () => {
  const folderWithMarker = '/synthetic?vc=abcdef/photos/image.jpg';
  const folderOnly = '/synthetic-folder?vc=abcdef';
  const queryLikeName = '/synthetic/photos/image.jpg?version=2';

  assert.equal(splitVirtualCopyPath(folderWithMarker), null);
  assert.equal(stripVirtualCopySuffix(folderWithMarker), folderWithMarker);
  assert.equal(isVirtualCopyPath(folderOnly), false);
  assert.equal(stripVirtualCopySuffix(queryLikeName), queryLikeName);
});
