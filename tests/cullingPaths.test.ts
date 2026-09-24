import assert from 'node:assert/strict';
import test from 'node:test';
import { getDirectFolderImagePaths, isDirectChildPath } from '../src/utils/cullingPaths.ts';

test('culling only includes direct supported-folder children', () => {
  const paths = [
    '/synthetic-fixtures/one.jpg',
    '/synthetic-fixtures/two.raw',
    '/synthetic-fixtures/subfolder/hidden.jpg',
    '/other-folder/other.jpg',
    '/synthetic-fixtures/one.jpg?vc=abcdef',
    '/synthetic-fixtures?/question.jpg',
    '/synthetic?vc=abcdef/photo.jpg',
    '/synthetic-fixtures/one.jpg?version=2',
  ];

  assert.deepEqual(getDirectFolderImagePaths(paths, '/synthetic-fixtures'), [
    '/synthetic-fixtures/one.jpg',
    '/synthetic-fixtures/two.raw',
    '/synthetic-fixtures/one.jpg?vc=abcdef',
    '/synthetic-fixtures/one.jpg?version=2',
  ]);
  assert.deepEqual(getDirectFolderImagePaths(paths, '/synthetic-fixtures?'), ['/synthetic-fixtures?/question.jpg']);
  assert.deepEqual(getDirectFolderImagePaths(paths, '/synthetic?vc=abcdef'), ['/synthetic?vc=abcdef/photo.jpg']);
  assert.equal(isDirectChildPath('/synthetic-fixtures/subfolder/hidden.jpg', '/synthetic-fixtures'), false);
  assert.deepEqual(getDirectFolderImagePaths(paths, null), []);
});
