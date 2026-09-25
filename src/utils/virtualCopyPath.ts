export interface VirtualCopyPath {
  sourcePath: string;
  copyId: string;
}

const VIRTUAL_COPY_ID_PATTERN = /^[0-9a-f]{6}$/;

export function splitVirtualCopyPath(path: string): VirtualCopyPath | null {
  const markerIndex = path.lastIndexOf('?vc=');
  if (markerIndex === -1) return null;

  const sourcePath = path.slice(0, markerIndex);
  const copyId = path.slice(markerIndex + '?vc='.length);
  if (!VIRTUAL_COPY_ID_PATTERN.test(copyId)) return null;

  const fileName = sourcePath.split(/[\\/]/).pop() || '';
  const extensionIndex = fileName.lastIndexOf('.');
  if (extensionIndex <= 0 || extensionIndex === fileName.length - 1) return null;

  return { sourcePath, copyId };
}

export function stripVirtualCopySuffix(path: string): string {
  return splitVirtualCopyPath(path)?.sourcePath ?? path;
}

export function isVirtualCopyPath(path: string): boolean {
  return splitVirtualCopyPath(path) !== null;
}
