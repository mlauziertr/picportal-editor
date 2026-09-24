import { stripVirtualCopySuffix } from './virtualCopyPath';

export function isDirectChildPath(path: string, folderPath: string): boolean {
  const normalizedPath = stripVirtualCopySuffix(path).replace(/\\/g, '/');
  const normalizedFolder = folderPath.replace(/\\/g, '/').replace(/\/+$/, '');
  const folderPrefix = normalizedFolder ? `${normalizedFolder}/` : '/';
  if (!normalizedPath.startsWith(folderPrefix)) return false;
  const relativePath = normalizedPath.slice(folderPrefix.length);
  return relativePath.length > 0 && !relativePath.includes('/');
}

export function getDirectFolderImagePaths(paths: string[], folderPath: string | null): string[] {
  if (!folderPath) return [];
  return paths.filter((path) => isDirectChildPath(path, folderPath));
}
