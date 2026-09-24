export interface PathBoundSnapshot<T> {
  path: string | null;
  value: T | null;
}

export function valueForPath<T>(snapshot: PathBoundSnapshot<T>, path: string | null): T | null {
  return path !== null && snapshot.path === path ? snapshot.value : null;
}
