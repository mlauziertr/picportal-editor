export interface RatingPersistenceFailure {
  rating: number;
  paths: string[];
  error: unknown;
}

export interface RatingPersistenceResult {
  succeeded: Record<string, number>;
  failures: RatingPersistenceFailure[];
}

export interface ColorPersistenceFailure {
  color: string | null;
  paths: string[];
  error: unknown;
}

export interface ColorPersistenceResult {
  succeeded: Record<string, string | null>;
  failures: ColorPersistenceFailure[];
}

export interface PathPersistenceFailure {
  path: string;
  error: unknown;
}

export interface BatchReconciliationResult {
  succeeded: string[];
  failures: PathPersistenceFailure[];
}

function groupedFailureError(failures: PathPersistenceFailure[]): unknown {
  if (failures.length === 1) return failures[0].error;
  return new Error(failures.map(({ path, error }) => `${path}: ${String(error)}`).join('; '));
}

export async function persistBatchWithReconciliation(
  paths: string[],
  persistBatch: () => Promise<unknown>,
  persistOne: (path: string) => Promise<unknown>,
): Promise<BatchReconciliationResult> {
  try {
    await persistBatch();
    return { succeeded: [...paths], failures: [] };
  } catch (batchError) {
    if (paths.length <= 1) {
      return {
        succeeded: [],
        failures: paths.map((path) => ({ path, error: batchError })),
      };
    }
    const results = await Promise.all(
      paths.map(async (path) => {
        try {
          await persistOne(path);
          return { path, ok: true as const };
        } catch (error) {
          return { path, ok: false as const, error };
        }
      }),
    );
    const succeeded: string[] = [];
    const failures: PathPersistenceFailure[] = [];
    for (const result of results) {
      if (result.ok) succeeded.push(result.path);
      else failures.push({ path: result.path, error: result.error });
    }
    return { succeeded, failures };
  }
}

export async function persistRatingAssignments(
  ratings: Record<string, number>,
  persist: (paths: string[], rating: number) => Promise<unknown>,
): Promise<RatingPersistenceResult> {
  const pathsByRating = new Map<number, string[]>();
  Object.entries(ratings).forEach(([path, rating]) => {
    const paths = pathsByRating.get(rating) ?? [];
    paths.push(path);
    pathsByRating.set(rating, paths);
  });

  const outcomes = await Promise.all(
    Array.from(pathsByRating, async ([rating, paths]) => {
      const reconciliation = await persistBatchWithReconciliation(
        paths,
        () => persist(paths, rating),
        (path) => persist([path], rating),
      );
      return { rating, ...reconciliation };
    }),
  );
  const succeeded: Record<string, number> = {};
  const failures: RatingPersistenceFailure[] = [];
  outcomes.forEach(({ rating, succeeded: succeededPaths, failures: pathFailures }) => {
    succeededPaths.forEach((path) => {
      succeeded[path] = rating;
    });
    if (pathFailures.length > 0) {
      failures.push({
        rating,
        paths: pathFailures.map((failure) => failure.path),
        error: groupedFailureError(pathFailures),
      });
    }
  });
  return { succeeded, failures };
}

export async function persistColorAssignments(
  colors: Record<string, string | null>,
  persist: (paths: string[], color: string | null) => Promise<unknown>,
): Promise<ColorPersistenceResult> {
  const pathsByColor = new Map<string, { color: string | null; paths: string[] }>();
  Object.entries(colors).forEach(([path, color]) => {
    const key = color ?? '__none__';
    const entry = pathsByColor.get(key) ?? { color, paths: [] };
    entry.paths.push(path);
    pathsByColor.set(key, entry);
  });

  const outcomes = await Promise.all(
    Array.from(pathsByColor.values(), async ({ color, paths }) => {
      const reconciliation = await persistBatchWithReconciliation(
        paths,
        () => persist(paths, color),
        (path) => persist([path], color),
      );
      return { color, ...reconciliation };
    }),
  );
  const succeeded: Record<string, string | null> = {};
  const failures: ColorPersistenceFailure[] = [];
  outcomes.forEach(({ color, succeeded: succeededPaths, failures: pathFailures }) => {
    succeededPaths.forEach((path) => {
      succeeded[path] = color;
    });
    if (pathFailures.length > 0) {
      failures.push({
        color,
        paths: pathFailures.map((failure) => failure.path),
        error: groupedFailureError(pathFailures),
      });
    }
  });
  return { succeeded, failures };
}
