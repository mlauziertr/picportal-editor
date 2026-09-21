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

export async function persistBatchWithReconciliation(
  paths: string[],
  persistBatch: () => Promise<unknown>,
  persistOne: (path: string) => Promise<unknown>,
): Promise<void> {
  try {
    await persistBatch();
  } catch (batchError) {
    const failures = (
      await Promise.all(
        paths.map(async (path) => {
          try {
            await persistOne(path);
            return null;
          } catch (error) {
            return { path, error };
          }
        }),
      )
    ).filter((failure): failure is { path: string; error: unknown } => failure !== null);
    if (failures.length > 0) {
      throw new Error(
        `Batch persistence failed (${String(batchError)}); individual retries failed: ${failures
          .map(({ path, error }) => `${path}: ${String(error)}`)
          .join('; ')}`,
      );
    }
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
      try {
        await persist(paths, rating);
        return { rating, paths };
      } catch (error) {
        return { rating, paths, error };
      }
    }),
  );
  const succeeded: Record<string, number> = {};
  const failures: RatingPersistenceFailure[] = [];
  outcomes.forEach((outcome) => {
    if ('error' in outcome) {
      failures.push({ rating: outcome.rating, paths: outcome.paths, error: outcome.error });
      return;
    }
    outcome.paths.forEach((path) => {
      succeeded[path] = outcome.rating;
    });
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
      try {
        await persist(paths, color);
        return { color, paths };
      } catch (error) {
        return { color, paths, error };
      }
    }),
  );
  const succeeded: Record<string, string | null> = {};
  const failures: ColorPersistenceFailure[] = [];
  outcomes.forEach((outcome) => {
    if ('error' in outcome) {
      failures.push({ color: outcome.color, paths: outcome.paths, error: outcome.error });
      return;
    }
    outcome.paths.forEach((path) => {
      succeeded[path] = outcome.color;
    });
  });
  return { succeeded, failures };
}
