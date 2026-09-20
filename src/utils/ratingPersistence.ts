export interface RatingPersistenceFailure {
  rating: number;
  paths: string[];
  error: unknown;
}

export interface RatingPersistenceResult {
  succeeded: Record<string, number>;
  failures: RatingPersistenceFailure[];
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
