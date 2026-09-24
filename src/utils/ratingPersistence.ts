export function isRatingProtectedFromCulling(ratingIsManual: boolean | null | undefined): boolean {
  return ratingIsManual !== false;
}

export interface CullingProtectedPathSets {
  ratingPaths: Set<string>;
  colorLabelPaths: Set<string>;
  allPaths: Set<string>;
}

export function getCullingProtectedPaths(
  images: readonly {
    path: string;
    rating_is_manual: boolean | null | undefined;
    color_label_is_manual?: boolean | null;
  }[],
): CullingProtectedPathSets {
  const ratingPaths = new Set<string>();
  const colorLabelPaths = new Set<string>();

  for (const image of images) {
    if (isRatingProtectedFromCulling(image.rating_is_manual)) ratingPaths.add(image.path);
    if (image.color_label_is_manual !== false) colorLabelPaths.add(image.path);
  }

  return {
    ratingPaths,
    colorLabelPaths,
    allPaths: new Set([...ratingPaths, ...colorLabelPaths]),
  };
}

export function filterCullingAssignmentsForCurrentImages(
  images: readonly {
    path: string;
    rating_is_manual: boolean | null | undefined;
    color_label_is_manual?: boolean | null;
  }[],
  ratings: Record<string, number>,
  colors: Record<string, string | null>,
): { ratings: Record<string, number>; colors: Record<string, string | null> } {
  const imagesByPath = new Map<string, (typeof images)[number]>();
  images.forEach((image) => imagesByPath.set(image.path, image));

  const currentRatings: Record<string, number> = {};
  Object.entries(ratings).forEach(([path, rating]) => {
    const image = imagesByPath.get(path);
    if (image?.rating_is_manual === false) currentRatings[path] = rating;
  });

  const currentColors: Record<string, string | null> = {};
  Object.entries(colors).forEach(([path, color]) => {
    const image = imagesByPath.get(path);
    if (image?.color_label_is_manual === false) currentColors[path] = color;
  });

  return { ratings: currentRatings, colors: currentColors };
}

export function mergeThumbnailRatings(
  currentRatings: Record<string, number>,
  pendingRatings: Record<string, number>,
  images: readonly {
    path: string;
    rating: number;
    rating_is_manual: boolean | null | undefined;
  }[],
): Record<string, number> {
  const imagesByPath = new Map(images.map((image) => [image.path, image]));
  const mergedRatings = { ...currentRatings };

  Object.entries(pendingRatings).forEach(([path, rating]) => {
    const image = imagesByPath.get(path);
    mergedRatings[path] = image?.rating_is_manual === true ? image.rating : rating;
  });

  return mergedRatings;
}

export function getNextManualRating(currentRating: number, selectedRating: number): number {
  return selectedRating === currentRating ? 0 : selectedRating;
}

export function mergeLoadedRating(
  currentRating: number | undefined,
  currentIsManual: boolean | null | undefined,
  loadedRating: number,
  loadedIsManual: boolean | null,
): { rating: number; ratingIsManual: boolean | null } {
  if (currentIsManual === true) {
    return { rating: currentRating ?? 0, ratingIsManual: true };
  }
  return { rating: loadedRating, ratingIsManual: loadedIsManual };
}

export function mergeLoadedColorLabel(
  currentTags: string[] | null | undefined,
  currentIsManual: boolean | null | undefined,
  loadedTags: string[] | null,
  loadedIsManual: boolean | null,
): { tags: string[] | null; colorLabelIsManual: boolean | null } {
  if (currentIsManual !== true) {
    return { tags: loadedTags, colorLabelIsManual: loadedIsManual };
  }

  const currentColorTags = (currentTags || []).filter((tag) => tag.startsWith('color:'));
  const incomingTags = (loadedTags ?? currentTags ?? []).filter((tag) => !tag.startsWith('color:'));
  const tags = [...incomingTags, ...currentColorTags];
  return {
    tags: tags.length > 0 ? tags : null,
    colorLabelIsManual: true,
  };
}

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
