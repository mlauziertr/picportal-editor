import { isRatingProtectedFromCulling } from './ratingPersistence';

// Culling proposes first and writes only when the user applies (spec MAX-19 §1.6).
// These helpers are pure so the plan and the undo journal can be tested without Tauri.

export interface CullingImageState {
  path: string;
  rating?: number;
  rating_is_manual: boolean | null | undefined;
  color_label_is_manual?: boolean | null;
  tags?: string[] | null;
}

export interface CullingProposals {
  results: ReadonlyArray<{ path: string }>;
  starAssignments: Record<string, number>;
  colorAssignments: Record<string, string | null>;
}

/** Values before application; restored as-is by "Undo". */
export interface CullingUndoEntry {
  path: string;
  rating: number;
  colorLabel: string | null;
  ratingIsManual: boolean | null;
  colorLabelIsManual: boolean | null;
}

export interface CullingApplicationPlan {
  ratings: Record<string, number>;
  colors: Record<string, string | null>;
  /** Analyzed photos holding a decision of the user that culling keeps untouched. */
  protectedPaths: string[];
  /** Photos whose rating or color label will actually change. */
  photoCount: number;
  undo: CullingUndoEntry[];
}

export function colorLabelFromTags(tags: readonly string[] | null | undefined): string | null {
  const tag = (tags || []).find((candidate) => candidate.startsWith('color:'));
  return tag ? tag.slice('color:'.length) : null;
}

export function planCullingApplication(
  proposals: CullingProposals,
  images: readonly CullingImageState[],
): CullingApplicationPlan {
  const imagesByPath = new Map(images.map((image) => [image.path, image]));
  const ratings: Record<string, number> = {};
  const colors: Record<string, string | null> = {};
  const protectedPaths: string[] = [];
  const undo: CullingUndoEntry[] = [];

  for (const { path } of proposals.results) {
    const image = imagesByPath.get(path);
    // The photo left the current folder (or was never loaded): never write blindly.
    if (!image) continue;
    const ratingProtected = isRatingProtectedFromCulling(image.rating_is_manual);
    const colorProtected = image.color_label_is_manual !== false;
    if (ratingProtected || colorProtected) protectedPaths.push(path);

    const currentRating = image.rating ?? 0;
    const currentColor = colorLabelFromTags(image.tags);
    let changed = false;
    const proposedRating = proposals.starAssignments[path];
    if (!ratingProtected && proposedRating !== undefined && proposedRating !== currentRating) {
      ratings[path] = proposedRating;
      changed = true;
    }
    if (!colorProtected && path in proposals.colorAssignments) {
      const proposedColor = proposals.colorAssignments[path] ?? null;
      if (proposedColor !== currentColor) {
        colors[path] = proposedColor;
        changed = true;
      }
    }
    if (changed) {
      undo.push({
        path,
        rating: currentRating,
        colorLabel: currentColor,
        ratingIsManual: image.rating_is_manual ?? null,
        colorLabelIsManual: image.color_label_is_manual ?? null,
      });
    }
  }

  return { ratings, colors, protectedPaths, photoCount: undo.length, undo };
}

/**
 * Assignments that restore the pre-application values, limited to what was
 * really written. Only automatic values were overwritten (manual ones are
 * protected), so restoring them as automatic reproduces the previous flags.
 */
export function buildCullingUndoAssignments(
  undo: readonly CullingUndoEntry[],
  writtenRatings: Record<string, number>,
  writtenColors: Record<string, string | null>,
): { ratings: Record<string, number>; colors: Record<string, string | null> } {
  const ratings: Record<string, number> = {};
  const colors: Record<string, string | null> = {};
  for (const entry of undo) {
    if (entry.path in writtenRatings) ratings[entry.path] = entry.rating;
    if (entry.path in writtenColors) colors[entry.path] = entry.colorLabel;
  }
  return { ratings, colors };
}
