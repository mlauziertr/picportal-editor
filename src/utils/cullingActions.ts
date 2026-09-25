import { invoke } from '@tauri-apps/api/core';
import { useLibraryStore } from '../store/useLibraryStore';
import { CullingPersistenceSummary, CullingSuggestions, Invokes } from '../components/ui/AppProperties';
import { buildCullingUndoAssignments, planCullingApplication } from './cullingApplication';
import {
  filterCullingAssignmentsForCurrentImages,
  persistColorAssignments,
  persistRatingAssignments,
} from './ratingPersistence';

// Last application (or undo) that wrote each photo, so an older "Undo" never
// overwrites a newer decision. In memory only, like the undo journal.
let nextApplicationId = 1;
const latestWriterByPath = new Map<string, number>();

// Applications and undos run one at a time, each deciding from the values the
// previous one left: an undo checks "still this application's value" right
// before its own write, so a newer application can no longer slip in between.
let cullingWrites: Promise<unknown> = Promise.resolve();

function serializeCullingWrite<T>(write: () => Promise<T>): Promise<T> {
  const run = cullingWrites.then(write, write);
  cullingWrites = run.catch(() => undefined);
  return run;
}

function recordWriter(paths: Iterable<string>, applicationId: number | null) {
  for (const path of paths) {
    if (applicationId === null) latestWriterByPath.delete(path);
    else latestWriterByPath.set(path, applicationId);
  }
}

async function persistAutomaticAssignments(ratings: Record<string, number>, colors: Record<string, string | null>) {
  // Ratings and labels share one sidecar file. Keep the two passes ordered so
  // concurrent read/modify/write calls cannot discard the other decision.
  const ratingResult = await persistRatingAssignments(ratings, (paths, rating) =>
    invoke(Invokes.SetRatingForPaths, { paths, rating, ratingIsManual: false }),
  );
  const colorResult = await persistColorAssignments(colors, (paths, color) =>
    invoke(Invokes.SetColorLabelForPaths, { paths, color, colorLabelIsManual: false }),
  );

  if (Object.keys(ratingResult.succeeded).length > 0 || Object.keys(colorResult.succeeded).length > 0) {
    useLibraryStore.getState().setLibrary((state) => {
      const currentAssignments = filterCullingAssignmentsForCurrentImages(
        state.imageList,
        ratingResult.succeeded,
        colorResult.succeeded,
      );
      if (Object.keys(currentAssignments.ratings).length === 0 && Object.keys(currentAssignments.colors).length === 0) {
        return state;
      }

      return {
        imageRatings: { ...state.imageRatings, ...currentAssignments.ratings },
        imageList: state.imageList.map((image) => {
          const rating = currentAssignments.ratings[image.path];
          const hasRatingUpdate = rating !== undefined;
          const hasColorUpdate = image.path in currentAssignments.colors;
          if (!hasRatingUpdate && !hasColorUpdate) return image;
          const color = currentAssignments.colors[image.path];
          const otherTags = (image.tags || []).filter((tag) => !tag.startsWith('color:'));
          return {
            ...image,
            ...(hasRatingUpdate ? { rating, rating_is_manual: false } : {}),
            ...(hasColorUpdate
              ? {
                  tags: color ? [...otherTags, `color:${color}`] : otherTags.length > 0 ? otherTags : null,
                  color_label_is_manual: false,
                }
              : {}),
          };
        }),
      };
    });
  }
  return { ratingResult, colorResult };
}

/** Writes the culling proposals the user chose to apply; nothing is written before this call. */
export function applyCullingSuggestions(suggestions: CullingSuggestions): Promise<CullingPersistenceSummary> {
  return serializeCullingWrite(async () => {
    const plan = planCullingApplication(suggestions, useLibraryStore.getState().imageList);
    const applicationId = nextApplicationId++;
    const { ratingResult, colorResult } = await persistAutomaticAssignments(plan.ratings, plan.colors);
    // Only a confirmed write takes a photo over; a failed one leaves the previous writer's undo usable.
    recordWriter(
      new Set([...Object.keys(ratingResult.succeeded), ...Object.keys(colorResult.succeeded)]),
      applicationId,
    );
    return {
      applicationId,
      succeededRatings: ratingResult.succeeded,
      succeededColors: colorResult.succeeded,
      failedRatings: ratingResult.failures,
      failedColors: colorResult.failures,
      skippedPaths: plan.protectedPaths,
      undo: plan.undo,
    };
  });
}

/**
 * Restores the rating and color label each photo had before `applyCullingSuggestions`,
 * only where this application's value is still in place (see `buildCullingUndoAssignments`).
 */
export function undoCullingApplication(
  summary: CullingPersistenceSummary,
): Promise<{ restored: number; skipped: number }> {
  return serializeCullingWrite(async () => {
    const { ratings, colors, skippedPaths } = buildCullingUndoAssignments(
      summary.undo,
      summary.succeededRatings,
      summary.succeededColors,
      useLibraryStore.getState().imageList,
      (path) => latestWriterByPath.get(path) === summary.applicationId,
    );
    const restoredPaths = new Set([...Object.keys(ratings), ...Object.keys(colors)]);
    const { ratingResult, colorResult } = await persistAutomaticAssignments(ratings, colors);
    // A photo stays this application's until every one of its restores is confirmed, so a failed undo can be retried.
    recordWriter(
      [...restoredPaths].filter(
        (path) =>
          (!(path in ratings) || path in ratingResult.succeeded) &&
          (!(path in colors) || path in colorResult.succeeded),
      ),
      null,
    );
    const failures = ratingResult.failures.length + colorResult.failures.length;
    if (failures > 0) {
      throw new Error(
        [...ratingResult.failures, ...colorResult.failures].map((failure) => String(failure.error)).join('; '),
      );
    }
    return { restored: restoredPaths.size, skipped: skippedPaths.length };
  });
}
