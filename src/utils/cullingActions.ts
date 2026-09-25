import { invoke } from '@tauri-apps/api/core';
import { useLibraryStore } from '../store/useLibraryStore';
import { CullingPersistenceSummary, CullingSuggestions, Invokes } from '../components/ui/AppProperties';
import { buildCullingUndoAssignments, planCullingApplication } from './cullingApplication';
import {
  filterCullingAssignmentsForCurrentImages,
  persistColorAssignments,
  persistRatingAssignments,
} from './ratingPersistence';

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
export async function applyCullingSuggestions(suggestions: CullingSuggestions): Promise<CullingPersistenceSummary> {
  const plan = planCullingApplication(suggestions, useLibraryStore.getState().imageList);
  const { ratingResult, colorResult } = await persistAutomaticAssignments(plan.ratings, plan.colors);
  return {
    succeededRatings: ratingResult.succeeded,
    succeededColors: colorResult.succeeded,
    failedRatings: ratingResult.failures,
    failedColors: colorResult.failures,
    skippedPaths: plan.protectedPaths,
    undo: plan.undo,
  };
}

/** Restores the rating and color label each photo had before `applyCullingSuggestions`. */
export async function undoCullingApplication(summary: CullingPersistenceSummary): Promise<number> {
  const { ratings, colors } = buildCullingUndoAssignments(
    summary.undo,
    summary.succeededRatings,
    summary.succeededColors,
  );
  const { ratingResult, colorResult } = await persistAutomaticAssignments(ratings, colors);
  const failures = ratingResult.failures.length + colorResult.failures.length;
  if (failures > 0) {
    throw new Error(
      [...ratingResult.failures, ...colorResult.failures].map((failure) => String(failure.error)).join('; '),
    );
  }
  return new Set([...Object.keys(ratings), ...Object.keys(colors)]).size;
}
