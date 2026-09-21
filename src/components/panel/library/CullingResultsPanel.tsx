import { useEffect, useMemo, useState } from 'react';
import { Check, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  CullingCategory,
  CullingPersistenceSummary,
  CullingSuggestions,
  ImageAnalysisResult,
} from '../../ui/AppProperties';
import { useLibraryStore } from '../../../store/useLibraryStore';
import Button from '../../ui/Button';
import Text from '../../ui/Text';
import { TextColors, TextVariants } from '../../../types/typography';

const categories: Array<{ key: CullingCategory; dotClass: string; label: string }> = [
  { key: 'selected', dotClass: 'bg-green-500', label: 'Selected' },
  { key: 'highlights', dotClass: 'bg-blue-500', label: 'Highlights' },
  { key: 'duplicate', dotClass: 'bg-yellow-500', label: 'Duplicates' },
  { key: 'blurred', dotClass: 'bg-red-500', label: 'Blurred' },
  { key: 'closedEyes', dotClass: 'bg-purple-500', label: 'Closed eyes' },
  { key: 'unrated', dotClass: 'bg-gray-500', label: 'Unrated' },
];

interface CullingResultsPanelProps {
  suggestions: CullingSuggestions;
  persistence: CullingPersistenceSummary | null;
  folderPath: string | null;
  initialSelectedPath: string | null;
  onClose(): void;
}

function categoryResults(suggestions: CullingSuggestions, category: CullingCategory): ImageAnalysisResult[] {
  return suggestions.results.filter((result) => result.category === category);
}

export default function CullingResultsPanel({
  suggestions,
  persistence,
  folderPath,
  initialSelectedPath,
  onClose,
}: CullingResultsPanelProps) {
  const { t } = useTranslation();
  const [activeCategory, setActiveCategory] = useState<CullingCategory>('selected');
  const [selectedPath, setSelectedPath] = useState<string | null>(initialSelectedPath);
  const setLibrary = useLibraryStore((state) => state.setLibrary);
  const selectedResult = useMemo(
    () => suggestions.results.find((result) => result.path === selectedPath) || null,
    [selectedPath, suggestions.results],
  );
  const activeResults = categoryResults(suggestions, activeCategory);
  const failedCount = (persistence?.failedRatings.length || 0) + (persistence?.failedColors.length || 0);

  useEffect(() => {
    if (initialSelectedPath) {
      setLibrary({ libraryActivePath: initialSelectedPath, multiSelectedPaths: [initialSelectedPath] });
    }
  }, [initialSelectedPath, setLibrary]);

  const handleResultClick = (result: ImageAnalysisResult) => {
    setSelectedPath(result.path);
    setLibrary({ libraryActivePath: result.path, multiSelectedPaths: [result.path] });
  };

  return (
    <div
      className="absolute inset-0 z-30 flex min-h-0 flex-col bg-surface/95 p-4 backdrop-blur-sm"
      data-testid="culling-results"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <Text variant={TextVariants.title}>
            {t('modals.culling.resultsTitle', { defaultValue: 'Culling results' })}
          </Text>
          <Text color={TextColors.secondary} className="mt-1 break-all">
            {folderPath || t('modals.culling.noFolder', { defaultValue: 'Current folder' })}
          </Text>
        </div>
        <Button variant="ghost" onClick={onClose} aria-label={t('modals.culling.close')}>
          <X size={18} />
        </Button>
      </div>

      <div
        className="mt-3 flex flex-wrap gap-2"
        role="group"
        aria-label={t('modals.culling.resultCategories', { defaultValue: 'Culling result categories' })}
      >
        {categories.map((category) => {
          const count = categoryResults(suggestions, category.key).length;
          return (
            <button
              key={category.key}
              type="button"
              aria-pressed={activeCategory === category.key}
              className={`rounded-full border px-3 py-1.5 text-sm ${
                activeCategory === category.key
                  ? 'border-accent bg-accent/15 text-text-primary'
                  : 'border-border-color/50 text-text-secondary hover:text-text-primary'
              }`}
              onClick={() => setActiveCategory(category.key)}
            >
              <span className={`mr-1 inline-block h-2 w-2 rounded-full ${category.dotClass}`} />
              {t(`modals.culling.category.${category.key}`, { defaultValue: category.label })} {count}
            </button>
          );
        })}
      </div>

      <div className="mt-4 grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden lg:grid-cols-[minmax(0,1fr)_minmax(260px,0.8fr)]">
        <div className="min-h-0 overflow-y-auto rounded-lg border border-border-color/40 p-2">
          {activeResults.length === 0 ? (
            <div className="flex h-full items-center justify-center p-6 text-center">
              <Text color={TextColors.secondary}>
                {t('modals.culling.noResults', { defaultValue: 'No photos have this result.' })}
              </Text>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
              {activeResults.map((result) => (
                <button
                  type="button"
                  key={result.path}
                  className={`rounded-lg border p-2 text-left transition-colors ${
                    selectedPath === result.path
                      ? 'border-accent bg-accent/10'
                      : 'border-border-color/30 hover:border-border-color'
                  }`}
                  onClick={() => handleResultClick(result)}
                  title={result.path}
                >
                  <div className="flex aspect-square items-center justify-center rounded bg-bg-primary text-xs text-text-secondary">
                    {result.faceCount > 0 ? `${result.faceCount} face${result.faceCount === 1 ? '' : 's'}` : 'Photo'}
                  </div>
                  <Text variant={TextVariants.small} className="mt-2 truncate">
                    {result.path.split(/[\\/]/).pop()}
                  </Text>
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.ratingValue', { rating: result.suggestedRating, defaultValue: '{{rating}}★' })}
                  </Text>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="min-h-0 overflow-y-auto rounded-lg border border-border-color/40 p-4">
          {selectedResult ? (
            <>
              <Text variant={TextVariants.heading} className="break-all">
                {selectedResult.path.split(/[\\/]/).pop()}
              </Text>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <div className="rounded bg-bg-primary p-2">
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.detailRating', { defaultValue: 'Suggested rating' })}
                  </Text>
                  <div>
                    {t('modals.culling.detailRatingValue', {
                      rating: selectedResult.suggestedRating,
                      defaultValue: '{{rating}} / 5',
                    })}
                  </div>
                </div>
                <div className="rounded bg-bg-primary p-2">
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.detailQuality', { defaultValue: 'Quality' })}
                  </Text>
                  <div>{Math.round(selectedResult.qualityScore * 100)}%</div>
                </div>
                <div className="rounded bg-bg-primary p-2">
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.detailSharpness', { defaultValue: 'Sharpness' })}
                  </Text>
                  <div>{selectedResult.sharpnessMetric.toFixed(1)}</div>
                </div>
                <div className="rounded bg-bg-primary p-2">
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.detailFaces', { defaultValue: 'Faces' })}
                  </Text>
                  <div>{selectedResult.faceCount}</div>
                </div>
                <div className="rounded bg-bg-primary p-2">
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.detailEyes', { defaultValue: 'Eyes' })}
                  </Text>
                  <div>{selectedResult.eyeState}</div>
                </div>
              </div>
              {selectedResult.faceThumbnails.length > 0 && (
                <div className="mt-4">
                  <Text color={TextColors.secondary} variant={TextVariants.small}>
                    {t('modals.culling.localFaceCrops', { defaultValue: 'Local face crops' })}
                  </Text>
                  <div className="mt-2 flex gap-2">
                    {selectedResult.faceThumbnails.map((thumbnail, index) => (
                      <img
                        key={`${selectedResult.path}-face-${index}`}
                        src={thumbnail}
                        alt={t('modals.culling.detectedFace', {
                          index: index + 1,
                          defaultValue: 'Detected face {{index}}',
                        })}
                        className="h-12 w-12 rounded object-cover"
                      />
                    ))}
                  </div>
                </div>
              )}
              <Text variant={TextVariants.heading} className="mt-5">
                {t('modals.culling.why', { defaultValue: 'Why this result' })}
              </Text>
              <ul className="mt-2 space-y-1 text-sm text-text-secondary">
                {(selectedResult.reasons.length > 0 ? selectedResult.reasons : ['noDetectorWarning']).map((reason) => (
                  <li key={reason} className="flex gap-2">
                    <Check size={15} className="mt-0.5 shrink-0 text-accent" />
                    {reason === 'noDetectorWarning'
                      ? t('modals.culling.noDetectorWarning', {
                          defaultValue: 'No detector warning; this photo is unrated.',
                        })
                      : t(`modals.culling.reason.${reason}`, { defaultValue: reason })}
                  </li>
                ))}
              </ul>
              <Text color={TextColors.secondary} variant={TextVariants.small} className="mt-5">
                {t('modals.culling.detectorNote', {
                  method: selectedResult.qualityMethod,
                  defaultValue:
                    '{{method}}. Detector output is local and may be unknown when the signal is unavailable.',
                })}
              </Text>
            </>
          ) : (
            <div className="flex h-full items-center justify-center text-center">
              <Text color={TextColors.secondary}>
                {t('modals.culling.selectResult', {
                  defaultValue: 'Select a photo to see its Lightroom-style details and explanation.',
                })}
              </Text>
            </div>
          )}
        </div>
      </div>

      {(failedCount > 0 || (persistence?.skippedPaths.length || 0) > 0 || suggestions.failedPaths.length > 0) && (
        <div className="mt-3 rounded-lg border border-yellow-500/40 bg-yellow-500/10 p-3 text-sm text-text-secondary">
          {failedCount > 0 && (
            <div>
              {t('modals.culling.persistenceFailure', {
                count: failedCount,
                defaultValue: '{{count}} persistence operation group(s) failed; successful photos remain applied.',
              })}
            </div>
          )}
          {(persistence?.skippedPaths.length || 0) > 0 && (
            <div>
              {t('modals.culling.preservedCount', {
                count: persistence?.skippedPaths.length,
                defaultValue: '{{count}} existing decisions were preserved.',
              })}
            </div>
          )}
          {suggestions.failedPaths.length > 0 && (
            <div>
              {t('modals.culling.failedAnalysisCount', {
                count: suggestions.failedPaths.length,
                defaultValue: '{{count}} photo(s) could not be analyzed.',
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
