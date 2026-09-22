import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { CheckCircle, XCircle, Loader2, Users, Trash2, Star, Tag } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  CullingSettings,
  CullingSuggestions,
  ImageAnalysisResult,
  Invokes,
  Progress,
} from '../ui/AppProperties';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Slider from '../ui/Slider';
import Dropdown from '../ui/Dropdown';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { useUIStore } from '../../store/useUIStore';
import {
  beginCullingInvocation,
  createCullingInvocationId,
  cullingAnalysisMessageKey,
  emptyCullingResultsHeadline,
  populatedCullingResultsTab,
} from '../../utils/cullingReviewSession';

interface CullingModalProps {
  isOpen: boolean;
  onClose(): void;
  progress: Progress | null;
  suggestions: CullingSuggestions | null;
  error: string | null;
  imagePaths: string[];
  thumbnails: Record<string, string>;
  onApply(action: 'reject' | 'rate_zero' | 'delete', paths: string[]): void;
  onOpenImage(path: string): void;
  onError(error: string): void;
}

type CullAction = 'reject' | 'rate_zero' | 'delete';

interface ImageThumbnailProps {
  path: string;
  thumbnails: Record<string, string>;
  isSelected: boolean;
  onToggle(): void;
  children?: ReactNode;
}

function ImageThumbnail({ path, thumbnails, isSelected, onToggle, children }: ImageThumbnailProps) {
  const thumbnailUrl = thumbnails[path];
  return (
    <div
      className={`relative group rounded-md overflow-hidden border-2 transition-colors cursor-pointer ${
        isSelected ? 'border-accent' : 'border-transparent hover:border-surface'
      }`}
      onClick={onToggle}
    >
      <img
        src={thumbnailUrl}
        alt={path}
        className={`w-full h-full object-cover transition-opacity ${isSelected ? 'opacity-100' : 'opacity-75 group-hover:opacity-100'}`}
      />
      <div
        className={`absolute inset-0 bg-black/50 transition-opacity ${
          isSelected ? 'opacity-0' : 'opacity-100 group-hover:opacity-0'
        }`}
      />
      <div className="absolute top-2 right-2">{isSelected && <CheckCircle size={16} className="text-accent" />}</div>
      {children && (
        <Text
          as="div"
          variant={TextVariants.small}
          color={TextColors.white}
          className="absolute bottom-0 left-0 right-0 p-1 bg-black/60"
        >
          {children}
        </Text>
      )}
    </div>
  );
}

interface ReviewImageCardProps {
  image: ImageAnalysisResult;
  thumbnails: Record<string, string>;
  isSelected: boolean;
  onToggle(): void;
  onOpenImage(): void;
}

function ReviewImageCard({ image, thumbnails, isSelected, onToggle, onOpenImage }: ReviewImageCardProps) {
  const { t } = useTranslation();
  const alertLabels = image.reviewAlerts.map((alert) => {
    switch (alert) {
      case 'focusReview':
        return t('modals.culling.focusReviewAlert');
      case 'eyesClosed':
        return t('modals.culling.eyesClosedAlert');
      case 'eyesUnknown':
        return t('modals.culling.eyesUnknownAlert');
      case 'subjectUnknown':
        return t('modals.culling.subjectUnknown');
      default:
        return alert;
    }
  });
  const roleClass = (role: string) =>
    role === 'primary' ? 'border-amber-300' : role === 'unknown' ? 'border-red-300' : 'border-slate-300';

  return (
    <div className={`bg-surface rounded-md overflow-hidden border ${isSelected ? 'border-accent' : 'border-transparent'}`}>
      <button
        type="button"
        className="relative block w-full bg-black cursor-pointer"
        style={{ aspectRatio: `${image.width} / ${image.height}` }}
        onClick={onToggle}
        title={t('modals.culling.toggleManualReviewSelection')}
      >
        <img src={thumbnails[image.path]} alt={image.path} className="w-full h-full object-contain" />
        {image.subjectBoxes.map((box, index) => (
          <span
            key={`subject-${index}`}
            className="absolute border-2 border-cyan-300 pointer-events-none"
            style={{
              left: `${box.x * 100}%`,
              top: `${box.y * 100}%`,
              width: `${box.width * 100}%`,
              height: `${box.height * 100}%`,
            }}
          />
        ))}
        {image.attributedFaces.map((face, index) => (
          <span
            key={`face-${index}`}
            className={`absolute border-2 ${roleClass(face.role)} pointer-events-none`}
            style={{
              left: `${face.x * 100}%`,
              top: `${face.y * 100}%`,
              width: `${face.width * 100}%`,
              height: `${face.height * 100}%`,
            }}
          />
        ))}
        {isSelected && <CheckCircle size={18} className="absolute top-2 right-2 text-accent" />}
      </button>
      <div className="p-2 space-y-1">
        <Text variant={TextVariants.small} className="truncate" title={image.path}>
          {alertLabels.join(', ') || t('modals.culling.unknownReason')}
        </Text>
        <Text variant={TextVariants.small} className="text-text-secondary">
          {t('modals.culling.subjectStatus')}:{' '}
          {image.subjectStatus === 'multiple' ? t('modals.culling.multipleSubjects') : image.subjectStatus};{' '}
          {t('modals.culling.focusStatus')}: {image.focusStatus}
        </Text>
        {image.focusSignal !== null && (
          <Text variant={TextVariants.small} className="text-text-secondary">
            {t('modals.culling.focusScore', { score: image.focusSignal.toFixed(3) })}
            {image.focusCropWidth && image.focusCropHeight
              ? ` · ${t('modals.culling.nativeCrop', { width: image.focusCropWidth, height: image.focusCropHeight })}`
              : ''}
          </Text>
        )}
        {image.attributedFaces.length > 0 && (
          <Text variant={TextVariants.small} className="text-text-secondary">
            {t('modals.culling.facesDetected', { count: image.attributedFaces.length })}:{' '}
            {image.attributedFaces.map((face) => `${face.role}/${face.eyeState}`).join(', ')}
          </Text>
        )}
        <button
          type="button"
          className="text-xs text-accent hover:underline"
          onClick={(event) => {
            event.stopPropagation();
            onOpenImage();
          }}
        >
          {t('modals.culling.openInEditor')}
        </button>
      </div>
    </div>
  );
}

export default function CullingModal({
  isOpen,
  onClose,
  progress,
  suggestions,
  error,
  imagePaths,
  thumbnails,
  onApply,
  onOpenImage,
  onError,
}: CullingModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [stage, setStage] = useState<'settings' | 'progress' | 'results'>('settings');

  const [settings, setSettings] = useState<CullingSettings>({
    groupSimilar: true,
    similarityThreshold: 28,
    filterBlurry: true,
    blurThreshold: 100.0,
    selectionAmount: 'standard',
    blurSeverity: 'moderate',
    detectSubject: true,
    subjectProfile: 'general',
    detectClosedEyes: false,
    reviewFocus: true,
  });

  const [selectedRejects, setSelectedRejects] = useState<Set<string>>(new Set());
  const [action, setAction] = useState<CullAction>('reject');
  const [activeTab, setActiveTab] = useState<'similar' | 'blurry' | 'alerts' | 'unknown'>('similar');
  const retainedReviewRef = useRef(false);
  retainedReviewRef.current = Boolean(suggestions || error || progress);

  const CULL_ACTIONS = useMemo(
    () => [
      {
        value: 'reject' as const,
        label: t('modals.culling.actionReject'),
        icon: <Tag size={16} className="text-red-500" />,
      },
      { value: 'rate_zero' as const, label: t('modals.culling.actionRateZero'), icon: <Star size={16} /> },
      { value: 'delete' as const, label: t('modals.culling.actionDelete'), icon: <Trash2 size={16} /> },
    ],
    [t],
  );

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    }
    setShow(false);
    if (retainedReviewRef.current) {
      setIsMounted(false);
      return;
    }
    const timer = setTimeout(() => {
      setIsMounted(false);
      setStage('settings');
      setSelectedRejects(new Set());
    }, 300);
    return () => clearTimeout(timer);
  }, [isOpen]);

  useEffect(() => {
    if (suggestions || error) {
      setStage('results');
    } else if (progress) {
      setStage('progress');
    } else if (isOpen) {
      setStage('settings');
    }
  }, [progress, suggestions, error, isOpen]);

  useEffect(() => {
    if (stage === 'results' && suggestions) {
      const initialRejects = new Set<string>();
      suggestions.similarGroups.forEach((group) => {
        const keepCount =
          settings.selectionAmount === 'extreme'
            ? 1
            : settings.selectionAmount === 'few'
              ? Math.max(1, Math.ceil((group.duplicates.length + 1) * 0.25))
              : settings.selectionAmount === 'more'
                ? Math.max(1, Math.ceil((group.duplicates.length + 1) * 0.75))
                : Math.max(1, Math.ceil((group.duplicates.length + 1) * 0.5));
        group.duplicates.slice(Math.max(0, keepCount - 1)).forEach((dup) => initialRejects.add(dup.path));
      });
      suggestions.blurryImages.forEach((img) => initialRejects.add(img.path));
      setSelectedRejects(initialRejects);
    }
  }, [stage, suggestions, settings.selectionAmount]);

  useEffect(() => {
    if (suggestions) {
      setActiveTab(populatedCullingResultsTab(suggestions));
    }
  }, [suggestions]);

  const handleStartCulling = useCallback(async () => {
    const invocationId = createCullingInvocationId();
    useUIStore.getState().setUI((state) => ({
      cullingModalState: beginCullingInvocation(state.cullingModalState, invocationId),
    }));
    try {
      await invoke(Invokes.CullImages, { paths: imagePaths, settings, invocationId });
    } catch (err) {
      console.error('Culling failed to start:', err);
      onError(String(err));
    }
  }, [imagePaths, settings, onError]);

  const handleToggleReject = (path: string) => {
    setSelectedRejects((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  };

  const handleApply = () => {
    onApply(action, Array.from(selectedRejects));
  };

  const numSimilar = suggestions?.similarGroups.reduce((acc, group) => acc + group.duplicates.length, 0) || 0;
  const numBlurry = suggestions?.blurryImages.length || 0;
  const numAlerts = suggestions?.reviewAlerts.length || 0;
  const numUnknown = suggestions?.unknownImages.length || 0;

  const renderSettings = () => (
    <>
      <div className="flex items-center justify-center mb-4">
        <Users className="w-12 h-12 text-accent" />
      </div>
      <Text variant={TextVariants.title} className="mb-6 text-center">
        {t('modals.culling.title')}
      </Text>
      <div className="space-y-6 text-sm">
        <div className="space-y-3">
          <Text variant={TextVariants.heading}>{t('modals.culling.assistedReview')}</Text>
          <Text variant={TextVariants.small}>{t('modals.culling.assistedReviewDesc')}</Text>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <Text variant={TextVariants.label} className="mb-1">
                {t('modals.culling.selectionAmount')}
              </Text>
              <Dropdown
                options={[
                  { value: 'extreme' as const, label: t('modals.culling.selectionAmountExtreme') },
                  { value: 'few' as const, label: t('modals.culling.selectionAmountFew') },
                  { value: 'standard' as const, label: t('modals.culling.selectionAmountStandard') },
                  { value: 'more' as const, label: t('modals.culling.selectionAmountMore') },
                ]}
                value={settings.selectionAmount}
                onChange={(value) => setSettings((s) => ({ ...s, selectionAmount: value }))}
              />
            </div>
            <div>
              <Text variant={TextVariants.label} className="mb-1">
                {t('modals.culling.blurSeverity')}
              </Text>
              <Dropdown
                options={[
                  { value: 'lenient' as const, label: t('modals.culling.blurSeverityLenient') },
                  { value: 'moderate' as const, label: t('modals.culling.blurSeverityModerate') },
                  { value: 'strict' as const, label: t('modals.culling.blurSeverityStrict') },
                ]}
                value={settings.blurSeverity}
                onChange={(value) => setSettings((s) => ({ ...s, blurSeverity: value }))}
              />
              <Text variant={TextVariants.small} className="mt-1">
                {t('modals.culling.blurThresholdDesc')}
              </Text>
            </div>
          </div>
        </div>
        <div className="space-y-3">
          <Switch
            label={t('modals.culling.detectSubject')}
            checked={settings.detectSubject}
            onChange={(value) => setSettings((s) => ({ ...s, detectSubject: value }))}
          />
          {settings.detectSubject && (
            <div className="pl-4 border-l-2 border-border-color ml-1">
              <Text variant={TextVariants.label} className="mb-1">
                {t('modals.culling.subjectProfile')}
              </Text>
              <Dropdown
                options={[
                  { value: 'general' as const, label: t('modals.culling.profileGeneral') },
                  { value: 'portrait' as const, label: t('modals.culling.profilePortrait') },
                  { value: 'wedding' as const, label: t('modals.culling.profileWedding') },
                  { value: 'sports' as const, label: t('modals.culling.profileSports') },
                  { value: 'dance' as const, label: t('modals.culling.profileDance') },
                ]}
                value={settings.subjectProfile}
                onChange={(value) => setSettings((s) => ({ ...s, subjectProfile: value }))}
              />
            </div>
          )}
          <Switch
            label={t('modals.culling.detectClosedEyes')}
            checked={settings.detectClosedEyes}
            onChange={(value) => setSettings((s) => ({ ...s, detectClosedEyes: value }))}
          />
          <Switch
            label={t('modals.culling.reviewFocus')}
            checked={settings.reviewFocus}
            onChange={(value) => setSettings((s) => ({ ...s, reviewFocus: value }))}
          />
          <Text variant={TextVariants.small}>{t('modals.culling.assistedSignalsNote')}</Text>
        </div>
        <div>
          <Switch
            label={t('modals.culling.groupSimilar')}
            checked={settings.groupSimilar}
            onChange={(v) => setSettings((s) => ({ ...s, groupSimilar: v }))}
          />
          {settings.groupSimilar && (
            <div className="mt-2 pl-4 border-l-2 border-border-color ml-1">
              <Slider
                label={t('modals.culling.similarityThreshold')}
                min={1}
                max={64}
                step={1}
                value={settings.similarityThreshold}
                defaultValue={28}
                onChange={(e) => setSettings((s) => ({ ...s, similarityThreshold: Number(e.target.value) }))}
                fillOrigin="min"
              />
              <Text variant={TextVariants.small} className="mt-1">
                {t('modals.culling.similarityThresholdDesc')}
              </Text>
            </div>
          )}
        </div>
        <div>
          <Switch
            label={t('modals.culling.filterBlurry')}
            checked={settings.filterBlurry}
            onChange={(v) => setSettings((s) => ({ ...s, filterBlurry: v }))}
          />
        </div>
      </div>
      <div className="flex justify-end gap-3 mt-8">
        <button
          className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors"
          onClick={onClose}
        >
          {t('modals.culling.cancel')}
        </button>
        <Button onClick={handleStartCulling}>{t('modals.culling.startCulling')}</Button>
      </div>
    </>
  );

  const renderProgress = () => (
    <div className="flex flex-col items-center justify-center h-48">
      <Loader2 className="w-16 h-16 text-accent animate-spin" />
      <p className="mt-4 text-text-primary">{progress?.stage || t('modals.culling.starting')}</p>
      {progress && progress.total > 0 && (
        <div className="w-full bg-surface rounded-full h-2.5 mt-2">
          <div
            className="bg-accent h-2.5 rounded-full"
            style={{ width: `${(progress.current / progress.total) * 100}%` }}
          />
        </div>
      )}
    </div>
  );

  const renderResults = () => {
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-48">
          <XCircle className="w-16 h-16 text-red-500" />
          <Text variant={TextVariants.heading} className="mt-4 text-center">
            {t('modals.culling.cullingFailed')}
          </Text>
          <Text>{error}</Text>
          <div className="mt-6">
            <Button onClick={onClose}>{t('modals.culling.close')}</Button>
          </div>
        </div>
      );
    }

    if (!suggestions) return null;

    const totalSuggestions = numSimilar + numBlurry + numAlerts + numUnknown;
    const analysisMessageKey = cullingAnalysisMessageKey(suggestions.subjectAnalysisStatus);
    if (totalSuggestions === 0) {
      const headline = emptyCullingResultsHeadline(suggestions.subjectAnalysisStatus);
      if (headline !== 'noIssuesFound') {
        return (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Text variant={TextVariants.heading} className="mt-4">
              {t(`modals.culling.${headline}`)}
            </Text>
            <div className="mt-6">
              <Button onClick={onClose}>{t('modals.culling.done')}</Button>
            </div>
          </div>
        );
      }
      return (
        <div className="flex flex-col items-center justify-center h-48">
          <CheckCircle className="w-16 h-16 text-green-500" />
          <Text variant={TextVariants.heading} className="mt-4">
            {t('modals.culling.noIssuesFound')}
          </Text>
          <Text>{t('modals.culling.noIssuesDesc')}</Text>
          <div className="mt-6">
            <Button onClick={onClose}>{t('modals.culling.done')}</Button>
          </div>
        </div>
      );
    }

    return (
      <>
        <Text variant={TextVariants.title} className="mb-4">
          {t('modals.culling.cullingSuggestions')}
        </Text>
        {analysisMessageKey && (
          <Text variant={TextVariants.small} className="mb-3 text-text-secondary">
            {t(`modals.culling.${analysisMessageKey}`)}
          </Text>
        )}
        <div className="border-b border-surface mb-4">
          <nav className="-mb-px flex space-x-4" aria-label="Tabs">
            {numSimilar > 0 && (
              <button
                onClick={() => setActiveTab('similar')}
                className={`${
                  activeTab === 'similar'
                    ? 'border-accent text-accent'
                    : 'border-transparent text-text-secondary hover:text-text-primary hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              >
                {t('modals.culling.similarGroupsTab')}{' '}
                <span className="bg-surface text-text-secondary rounded-full px-2 py-0.5 text-xs">{numSimilar}</span>
              </button>
            )}
            {numBlurry > 0 && (
              <button
                onClick={() => setActiveTab('blurry')}
                className={`${
                  activeTab === 'blurry'
                    ? 'border-accent text-accent'
                    : 'border-transparent text-text-secondary hover:text-text-primary hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              >
                {t('modals.culling.blurryImagesTab')}{' '}
                <span className="bg-surface text-text-secondary rounded-full px-2 py-0.5 text-xs">{numBlurry}</span>
              </button>
            )}
            {numAlerts > 0 && (
              <button
                onClick={() => setActiveTab('alerts')}
                className={`${
                  activeTab === 'alerts'
                    ? 'border-accent text-accent'
                    : 'border-transparent text-text-secondary hover:text-text-primary hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              >
                {t('modals.culling.reviewAlertsTab')}{' '}
                <span className="bg-surface text-text-secondary rounded-full px-2 py-0.5 text-xs">{numAlerts}</span>
              </button>
            )}
            {numUnknown > 0 && (
              <button
                onClick={() => setActiveTab('unknown')}
                className={`${
                  activeTab === 'unknown'
                    ? 'border-accent text-accent'
                    : 'border-transparent text-text-secondary hover:text-text-primary hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm`}
              >
                {t('modals.culling.unknownTab')}{' '}
                <span className="bg-surface text-text-secondary rounded-full px-2 py-0.5 text-xs">{numUnknown}</span>
              </button>
            )}
          </nav>
        </div>

        <div className="bg-bg-primary rounded-lg p-2 h-[50vh] overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === 'similar' && (
                <div className="space-y-4">
                  {suggestions.similarGroups.map((group, index) => (
                    <div key={index} className="bg-surface rounded-lg p-3">
                      <Text variant={TextVariants.heading} className="mb-2">
                        {t('modals.culling.groupHeader', { index: index + 1 })}
                      </Text>
                      <div className="grid grid-cols-[1fr_3fr] gap-3">
                        <div>
                          <Text variant={TextVariants.label} className="mb-1">
                            {t('modals.culling.bestImage')}
                          </Text>
                          <div className="relative rounded-md overflow-hidden border-2 border-green-500">
                            <img
                              src={thumbnails[group.representative.path]}
                              alt="Representative"
                              className="w-full h-full object-cover"
                            />
                            <Text
                              as="div"
                              variant={TextVariants.small}
                              color={TextColors.white}
                              className="absolute bottom-0 left-0 right-0 p-1 bg-black/60"
                            >
                              {t('modals.culling.score', { score: group.representative.qualityScore.toFixed(2) })}
                            </Text>
                          </div>
                        </div>
                        <div>
                          <Text variant={TextVariants.label} className="mb-1">
                            {t('modals.culling.duplicatesHeader', { count: group.duplicates.length })}
                          </Text>
                          <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
                            {group.duplicates.map((dup) => (
                              <ImageThumbnail
                                key={dup.path}
                                path={dup.path}
                                thumbnails={thumbnails}
                                isSelected={selectedRejects.has(dup.path)}
                                onToggle={() => handleToggleReject(dup.path)}
                              >
                                {t('modals.culling.score', { score: dup.qualityScore.toFixed(2) })}
                              </ImageThumbnail>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {activeTab === 'blurry' && (
                <div className="grid grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
                  {suggestions.blurryImages.map((img) => (
                    <ImageThumbnail
                      key={img.path}
                      path={img.path}
                      thumbnails={thumbnails}
                      isSelected={selectedRejects.has(img.path)}
                      onToggle={() => handleToggleReject(img.path)}
                    >
                      {t('modals.culling.sharpness', { sharpness: img.sharpnessMetric.toFixed(0) })}
                    </ImageThumbnail>
                  ))}
                </div>
              )}
              {(activeTab === 'alerts' || activeTab === 'unknown') && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {(activeTab === 'alerts' ? suggestions.reviewAlerts : suggestions.unknownImages).map((img) => (
                    <ReviewImageCard
                      key={img.path}
                      image={img}
                      thumbnails={thumbnails}
                      isSelected={selectedRejects.has(img.path)}
                      onToggle={() => handleToggleReject(img.path)}
                      onOpenImage={() => onOpenImage(img.path)}
                    />
                  ))}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="flex justify-between items-center gap-3 mt-6">
          <div className="flex-1">
            <Dropdown
              options={CULL_ACTIONS.map(({ value, label }) => ({ value, label }))}
              value={action}
              onChange={(newValue: CullAction) => setAction(newValue)}
              className="w-full"
            />
          </div>
          <div className="flex gap-3">
            <button
              className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors"
              onClick={onClose}
            >
              {t('modals.culling.cancel')}
            </button>
            <Button onClick={handleApply} disabled={selectedRejects.size === 0}>
              {t('modals.culling.applyButton', { count: selectedRejects.size })}
            </Button>
          </div>
        </div>
      </>
    );
  };

  const renderContent = () => {
    switch (stage) {
      case 'settings':
        return renderSettings();
      case 'progress':
        return renderProgress();
      case 'results':
        return renderResults();
      default:
        return null;
    }
  };

  if (!isMounted) return null;

  return (
    <div
      className={`fixed inset-0 flex items-center justify-center z-50 bg-black/30 backdrop-blur-xs transition-opacity duration-300 ease-in-out ${
        show ? 'opacity-100' : 'opacity-0'
      }`}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className={`bg-surface rounded-lg shadow-xl p-6 w-full max-w-3xl transform transition-all duration-300 ease-out ${
          show ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 -translate-y-4'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {renderContent()}
      </div>
    </div>
  );
}
