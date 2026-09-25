import { WandSparkles } from 'lucide-react';
import Button from '../../ui/Button';

export interface AutomaticCullingModalState {
  isOpen: boolean;
  suggestions: null;
  progress: null;
  error: null;
  pathsToCull: string[];
  folderPath: string | null;
}

interface AutomaticCullingButtonProps {
  label: string;
  unavailableLabel: string;
  folderPath: string | null;
  folderPaths: string[];
  onOpen(state: AutomaticCullingModalState): void;
}

export function createAutomaticCullingModalState(
  folderPaths: string[],
  folderPath: string | null = null,
): AutomaticCullingModalState {
  return {
    isOpen: true,
    suggestions: null,
    progress: null,
    error: null,
    pathsToCull: [...folderPaths],
    folderPath,
  };
}

export function openAutomaticCullingOptions(
  folderPaths: string[],
  folderPath: string | null,
  onOpen: (state: AutomaticCullingModalState) => void,
): void {
  if (!folderPath) return;
  onOpen(createAutomaticCullingModalState(folderPaths, folderPath));
}

export default function AutomaticCullingButton({
  label,
  unavailableLabel,
  folderPath,
  folderPaths,
  onOpen,
}: AutomaticCullingButtonProps) {
  const isUnavailable = !folderPath;
  const accessibleLabel = isUnavailable ? unavailableLabel : label;

  return (
    <span data-tooltip={accessibleLabel}>
      <Button
        className="h-14 px-4 whitespace-nowrap"
        disabled={isUnavailable}
        onClick={() => openAutomaticCullingOptions(folderPaths, folderPath, onOpen)}
        aria-label={accessibleLabel}
        title={accessibleLabel}
      >
        <WandSparkles className="w-5 h-5" aria-hidden="true" />
        <span>{label}</span>
      </Button>
    </span>
  );
}
