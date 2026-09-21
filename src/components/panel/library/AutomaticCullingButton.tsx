import { WandSparkles } from 'lucide-react';
import Button from '../../ui/Button';

export interface AutomaticCullingModalState {
  isOpen: boolean;
  suggestions: null;
  progress: null;
  error: null;
  pathsToCull: string[];
}

interface AutomaticCullingButtonProps {
  label: string;
  unavailableLabel: string;
  selectedPaths: string[];
  onOpen(state: AutomaticCullingModalState): void;
}

export function createAutomaticCullingModalState(selectedPaths: string[]): AutomaticCullingModalState {
  return {
    isOpen: true,
    suggestions: null,
    progress: null,
    error: null,
    pathsToCull: [...selectedPaths],
  };
}

export function openAutomaticCullingOptions(
  selectedPaths: string[],
  onOpen: (state: AutomaticCullingModalState) => void,
): void {
  if (selectedPaths.length < 2) return;
  onOpen(createAutomaticCullingModalState(selectedPaths));
}

export default function AutomaticCullingButton({
  label,
  unavailableLabel,
  selectedPaths,
  onOpen,
}: AutomaticCullingButtonProps) {
  const isUnavailable = selectedPaths.length < 2;
  const accessibleLabel = isUnavailable ? unavailableLabel : label;

  return (
    <span data-tooltip={accessibleLabel}>
      <Button
        className="h-14 px-4 whitespace-nowrap"
        disabled={isUnavailable}
        onClick={() => openAutomaticCullingOptions(selectedPaths, onOpen)}
        aria-label={accessibleLabel}
        title={accessibleLabel}
      >
        <WandSparkles className="w-5 h-5" aria-hidden="true" />
        <span>{label}</span>
      </Button>
    </span>
  );
}
