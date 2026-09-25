export function retainExplicitGallerySelection(currentId: string, availableIds: string[]): string {
  return currentId && availableIds.includes(currentId) ? currentId : '';
}

export function picPortalDestinationChanged(
  previousAccount: string | null,
  previousGalleryId: string,
  nextAccount: string | null,
  nextGalleryId: string,
): boolean {
  return previousAccount !== nextAccount || previousGalleryId !== nextGalleryId;
}
