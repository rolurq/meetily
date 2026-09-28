'use client';

import { DiarizationModelManager } from '@/components/DiarizationModelManager';
import { SpeakerProfilesSettings } from '@/components/SpeakerProfilesSettings';

/**
 * Settings tab for speaker identification (diarization): which voiceprint model
 * to use, and the list of speakers diarization has learned so far.
 */
export function SpeakerSettings() {
  return (
    <div className="flex flex-col gap-4">
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <DiarizationModelManager />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-1">Saved Speakers</h3>
        <p className="text-sm text-gray-600 mb-6">
          Speakers diarization has identified across your meetings. Rename or remove them here.
        </p>
        <SpeakerProfilesSettings />
      </div>
    </div>
  );
}
