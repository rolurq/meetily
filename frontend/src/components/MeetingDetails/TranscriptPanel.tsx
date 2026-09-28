"use client";

import { Transcript, TranscriptSegmentData } from '@/types';
import { TranscriptView } from '@/components/TranscriptView';
import { VirtualizedTranscriptView } from '@/components/VirtualizedTranscriptView';
import { TranscriptButtonGroup } from './TranscriptButtonGroup';
import { useMemo, useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { toast } from 'sonner';

interface SpeakerProfile {
  id: string;
  name: string;
  is_named: boolean;
}

interface TranscriptPanelProps {
  transcripts: Transcript[];
  customPrompt: string;
  onPromptChange: (value: string) => void;
  onCopyTranscript: () => void;
  onOpenMeetingFolder: () => Promise<void>;
  isRecording: boolean;
  disableAutoScroll?: boolean;

  // Optional pagination props (when using virtualization)
  usePagination?: boolean;
  segments?: TranscriptSegmentData[];
  hasMore?: boolean;
  isLoadingMore?: boolean;
  totalCount?: number;
  loadedCount?: number;
  onLoadMore?: () => void;

  // Retranscription props
  meetingId?: string;
  meetingFolderPath?: string | null;
  onRefetchTranscripts?: () => Promise<void>;
}

export function TranscriptPanel({
  transcripts,
  customPrompt,
  onPromptChange,
  onCopyTranscript,
  onOpenMeetingFolder,
  isRecording,
  disableAutoScroll = false,
  usePagination = false,
  segments,
  hasMore,
  isLoadingMore,
  totalCount,
  loadedCount,
  onLoadMore,
  meetingId,
  meetingFolderPath,
  onRefetchTranscripts,
}: TranscriptPanelProps) {
  // Convert transcripts to segments if pagination is not used but we want virtualization
  const convertedSegments = useMemo(() => {
    if (usePagination && segments) {
      return segments;
    }
    // Convert transcripts to segments for virtualization
    return transcripts.map(t => ({
      id: t.id,
      timestamp: t.audio_start_time ?? 0,
      endTime: t.audio_end_time,
      text: t.text,
      confidence: t.confidence,
      speakerId: t.speaker_id,
      speakerConfidence: t.speaker_confidence,
    }));
  }, [transcripts, usePagination, segments]);

  // Known speaker profiles, for showing names on segments and offering reassignment.
  const [speakerProfiles, setSpeakerProfiles] = useState<SpeakerProfile[]>([]);

  const loadSpeakerProfiles = useCallback(async () => {
    try {
      const profiles = await invoke<SpeakerProfile[]>('diarization_list_speaker_profiles');
      setSpeakerProfiles(profiles);
    } catch (error) {
      console.error('Failed to load speaker profiles:', error);
    }
  }, []);

  useEffect(() => {
    loadSpeakerProfiles();
  }, [loadSpeakerProfiles]);

  // Refresh the speaker list and transcript segments whenever diarization runs or a
  // segment is reassigned for this meeting, so newly detected/renamed speakers show up.
  useEffect(() => {
    if (!meetingId) return;
    let unlistenComplete: (() => void) | undefined;
    let unlistenUpdated: (() => void) | undefined;

    listen<{ meetingId: string }>('diarization-complete', (event) => {
      if (event.payload.meetingId !== meetingId) return;
      loadSpeakerProfiles();
      onRefetchTranscripts?.();
    }).then((fn) => (unlistenComplete = fn));

    listen<{ meetingId: string }>('diarization-segment-updated', (event) => {
      if (event.payload.meetingId !== meetingId) return;
      loadSpeakerProfiles();
      onRefetchTranscripts?.();
    }).then((fn) => (unlistenUpdated = fn));

    return () => {
      unlistenComplete?.();
      unlistenUpdated?.();
    };
  }, [meetingId, loadSpeakerProfiles, onRefetchTranscripts]);

  const handleReassignSegment = useCallback(async (transcriptId: string, targetSpeakerId: string) => {
    if (!meetingId) return;
    try {
      await invoke('diarization_reassign_segment', {
        meetingId,
        transcriptId,
        targetSpeakerId,
      });
      await Promise.all([loadSpeakerProfiles(), onRefetchTranscripts?.()]);
      toast.success('Speaker updated');
    } catch (error) {
      console.error('Failed to reassign speaker:', error);
      toast.error('Failed to update speaker', { description: String(error) });
    }
  }, [meetingId, loadSpeakerProfiles, onRefetchTranscripts]);

  return (
    <div className="flex h-full min-w-0 w-full bg-white flex-col relative @container">
      {/* Title area */}
      <div className="p-4 border-b border-gray-200">
        <TranscriptButtonGroup
          transcriptCount={usePagination ? (totalCount ?? convertedSegments.length) : (transcripts?.length || 0)}
          onCopyTranscript={onCopyTranscript}
          onOpenMeetingFolder={onOpenMeetingFolder}
          meetingId={meetingId}
          meetingFolderPath={meetingFolderPath}
          onRefetchTranscripts={onRefetchTranscripts}
        />
      </div>

      {/* Transcript content - use virtualized view for better performance */}
      <div className="flex-1 overflow-hidden pb-4">
        <VirtualizedTranscriptView
          segments={convertedSegments}
          isRecording={isRecording}
          isPaused={false}
          isProcessing={false}
          isStopping={false}
          enableStreaming={false}
          showConfidence={true}
          disableAutoScroll={disableAutoScroll}
          hasMore={hasMore}
          isLoadingMore={isLoadingMore}
          totalCount={totalCount}
          loadedCount={loadedCount}
          onLoadMore={onLoadMore}
          speakerProfiles={speakerProfiles}
          onReassignSegment={meetingId ? handleReassignSegment : undefined}
        />
      </div>

      {/* Custom prompt input at bottom of transcript section */}
      {!isRecording && convertedSegments.length > 0 && (
        <div className="p-1 border-t border-gray-200">
          <textarea
            placeholder="Add context for AI summary. For example people involved, meeting overview, objective etc..."
            className="w-full px-3 py-2 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 bg-white shadow-sm min-h-[80px] resize-y"
            value={customPrompt}
            onChange={(e) => onPromptChange(e.target.value)}
          />
        </div>
      )}
    </div>
  );
}
