"""
Filter metadata files to keep only entries with specific topics.

This script processes metadata files in the data/trials directory and creates
filtered versions that keep only entries matching the allowed topics.

Allowed topics:
- observations/state
- observations/events/player/victim_picked_up
- observations/events/player/victim_placed
- observations/events/player/triage
- observations/events/player/rubble_collapse
- observations/events/player/rubble_destroyed
- observations/events/server/victim_evacuated
- observations/events/player/location

Also filters out entries with mission_timer "Mission Timer not initialized."
"""

import json
from pathlib import Path
from typing import Optional, Set


# Allowed topics - only entries with these topics will be kept
ALLOWED_TOPICS: Set[str] = {
    "observations/state",
    "observations/events/player/victim_picked_up",
    "observations/events/player/victim_placed",
    "observations/events/player/triage",
    "observations/events/player/rubble_collapse",
    "observations/events/player/rubble_destroyed",
    "observations/events/server/victim_evacuated",
    "observations/events/player/location",
}


def get_topic_from_entry(entry: dict) -> Optional[str]:
    """
    Extract topic from an entry, checking multiple possible locations.
    
    Args:
        entry: A JSON object (dictionary) from the metadata file
        
    Returns:
        The topic string if found, None otherwise
    """
    # Check topic at top level
    if 'topic' in entry:
        topic = entry.get('topic')
        if isinstance(topic, str):
            return topic
    
    # Check topic in msg field
    if 'msg' in entry and isinstance(entry['msg'], dict):
        if 'topic' in entry['msg']:
            topic = entry['msg'].get('topic')
            if isinstance(topic, str):
                return topic
    
    # Check topic in data field
    if 'data' in entry and isinstance(entry['data'], dict):
        if 'topic' in entry['data']:
            topic = entry['data'].get('topic')
            if isinstance(topic, str):
                return topic
    
    return None


def has_invalid_mission_timer(entry: dict) -> bool:
    """
    Check if entry contains mission_timer "Mission Timer not initialized."
    
    Args:
        entry: A JSON object (dictionary) from the metadata file
        
    Returns:
        True if entry contains invalid mission_timer, False otherwise
    """
    # Check mission_timer in data field
    if 'data' in entry and isinstance(entry['data'], dict):
        if 'mission_timer' in entry['data']:
            mission_timer = entry['data'].get('mission_timer')
            if mission_timer == "Mission Timer not initialized.":
                return True
    
    # Check mission_timer at top level
    if 'mission_timer' in entry:
        mission_timer = entry.get('mission_timer')
        if mission_timer == "Mission Timer not initialized.":
            return True
    
    return False


def should_keep_entry(entry: dict) -> bool:
    """
    Check if an entry should be kept (i.e., has an allowed topic and valid mission_timer).
    
    Args:
        entry: A JSON object (dictionary) from the metadata file
        
    Returns:
        True if the entry has an allowed topic and valid mission_timer, False otherwise
    """
    # First check: exclude entries with invalid mission_timer
    if has_invalid_mission_timer(entry):
        return False
    
    # Second check: only keep entries with allowed topics
    topic = get_topic_from_entry(entry)
    
    if topic is None:
        return False
    
    return topic in ALLOWED_TOPICS


def filter_metadata_file(input_path: Path, output_path: Optional[Path] = None) -> tuple[int, int]:
    """
    Filter a metadata file to keep only entries with allowed topics.
    Also filters out entries with mission_timer "Mission Timer not initialized."
    
    Args:
        input_path: Path to the input metadata file
        output_path: Path to the output file. If None, creates a file with 
                     "_filtered" suffix in the same directory.
        
    Returns:
        Tuple of (total_entries, kept_entries)
    """
    if output_path is None:
        # Create output filename by adding "_filtered" before the extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_filtered{suffix}"
    
    total_entries = 0
    kept_entries = 0
    filtered_entries = 0
    topic_counts = {topic: 0 for topic in ALLOWED_TOPICS}
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"\nAllowed topics ({len(ALLOWED_TOPICS)}):")
    for topic in sorted(ALLOWED_TOPICS):
        print(f"  - {topic}")
    print()
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            total_entries += 1
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                continue
            
            if should_keep_entry(entry):
                topic = get_topic_from_entry(entry)
                if topic and topic in topic_counts:
                    topic_counts[topic] += 1
                outfile.write(line + '\n')
                kept_entries += 1
            else:
                filtered_entries += 1
    
    print(f"Summary:")
    print(f"  Total entries: {total_entries}")
    print(f"  Kept entries: {kept_entries}")
    print(f"  Filtered entries: {filtered_entries}")
    print(f"  Keep rate: {kept_entries/total_entries*100:.2f}%")
    print(f"\nTopic distribution:")
    for topic in sorted(ALLOWED_TOPICS):
        count = topic_counts[topic]
        if count > 0:
            print(f"  {count:6d}  {topic}")
    
    return total_entries, kept_entries


def main():
    """Process all metadata files in the data/trials directory."""
    trials_dir = Path(__file__).parent.parent / 'data' / 'trials'
    
    if not trials_dir.exists():
        print(f"Error: Trials directory not found: {trials_dir}")
        return
    
    # Find all metadata files (excluding already filtered ones)
    metadata_files = [f for f in trials_dir.glob('*.metadata') 
                     if not f.name.endswith('_filtered.metadata')]
    
    if not metadata_files:
        print(f"No metadata files found in {trials_dir}")
        return
    
    print(f"Found {len(metadata_files)} metadata file(s)\n")
    
    for metadata_file in metadata_files:
        print(f"{'='*70}")
        try:
            filter_metadata_file(metadata_file)
            print()
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}\n")
    
    print(f"{'='*70}")
    print("Processing complete!")


if __name__ == '__main__':
    main()
