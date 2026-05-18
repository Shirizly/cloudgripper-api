"""
Action Tracking System Validation Script

Use this script to verify that the action tracking system is working correctly
in your recording setup. Run it after a recording session to validate the output.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


class ActionTrackingValidator:
    """Validates action tracking output files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.states_file = os.path.join(output_dir, "states.json")
        self.actions_file = os.path.join(output_dir, "actions.json")
        self.states_data = None
        self.actions_data = None
        self.issues = []
        self.warnings = []
        self.info = []
    
    def validate(self) -> Tuple[bool, str]:
        """
        Run full validation suite.
        
        Returns:
            (is_valid: bool, report: str)
        """
        print("="*70)
        print("ACTION TRACKING VALIDATION REPORT")
        print("="*70)
        print(f"Output directory: {self.output_dir}\n")
        
        # Check file existence
        self._check_files_exist()
        
        # Load and validate JSON
        self._load_json_files()
        
        if self.states_data:
            self._validate_states_json()
        
        if self.actions_data:
            self._validate_actions_json()
        
        # Cross-validation
        if self.states_data and self.actions_data:
            self._cross_validate_states_and_actions()
        
        # Generate report
        report = self._generate_report()
        is_valid = len(self.issues) == 0
        
        print(report)
        print("="*70)
        
        return is_valid, report
    
    def _check_files_exist(self):
        """Check if required files exist."""
        print("STATUS: Checking files...")
        
        states_exists = os.path.exists(self.states_file)
        actions_exists = os.path.exists(self.actions_file)
        
        if states_exists:
            self.info.append(f"✓ states.json found ({os.path.getsize(self.states_file)} bytes)")
        else:
            self.issues.append("✗ states.json not found")
        
        if actions_exists:
            self.info.append(f"✓ actions.json found ({os.path.getsize(self.actions_file)} bytes)")
        else:
            self.warnings.append("⚠ actions.json not found (check if recording completed)")
        
        print()
    
    def _load_json_files(self):
        """Load JSON files."""
        print("STATUS: Loading JSON files...")
        
        if os.path.exists(self.states_file):
            try:
                with open(self.states_file, 'r') as f:
                    self.states_data = json.load(f)
                self.info.append(f"✓ states.json loaded ({len(self.states_data)} frames)")
            except Exception as e:
                self.issues.append(f"✗ Failed to load states.json: {e}")
        
        if os.path.exists(self.actions_file):
            try:
                with open(self.actions_file, 'r') as f:
                    self.actions_data = json.load(f)
                self.info.append(f"✓ actions.json loaded ({self.actions_data.get('total_actions', 0)} actions)")
            except Exception as e:
                self.issues.append(f"✗ Failed to load actions.json: {e}")
        
        print()
    
    def _validate_states_json(self):
        """Validate states.json structure and content."""
        print("STATUS: Validating states.json...")
        
        assert isinstance(self.states_data, list), "states.json should be a list"
        
        frames_with_actions = 0
        frames_without_actions = 0
        invalid_entries = 0
        
        for idx, state_entry in enumerate(self.states_data):
            # Check required fields
            if not isinstance(state_entry, dict):
                self.issues.append(f"  Frame {idx}: Entry is not a dict")
                invalid_entries += 1
                continue
            
            if 'frame_index' not in state_entry:
                self.warnings.append(f"  Frame {idx}: Missing frame_index field")
            
            if 'time' not in state_entry:
                self.warnings.append(f"  Frame {idx}: Missing time field")
            
            # Check action field
            if 'action' in state_entry:
                frames_with_actions += 1
                self._validate_action_in_state(idx, state_entry['action'])
            else:
                frames_without_actions += 1
        
        self.info.append(f"✓ {frames_with_actions} frames have action metadata")
        self.info.append(f"✓ {frames_without_actions} frames without action metadata")
        
        if invalid_entries > 0:
            self.issues.append(f"✗ {invalid_entries} invalid frame entries")
        
        print()
    
    def _validate_action_in_state(self, frame_idx: int, action: Dict):
        """Validate action embedded in state entry."""
        required_fields = [
            'action_id', 'action_type', 'phase',
            'start_frame', 'end_frame', 'is_planar_2d'
        ]
        
        for field in required_fields:
            if field not in action:
                self.warnings.append(
                    f"  Frame {frame_idx}: Action missing field '{field}'"
                )
        
        # Validate frame range
        start = action.get('start_frame')
        end = action.get('end_frame')
        
        if start is not None and end is not None:
            if end < start:
                self.issues.append(
                    f"  Frame {frame_idx}: Action has invalid frame range "
                    f"({start}-{end})"
                )
    
    def _validate_actions_json(self):
        """Validate actions.json structure and content."""
        print("STATUS: Validating actions.json...")
        
        assert isinstance(self.actions_data, dict), "actions.json should be a dict"
        
        total_actions = self.actions_data.get('total_actions', 0)
        actions_list = self.actions_data.get('actions', [])
        
        if total_actions != len(actions_list):
            self.issues.append(
                f"✗ Mismatch: total_actions={total_actions} "
                f"but {len(actions_list)} actions in list"
            )
        
        # Validate each action
        planar_2d_count = 0
        action_types = {}
        action_phases = {}
        
        for action in actions_list:
            if action.get('is_planar_2d'):
                planar_2d_count += 1
            
            action_type = action.get('action_type')
            action_types[action_type] = action_types.get(action_type, 0) + 1
            
            phase = action.get('phase')
            action_phases[phase] = action_phases.get(phase, 0) + 1
            
            # Validate action structure
            self._validate_action_structure(action)
        
        self.info.append(f"✓ {planar_2d_count} planar 2D actions")
        self.info.append(f"✓ Action types: {dict(action_types)}")
        self.info.append(f"✓ Action phases: {dict(action_phases)}")
        
        print()
    
    def _validate_action_structure(self, action: Dict):
        """Validate individual action structure."""
        required_fields = [
            'action_id', 'action_type', 'phase',
            'start_frame', 'end_frame', 'is_planar_2d'
        ]
        
        action_id = action.get('action_id')
        
        for field in required_fields:
            if field not in action:
                self.warnings.append(f"  Action {action_id}: Missing field '{field}'")
        
        # Validate frame range
        start = action.get('start_frame')
        end = action.get('end_frame')
        
        if start is not None and end is not None:
            if start < 0 or end < 0:
                self.issues.append(
                    f"  Action {action_id}: Negative frame index "
                    f"({start}-{end})"
                )
            
            if end < start:
                self.issues.append(
                    f"  Action {action_id}: Invalid frame range "
                    f"({start}-{end})"
                )
            
            duration = end - start
            if duration > 10000:  # Heuristic: check if suspiciously long
                self.warnings.append(
                    f"  Action {action_id}: Very long duration ({duration} frames)"
                )
    
    def _cross_validate_states_and_actions(self):
        """Cross-validate consistency between states.json and actions.json."""
        print("STATUS: Cross-validating states.json and actions.json...")
        
        # Check that frame ranges in actions match with states
        states_max_frame = max(
            (s.get('frame_index', -1) for s in self.states_data),
            default=-1
        )
        
        for action in self.actions_data.get('actions', []):
            action_id = action.get('action_id')
            start_frame = action.get('start_frame')
            end_frame = action.get('end_frame')
            
            if start_frame is not None and start_frame > states_max_frame:
                self.warnings.append(
                    f"  Action {action_id}: start_frame ({start_frame}) "
                    f"exceeds max frame index ({states_max_frame})"
                )
            
            if end_frame is not None and end_frame > states_max_frame:
                self.warnings.append(
                    f"  Action {action_id}: end_frame ({end_frame}) "
                    f"exceeds max frame index ({states_max_frame})"
                )
        
        self.info.append(f"✓ Max frame index in states: {states_max_frame}")
        
        print()
    
    def _generate_report(self) -> str:
        """Generate validation report."""
        report = []
        
        report.append("\nINFORMATION:")
        for msg in self.info:
            report.append(f"  {msg}")
        
        if self.warnings:
            report.append("\nWARNINGS:")
            for msg in self.warnings:
                report.append(f"  {msg}")
        
        if self.issues:
            report.append("\nERRORS:")
            for msg in self.issues:
                report.append(f"  {msg}")
        
        if not self.issues and not self.warnings:
            report.append("\n✓ ALL VALIDATIONS PASSED")
        elif not self.issues:
            report.append("\n⚠ Validation passed with warnings")
        else:
            report.append("\n✗ Validation FAILED")
        
        return "\n".join(report)


def main():
    """Run validation on provided output directory."""
    if len(sys.argv) < 2:
        print("Usage: python validate_action_tracking.py <output_dir>")
        print("\nExample:")
        print("  python validate_action_tracking.py /path/to/recording/output")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    validator = ActionTrackingValidator(output_dir)
    is_valid, report = validator.validate()
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
