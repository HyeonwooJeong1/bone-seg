# label_map.json Schema

Dataset-specific label mapping configuration. Each dataset directory contains one `label_map.json` file that maps original segmentation label values to unified taxonomy identifiers.

## Format

```jsonc
{
  "dataset": "string",           // Dataset identifier (e.g., "verse20", "ctpelvic1k")
  "source_format": "string",     // Format: nifti_seg | dicom_seg | nrrd | ply
  "provenance_license": "string",// License: ccby | ccbync | public | tcia
  "map": {                       // Original label value (string) → unified label name
    "1": "C1", "2": "C2", ... "25": "L5"
  },
  "grouped": {                   // Granularity limitations (partial/coarse labels)
    "key": {
      "source_value": <int>,     // Original voxel value in source segmentation
      "covers": ["L1","L2","L3"] // Unified labels this grouped region contains
    }
  },
  "present_labels": ["C1", "C2", ...],  // Unified labels actually annotated in this dataset
  "notes": "optional description"
}
```

## Fields

- **`dataset`**: Identifier for the source dataset (lowercase, underscore-separated).
- **`source_format`**: Original segmentation file format before conversion.
- **`provenance_license`**: License classification for attribution and reuse (affects training mix).
- **`map`**: Dictionary mapping original integer labels (as strings) to unified taxonomy label names. Missing/unmapped values default to background (0).
- **`grouped`**: For datasets where individual labels are grouped into a single source value (e.g., lumbar vertebrae as one blob). Grouped regions are remapped to `IGNORE_LABEL` (255) to exclude from per-label loss during training.
- **`present_labels`**: Set of unified labels annotated in this dataset. Enables partial-label learning: loss computed only over present channels, others treated as ignore. Subset of `map` values and grouped covers.
- **`notes`**: Optional metadata (data issues, preprocessing notes, etc.).

## Validation Rules

1. All values in `map` must exist in `ai_bone.taxonomy_v1.NAME_TO_ID`.
2. All labels in grouped `covers` arrays must exist in taxonomy.
3. `present_labels` ⊆ (values of `map`) ∪ (all `covers` from `grouped`).
4. Original source values used in `grouped` are remapped to `IGNORE_LABEL` during `LabelMap.remap_array()`.
