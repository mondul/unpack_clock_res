Changes:
- Implemented proper deduplication - Images are now written once per unique ref ID instead of once per layer
- Simplified file naming - Changed from layer_XXX_chunk_YYY.png to chunk_ZZZ.png where ZZZ is the ref ID
