# alo2glb

Standalone ALO -> GLB converter (no Blender required).

Features
- Parses common ALO mesh/submesh chunks and basic vertex layout.
- Embeds referenced textures into the produced GLB binary chunk.
- Resolves shader references preferentially to a nearby `Shaders` folder and prints absolute paths of referenced files.
- Skips exporting meshes whose name contains "Shadow" (case-insensitive).
- Applies an export node rotation; change or disable this in `build_glb()` inside `alo2glb.py`.

Limitations
- This is a pragmatic converter and does not aim for full compatibility with every ALO variant.
- Shader conversion is simplified: the first texture parameter is mapped to `baseColorTexture` of a PBR material.
- DDS textures are embedded verbatim (no automatic conversion). If you need PNG/JPEG, convert externally and re-run.

Requirements
- Python 3.8+

Usage
1. From the project root run:

```bash
python alo2glb.py "path/to/model.alo" -o "out.glb"
```

2. The script will print absolute paths of referenced textures and shader files it found (or warnings if not found).

Notes for troubleshooting
- If Blender reports missing buffers or index errors, ensure the GLB imports as a single file — this script embeds binary data into the GLB.
- To change rotation applied at export, edit the node creation in `build_glb()` inside [alo2glb.py](alo2glb.py).
- To disable skipping shadow meshes, remove the check that ignores mesh names containing "Shadow" in [alo2glb.py](alo2glb.py).

Development & improvements
- Improve shader->PBR mapping (support multiple params).
- Add skin/skeleton export.
- Convert DDS to PNG automatically when embedding.

Files
- `alo2glb.py`: converter script (run as shown above).

License
- Use as you like; no warranty provided.
