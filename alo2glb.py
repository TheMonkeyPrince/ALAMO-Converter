#!/usr/bin/env python3
"""alo2glb.py

Standalone ALO -> GLB converter (no Blender).

Usage: python alo2glb.py input.alo -o output.glb

Notes:
- Attempts to parse common ALO chunks (meshes, sub-meshes, vertex format v2).
- Embeds referenced textures/shaders into the GLB binary chunk.
- Prints absolute paths of referenced files (textures and shaders).
- Converts simple shader parameters: maps first texture parameter to baseColorTexture.

This is a pragmatic converter and may not support every ALO variant.
"""

import os
import sys
import struct
import json
import argparse
from pathlib import Path


def read_u32(f):
    return struct.unpack('<I', f.read(4))[0]


def read_i32(f):
    return struct.unpack('<i', f.read(4))[0]


def read_u16(f):
    return struct.unpack('<H', f.read(2))[0]


def read_f32(f):
    return struct.unpack('<f', f.read(4))[0]


def read_string_from_bytes(b):
    try:
        i = b.index(b'\x00')
        return b[:i].decode('utf-8', errors='replace')
    except ValueError:
        return b.decode('utf-8', errors='replace')


def align4(n):
    return (n + 3) & ~3


class ALOParser:
    def __init__(self, path):
        self.path = Path(path)
        self.dir = self.path.parent
        self.textures = []
        self.shaders = []
        self.meshes = []

    def parse(self):
        with open(self.path, 'rb') as f:
            filesize = os.fstat(f.fileno()).st_size
            self._read_chunks(f, 0, filesize)

    def _read_chunks(self, f, start, end):
        f.seek(start)
        while f.tell() < end:
            pos = f.tell()
            if end - pos < 8:
                break
            ctype = read_u32(f)
            csize = read_u32(f)
            has_children = (csize & 0x80000000) != 0
            body_size = csize & 0x7FFFFFFF
            body_start = f.tell()
            body_end = body_start + body_size
            if ctype == 0x400:  # Mesh container
                mesh = self._parse_mesh(f, body_start, body_end)
                if mesh:
                    self.meshes.append(mesh)
            else:
                if has_children:
                    self._read_chunks(f, body_start, body_end)
                # else skip data
            f.seek(body_end)

    def _parse_mesh(self, f, start, end):
        f.seek(start)
        mesh = {'name': None, 'submeshes': []}
        while f.tell() < end:
            cpos = f.tell()
            ctype = read_u32(f)
            csize = read_u32(f)
            has_children = (csize & 0x80000000) != 0
            body_size = csize & 0x7FFFFFFF
            bstart = f.tell()
            bend = bstart + body_size
            if ctype == 0x401:  # Mesh name
                data = f.read(body_size)
                mesh['name'] = read_string_from_bytes(data)
            elif ctype == 0x402:  # Mesh info
                # nMaterials (uint32), bounding box float3[2], two flags, then padding
                fpos = f.tell()
                try:
                    nMaterials = read_u32(f)
                    mesh['nMaterials'] = nMaterials
                except Exception:
                    pass
            elif ctype == 0x10100:  # Sub-mesh material information
                # parse children (shader name and params)
                mat = self._parse_material_chunk(f, bstart, bend)
                mesh['submeshes'].append({'material': mat})
            elif ctype == 0x10000:  # Sub-mesh data
                geom = self._parse_submesh_data(f, bstart, bend)
                # match with last submesh if present
                if mesh['submeshes']:
                    mesh['submeshes'][-1].update({'geometry': geom})
                else:
                    mesh['submeshes'].append({'geometry': geom})
            else:
                if has_children:
                    self._read_chunks(f, bstart, bend)
                # else ignore
            f.seek(bend)
        return mesh

    def _parse_material_chunk(self, f, start, end):
        f.seek(start)
        mat = {'shader': None, 'params': {}}
        while f.tell() < end:
            # child chunk header
            ctype = read_u32(f)
            csize = read_u32(f)
            body_size = csize & 0x7FFFFFFF
            bstart = f.tell()
            data = f.read(body_size)
            if ctype == 0x10101:  # Shader filename
                name = read_string_from_bytes(data) + "o" # add 'o' suffix to match actual shader files
                mat['shader'] = name
                if name:
                    self.shaders.append(self._resolve_ref(name, search_folder='..\\Shaders'))
            else:
                # try parse as sequence of mini-chunks
                off = 0
                last_name = None
                while off + 2 <= len(data):
                    mtype = data[off]
                    msize = data[off+1]
                    mdata = data[off+2:off+2+msize]
                    off += 2 + msize
                    if mtype == 0x01:
                        pname = read_string_from_bytes(mdata)
                        last_name = pname
                    elif mtype == 0x02:
                        # value; could be string or binary
                        try:
                            val = read_string_from_bytes(mdata)
                        except Exception:
                            val = None
                        if last_name:
                            mat['params'][last_name] = val
                            # if appears to be a texture filename
                            if isinstance(val, str) and val.strip():
                                self.textures.append(self._resolve_ref(val, search_folder='..\\Textures'))
            # continue
        return mat

    def _resolve_ref(self, filename, search_folder):
        # Resolve a referenced file relative to the ALO file.
        # If absolute already, return as-is.
        if not filename:
            return None
        # try search_folder relative
        cand2 = (self.dir / search_folder / filename).resolve()
        if cand2.exists():
            return str(cand2)
        print("Warning: Referenced file not found:", filename)
        return None

    def _parse_submesh_data(self, f, start, end):
        f.seek(start)
        geom = {'nVertices': 0, 'nPrimitives': 0, 'indices': None, 'vertices': None}
        while f.tell() < end:
            ctype = read_u32(f)
            csize = read_u32(f)
            body_size = csize & 0x7FFFFFFF
            bstart = f.tell()
            if ctype == 0x10001:
                nV = read_u32(f)
                nP = read_u32(f)
                geom['nVertices'] = nV
                geom['nPrimitives'] = nP
                # skip padding
                f.seek(bstart + body_size)
            elif ctype == 0x10002:
                data = f.read(body_size)
                fmt = read_string_from_bytes(data)
                geom['vertexFormat'] = fmt
            elif ctype == 0x10004:
                # indices: word[3 * nPrimitives]
                count = body_size // 2
                fbuf = f.read(body_size)
                inds = list(struct.unpack('<' + 'H'*count, fbuf))
                geom['indices'] = inds
            elif ctype == 0x10007:
                # vertex buffer v2
                n = geom.get('nVertices', 0)
                if n == 0:
                    # can't parse without n; read as many as possible
                    n = body_size // 144
                verts = []
                for i in range(n):
                    # position 3f
                    px = read_f32(f); py = read_f32(f); pz = read_f32(f)
                    nx = read_f32(f); ny = read_f32(f); nz = read_f32(f)
                    # texCoords: float2[4]
                    u0 = read_f32(f); v0 = read_f32(f)
                    u1 = read_f32(f); v1 = read_f32(f)
                    u2 = read_f32(f); v2 = read_f32(f)
                    u3 = read_f32(f); v3 = read_f32(f)
                    tx = read_f32(f); ty = read_f32(f); tz = read_f32(f)
                    bx = read_f32(f); by = read_f32(f); bz = read_f32(f)
                    cr = read_f32(f); cg = read_f32(f); cb = read_f32(f); ca = read_f32(f)
                    # unused 4 floats
                    f.read(4*4)
                    # boneIndices dword[4]
                    bidx = struct.unpack('<IIII', f.read(16))
                    bw = struct.unpack('<ffff', f.read(16))
                    verts.append({'pos': (px,py,pz), 'norm': (nx,ny,nz), 'uv': (u0,v0), 'tangent': (tx,ty,tz), 'color': (cr,cg,cb,ca)})
                geom['vertices'] = verts
            else:
                # skip
                f.seek(bstart + body_size)
        return geom


def build_glb(parser, out_path):
    # Build glTF structure and BIN buffer
    json_doc = {
        'asset': {'version': '2.0', 'generator': 'alo2glb.py'},
        'scenes': [{'nodes': []}],
        'nodes': [],
        'meshes': [],
        'buffers': [],
        'bufferViews': [],
        'accessors': [],
        'materials': [],
        'images': [],
        'textures': [],
    }
    bin_blob = bytearray()

    def add_bufferview(data, target=None):
        off = align4(len(bin_blob))
        while len(bin_blob) < off:
            bin_blob.append(0)
        start = len(bin_blob)
        bin_blob.extend(data)
        bv = {'buffer': 0, 'byteOffset': start, 'byteLength': len(data)}
        if target:
            bv['target'] = target
        idx = len(json_doc['bufferViews'])
        json_doc['bufferViews'].append(bv)
        return idx

    def add_accessor(bufferView, count, compType, typeStr, normalized=False, minv=None, maxv=None):
        acc = {'bufferView': bufferView, 'componentType': compType, 'count': count, 'type': typeStr}
        if normalized:
            acc['normalized'] = True
        if minv is not None:
            acc['min'] = minv
        if maxv is not None:
            acc['max'] = maxv
        idx = len(json_doc['accessors'])
        json_doc['accessors'].append(acc)
        return idx

    # Iterate meshes
    for mi, mesh in enumerate(parser.meshes):
        # Skip shadow meshes by name
        mname = (mesh.get('name') or '')
        if 'shadow' in mname.lower():
            continue
        for si, sm in enumerate(mesh.get('submeshes', [])):
            geom = sm.get('geometry')
            if not geom or not geom.get('vertices'):
                continue
            verts = geom['vertices']
            n = len(verts)
            # positions
            pos_bytes = b''.join(struct.pack('<fff', *v['pos']) for v in verts)
            pos_bv = add_bufferview(pos_bytes, target=34962)
            pos_min = [min(v['pos'][i] for v in verts) for i in range(3)]
            pos_max = [max(v['pos'][i] for v in verts) for i in range(3)]
            pos_acc = add_accessor(pos_bv, n, 5126, 'VEC3', minv=pos_min, maxv=pos_max)
            # normals
            nor_bytes = b''.join(struct.pack('<fff', *v['norm']) for v in verts)
            nor_bv = add_bufferview(nor_bytes, target=34962)
            nor_acc = add_accessor(nor_bv, n, 5126, 'VEC3')
            # texcoords
            tex_bytes = b''.join(struct.pack('<ff', *v['uv']) for v in verts)
            tex_bv = add_bufferview(tex_bytes, target=34962)
            tex_acc = add_accessor(tex_bv, n, 5126, 'VEC2')
            # indices
            inds = geom.get('indices', [])
            if not inds:
                continue
            # glTF prefers triangles; ALO stores triangle indices
            idx_bytes = b''.join(struct.pack('<H', i) for i in inds)
            idx_bv = add_bufferview(idx_bytes, target=34963)
            idx_acc = add_accessor(idx_bv, len(inds), 5123, 'SCALAR')

            # material: map first texture param to baseColor
            mat_idx = None
            mat = sm.get('material') or {}
            p = None
            for k, v in (mat.get('params') or {}).items():
                if isinstance(v, str) and v.strip():
                    p = v; break
            if p:
                img_path = parser._resolve_ref(p, '..\\Textures')
                # try load
                try:
                    with open(img_path, 'rb') as imf:
                        imdata = imf.read()
                except Exception:
                    imdata = b''
                if imdata:
                    mime = 'image/png'
                    ext = Path(img_path).suffix.lower()
                    if ext in ('.jpg', '.jpeg'):
                        mime = 'image/jpeg'
                    elif ext == '.dds':
                        mime = 'image/vnd-ms.dds'
                    img_bv = add_bufferview(imdata)
                    img_idx = len(json_doc['images'])
                    json_doc['images'].append({'bufferView': img_bv, 'mimeType': mime, 'name': Path(img_path).stem})
                    tex_idx = len(json_doc['textures'])
                    json_doc['textures'].append({'source': img_idx})
                    mat_idx = len(json_doc['materials'])
                    json_doc['materials'].append({'pbrMetallicRoughness': {'baseColorTexture': {'index': tex_idx}, 'metallicFactor': 0.0}})
            if mat_idx is None:
                # fallback default material
                mat_idx = len(json_doc['materials'])
                json_doc['materials'].append({'pbrMetallicRoughness': {'baseColorFactor': [1.0,1.0,1.0,1.0], 'metallicFactor': 0.0}})

            # build mesh primitive
            primitive = {'attributes': {'POSITION': pos_acc, 'NORMAL': nor_acc, 'TEXCOORD_0': tex_acc}, 'indices': idx_acc, 'material': mat_idx}
            mesh_idx = len(json_doc['meshes'])
            json_doc['meshes'].append({'primitives': [primitive], 'name': mesh.get('name') or f'mesh_{mi}_{si}'})
            # node
            node_idx = len(json_doc['nodes'])
            # Apply rotation: X 90°, then Z 180°, then Y 180° -> combined quaternion [-0.70710678, 0.0, 0.0, 0.70710678]
            json_doc['nodes'].append({'mesh': mesh_idx, 'name': mesh.get('name') or f'node_{mi}_{si}', 'rotation': [-0.70710678, 0.0, 0.0, 0.70710678]})
            json_doc['scenes'][0]['nodes'].append(node_idx)

    # finalize buffer
    if len(bin_blob) == 0:
        bin_blob = bytearray()
    # add buffers entry required by glTF consumers (GLB uses binary chunk)
    json_doc['buffers'] = [{ 'byteLength': len(bin_blob) }]
    json_bytes = json.dumps(json_doc, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    json_padded = json_bytes + b' ' * (align4(len(json_bytes)) - len(json_bytes))

    # GLB header
    glb = bytearray()
    glb.extend(b'glTF')
    glb.extend(struct.pack('<I', 2))
    # placeholder total length
    glb.extend(struct.pack('<I', 0))
    # JSON chunk
    glb.extend(struct.pack('<I', len(json_padded)))
    glb.extend(struct.pack('<I', 0x4E4F534A))
    glb.extend(json_padded)
    # BIN chunk
    bin_padded = bytes(bin_blob) + b'\x00' * (align4(len(bin_blob)) - len(bin_blob))
    glb.extend(struct.pack('<I', len(bin_padded)))
    glb.extend(struct.pack('<I', 0x004E4942))
    glb.extend(bin_padded)
    # write total length
    total_len = len(glb)
    struct.pack_into('<I', glb, 8, total_len)

    with open(out_path, 'wb') as outf:
        outf.write(glb)


def main():
    ap = argparse.ArgumentParser(description='Convert ALO -> GLB (standalone)')
    ap.add_argument('alo', help='input .alo file')
    ap.add_argument('-o', '--out', help='output .glb file', default=None)
    args = ap.parse_args()
    alo = Path(args.alo)
    if not alo.exists():
        print('Input ALO not found:', alo)
        sys.exit(1)
    out = args.out or (alo.with_suffix('.glb'))
    parser = ALOParser(alo)
    parser.parse()

    # Print absolute paths of referenced files
    refs = set()
    for t in parser.textures:
        if t:
            refs.add(os.path.abspath(t))
    for s in parser.shaders:
        if s:
            refs.add(os.path.abspath(s))
    print('Referenced files:')
    for r in sorted(refs):
        print(r)

    build_glb(parser, str(out))
    print('Wrote GLB to', out)


if __name__ == '__main__':
    main()
