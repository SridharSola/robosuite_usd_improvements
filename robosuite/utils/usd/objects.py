# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Objects module for USD exporter."""

import abc
import collections
from typing import Any, Dict, Optional

import mujoco
import numpy as np

# TODO: b/288149332 - Remove once USD Python Binding works well with pytype.
# pytype: disable=module-attr
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import robosuite.utils.usd.shapes as shapes_module
import robosuite.utils.usd.utils as utils_module


class USDObject(abc.ABC):
    """Abstract interface for all USD objects including meshes and primitives.

    Subclasses must implement:

    * `_get_uv_geometry(self)`: gets the nessecary UV information to
       wrap a texture around an object in USD. Each subclass implements
       their own method to getting UV information as different objects
       are contructed in different ways.

    * `_get_mesh_geometry(self)`: gets the mesh geometry of an object
       in the scene.
    """

    @staticmethod
    def factory(stage, model, geom, original_geom, obj_name, rgba=np.array([1, 1, 1, 1]), texture_file=None):
        """Factory method to create appropriate USD object based on geom type."""
        if geom.objtype == mujoco.mjtObj.mjOBJ_MESH:
            return USDMesh(stage, model, geom, obj_name, rgba, texture_file)
        elif geom.objtype == mujoco.mjtObj.mjOBJ_TENDON:
            mesh_config = shapes_module.get_tendon_mesh_config(geom.size)
            return USDTendon(mesh_config, stage, model, geom, obj_name, rgba, texture_file)
        else:
            return USDPrimitiveMesh(stage, model, geom, obj_name, rgba, texture_file)

    def __init__(
        self,
        stage: Usd.Stage,
        model: mujoco.MjModel,
        geom: mujoco.MjvGeom,
        obj_name: str,
        rgba: np.ndarray = np.array([1, 1, 1, 1]),
        texture_file: Optional[str] = None,
    ):
        self.stage = stage
        self.model = model
        self.geom = geom
        # Ensure obj_name is a valid string for a path
        if isinstance(obj_name, np.ndarray):
            obj_name = f"array_{hash(obj_name.tobytes())}"
        self.obj_name = str(obj_name).replace(" ", "_").replace("[", "").replace("]", "").replace(".", "_")
        self.rgba = rgba
        self.texture_file = texture_file

        # Create path string first, then convert to Sdf.Path
        xform_path_str = f"/World/Mesh_Xform_{self.obj_name}"
        self.xform_path = Sdf.Path(xform_path_str)
        
        # Use stage for Define, not model
        self.usd_xform = UsdGeom.Xform.Define(self.stage, self.xform_path)
        
        # Single transform operation
        self.transform_op = self.usd_xform.AddTransformOp()
        self.scale_op = self.usd_xform.AddScaleOp()

        self.last_visible_frame = -2

        # Get mesh name if it exists
        mesh_name = None
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom.objid]
            if mesh_id >= 0:
                mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)

        # Check if this is a visual geom based on various conditions
        self.is_visual_geom = (
            # Check geom name for _vis/visual
            "_vis" in obj_name.lower() or 
            "visual" in obj_name.lower() or
            # Check mesh name for _vis/visual and ensure it's not a collision mesh
            (mesh_name and 
             ("_vis" in mesh_name.lower() or "visual" in mesh_name.lower()) and
             not any(x in mesh_name.lower() for x in ["_coll", "collision"])) or
            # Special case for household objects that don't follow the _vis pattern
            (mesh_name and mesh_name.endswith("_vis"))
        )
        self.is_visual_geom = self.is_visual_geom #and self.rgba[3] != 0
        if mesh_name and self.is_visual_geom and "counter" in obj_name.lower():
            print(f"\nVisibility check for {obj_name}:")
            print(f"  Mesh name: {mesh_name}")
            print(f"  Is visual: {self.is_visual_geom}")
        # If it's not a visual geom, set initial visibility to invisible
        #if not self.is_visual_geom:
        #print("Here")
        self.update_visibility(self.is_visual_geom)

    @abc.abstractmethod
    def _get_uv_geometry(self):
        """Gets UV information for an object in the scene."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_mesh_geometry(self):
        """Gets structure of an object in the scene."""
        raise NotImplementedError

    def attach_image_material(self, usd_mesh):
        """Attaches a textured material to a USD object."""
        mtl_path = Sdf.Path(f"/World/_materials/Material_{self.obj_name}")
        mtl = UsdShade.Material.Define(self.stage, mtl_path)
        
        if self.texture_file:
            #print(f"\nSetting up texture for {self.obj_name}:")
            #print(f"  Texture file: {self.texture_file}")
            
            # Create shaders
            pbr_shader = UsdShade.Shader.Define(self.stage, mtl_path.AppendPath("Principled_BSDF"))
            texture_shader = UsdShade.Shader.Define(self.stage, mtl_path.AppendPath("diffuseTexture"))
            uvmap_shader = UsdShade.Shader.Define(self.stage, mtl_path.AppendPath("uvmap"))
            
            # Set shader IDs
            pbr_shader.CreateIdAttr("UsdPreviewSurface")
            texture_shader.CreateIdAttr("UsdUVTexture")
            uvmap_shader.CreateIdAttr("UsdPrimvarReader_float2")
            
            # Setup texture
            texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                self.texture_file if not self.texture_file.startswith('@') else self.texture_file.strip('@'))
            texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            
            # Setup UV coordinates
            uvmap_shader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uvmap_shader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uvmap_shader.ConnectableAPI(), "result")
            texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            
            # Connect texture to shader
            pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_shader.ConnectableAPI(), "rgb")
            
            # Set other material properties
            pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(self.rgba[-1]))
            pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(self.geom.shininess)
            pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0 - self.geom.shininess)
            pbr_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
            pbr_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
            
            if self.geom.shininess > 0.8:  # If highly metallic
                pbr_shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))
            
            if self.rgba[-1] < 1.0:  # If there's transparency
                pbr_shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)  # Index of refraction
                pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(self.rgba[-1])
            
            # Connect shader to material
            mtl.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
            
            # Apply material binding API and bind material
            usd_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
            UsdShade.MaterialBindingAPI(usd_mesh).Bind(mtl)

    def attach_solid_material(self, usd_mesh):
        """Attaches a solid colored material to a USD object."""
        mtl_path = Sdf.Path(f"/World/_materials/Material_{self.obj_name}")
        mtl = UsdShade.Material.Define(self.stage, mtl_path)
        
        # Create shader
        pbr_shader = UsdShade.Shader.Define(self.stage, mtl_path.AppendPath("Principled_BSDF"))
        pbr_shader.CreateIdAttr("UsdPreviewSurface")
        
        # Use rgba values for color
        material_color = self.rgba[:3]
        
        # Set material properties
        pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(material_color))
        pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(self.rgba[-1]))
        
        # Add these lines for glass/metallic properties
        if hasattr(self.geom, 'specular'):
            pbr_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
            pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(self.geom.specular)
        
        if hasattr(self.geom, 'reflectance'):
            pbr_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(self.geom.reflectance)
            pbr_shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.1)
        
        # Connect shader to material
        mtl.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
        
        # Bind material to mesh
        UsdShade.MaterialBindingAPI(usd_mesh).Bind(mtl)
        if "cab_1_left_group_right_door_door" in self.obj_name and "visual" in self.obj_name:
            print(f"\nMaterial for {self.obj_name}:")
            #pr

    def _set_refinement_properties(self, usd_prim, scheme="none"):
        usd_prim.GetAttribute("subdivisionScheme").Set(scheme)

    def update(self, pos, mat, visible, frame=None, scale=None):
        """Updates the position and orientation of an object in the scene."""
        # Quantize positions to ensure consistency
        pos = np.round(pos, decimals=6)
        mat = np.round(mat, decimals=6)
        
        # Create transformation matrix with validation
        transformation_mat = utils_module.create_transform_matrix(
            rotation_matrix=mat,
            translation_vector=pos
        ).T
        
        # Set transform with frame tracking
        utils_module.set_attr(
            attr=self.transform_op,
            value=Gf.Matrix4d(transformation_mat.tolist()),
            frame=frame
        )

        # Only update visibility for visual geoms
        if self.is_visual_geom:
            if not frame and not visible:
                self.update_visibility(False, frame)

            if frame is not None:
                if visible and frame - self.last_visible_frame > 1:
                    self.update_visibility(False, max(0, self.last_visible_frame))
                    self.update_visibility(True, frame)

                if visible:
                    self.last_visible_frame = frame

        if scale is not None:
            self.update_scale(scale, frame)

    def update_visibility(self, visible: bool, frame: Optional[int] = None):
        """Updates the visibility of an object in a scene for a given frame."""
        visibility_setting = "inherited" if visible else "invisible"
        utils_module.set_attr(attr=self.usd_xform.GetVisibilityAttr(), value=visibility_setting, frame=frame)

    def update_scale(self, scale: np.ndarray, frame: Optional[int]):
        """Updates the scale of the tendon."""
        utils_module.set_attr(attr=self.scale_op, value=Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])), frame=frame)

    def update_material_color(self, color):
        """Updates the material color for an object."""
        mtl_path = Sdf.Path(f"/World/_materials/Material_{self.obj_name}")
        material = UsdShade.Material.Get(self.stage, mtl_path)
        
        if material:
            shader = UsdShade.Shader.Get(self.stage, mtl_path.AppendPath("Principled_BSDF"))
            if shader:
                diffuse_input = shader.GetInput("diffuseColor")
                if diffuse_input:
                    diffuse_input.Set(tuple(color))


class USDMesh(USDObject):
    """Class that handles predefined meshes in the USD scene."""

    def __init__(
        self,
        stage: Usd.Stage,
        model: mujoco.MjModel,
        geom: mujoco.MjvGeom,
        obj_name: str,
        dataid: int,
        rgba: np.ndarray = np.array([1, 1, 1, 1]),
        texture_file: Optional[str] = None,
    ):
        super().__init__(stage, model, geom, obj_name, rgba, texture_file)

        self.dataid = dataid

        mesh_path = f"{self.xform_path}/Mesh_{obj_name}"
        self.usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        self.usd_prim = stage.GetPrimAtPath(mesh_path)

        # setting mesh structure properties
        mesh_vert, mesh_face, mesh_facenum = self._get_mesh_geometry()
        self.usd_mesh.GetPointsAttr().Set(mesh_vert)
        self.usd_mesh.GetFaceVertexCountsAttr().Set([3 for _ in range(mesh_facenum)])
        self.usd_mesh.GetFaceVertexIndicesAttr().Set(mesh_face)

        if self.texture_file:
            # setting mesh uv properties
            mesh_texcoord, mesh_facetexcoord = self._get_uv_geometry()
            self.texcoords = UsdGeom.PrimvarsAPI(self.usd_mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying,
            )
            self.texcoords.Set(mesh_texcoord)
            self.texcoords.SetIndices(Vt.IntArray(mesh_facetexcoord.tolist()))
            self.attach_image_material(self.usd_mesh)
        else:
            self.attach_solid_material(self.usd_mesh)

    def _get_facetexcoord_ranges(self, nmesh, arr):
        facetexcoords_ranges = [0]
        running_sum = 0
        for i in range(nmesh):
            running_sum += arr[i] * 3
            facetexcoords_ranges.append(running_sum)
        return facetexcoords_ranges

    def _get_uv_geometry(self):
        mesh_texcoord_adr_from = self.model.mesh_texcoordadr[self.dataid]
        mesh_texcoord_adr_to = (
            self.model.mesh_texcoordadr[self.dataid + 1]
            if self.dataid < self.model.nmesh - 1
            else len(self.model.mesh_texcoord)
        )
        mesh_texcoord = self.model.mesh_texcoord[mesh_texcoord_adr_from:mesh_texcoord_adr_to]

        mesh_facetexcoord_ranges = self._get_facetexcoord_ranges(self.model.nmesh, self.model.mesh_facenum)

        mesh_facetexcoord = self.model.mesh_facetexcoord.flatten()
        mesh_facetexcoord = mesh_facetexcoord[
            mesh_facetexcoord_ranges[self.dataid] : mesh_facetexcoord_ranges[self.dataid + 1]
        ]

        mesh_facetexcoord[mesh_facetexcoord == len(mesh_texcoord)] = 0
        return mesh_texcoord, mesh_facetexcoord

    def _get_mesh_geometry(self):
        mesh_vert_adr_from = self.model.mesh_vertadr[self.dataid]
        mesh_vert_adr_to = (
            self.model.mesh_vertadr[self.dataid + 1]
            if self.dataid < self.model.nmesh - 1
            else len(self.model.mesh_vert)
        )
        mesh_vert = self.model.mesh_vert[mesh_vert_adr_from:mesh_vert_adr_to]

        mesh_face_adr_from = self.model.mesh_faceadr[self.dataid]
        mesh_face_adr_to = (
            self.model.mesh_faceadr[self.dataid + 1]
            if self.dataid < self.model.nmesh - 1
            else len(self.model.mesh_face)
        )
        mesh_face = self.model.mesh_face[mesh_face_adr_from:mesh_face_adr_to]
        mesh_facenum = self.model.mesh_facenum[self.dataid]
        return mesh_vert, mesh_face, mesh_facenum


class USDPrimitiveMesh(USDObject):
    """Class to handle primitive shapes in the USD scene."""

    def __init__(
        self,
        mesh_config: Dict[Any, Any],
        stage: Usd.Stage,
        model: mujoco.MjModel,
        geom: mujoco.MjvGeom,
        obj_name: str,
        rgba: np.ndarray = np.array([1, 1, 1, 1]),
        texture_file: Optional[str] = None,
    ):
        super().__init__(stage, model, geom, obj_name, rgba, texture_file)

        self.mesh_config = mesh_config
        self.prim_mesh = self.generate_primitive_mesh()

        mesh_path = f"{self.xform_path}/Mesh_{obj_name}"
        self.usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        self.usd_prim = stage.GetPrimAtPath(mesh_path)

        mesh_vert, mesh_face, mesh_facenum = self._get_mesh_geometry()
        self.usd_mesh.GetPointsAttr().Set(mesh_vert)
        self.usd_mesh.GetFaceVertexCountsAttr().Set([3 for _ in range(mesh_facenum)])
        self.usd_mesh.GetFaceVertexIndicesAttr().Set(mesh_face)
        self._set_refinement_properties(self.usd_prim)

        if self.texture_file:
            # setting mesh uv properties
            mesh_texcoord, _ = self._get_uv_geometry()
            self.texcoords = UsdGeom.PrimvarsAPI(self.usd_mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying,
            )

            self.texcoords.Set(mesh_texcoord)
            self.texcoords.SetIndices(Vt.IntArray(list(range(mesh_facenum * 3))))

            self.attach_image_material(self.usd_mesh)
        else:
            self.attach_solid_material(self.usd_mesh)

    def generate_primitive_mesh(self):
        """Generates the mesh for the primitive USD object."""
        tex_type = self.model.tex_type[self.geom.texid] if self.texture_file else None
        _, prim_mesh = shapes_module.mesh_factory(self.mesh_config, tex_type)
        prim_mesh.translate(-prim_mesh.get_center())
        return prim_mesh

    def _get_uv_geometry(self):
        assert self.prim_mesh and self.prim_mesh.triangle_uvs is not None

        mesh_texcoord = np.array(self.prim_mesh.triangle_uvs)
        mesh_facetexcoord = np.asarray(self.prim_mesh.triangles)
        tex_type = self.model.tex_type[self.geom.texid]

        if tex_type == mujoco.mjtTexture.mjTEXTURE_2D:
            s_scale, t_scale = self.geom.texrepeat

            if self.geom.texuniform:
                if self.geom.size[0] > 0:
                    s_scale *= self.geom.size[0]
                if self.geom.size[1] > 0:
                    t_scale *= self.geom.size[1]

            s_size, t_size = self.geom.size[:2]
            if self.geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                s_size = s_size if s_size > 0 else 1
                t_size = t_size if t_size > 0 else 1

            if self.geom.texuniform:
                mesh_texcoord[:, 0] *= s_scale / (s_size * 2)
                mesh_texcoord[:, 1] *= t_scale / (t_size * 2)

        return mesh_texcoord, mesh_facetexcoord.flatten()

    def _get_mesh_geometry(self):
        assert self.prim_mesh

        # get mesh geometry
        mesh_vert = np.asarray(self.prim_mesh.vertices)
        mesh_face = np.asarray(self.prim_mesh.triangles)

        return mesh_vert, mesh_face, len(mesh_face)


class USDTendon(USDObject):
    """Class to handle tendons in the USD scene."""

    def __init__(
        self,
        mesh_config: Dict[Any, Any],
        stage: Usd.Stage,
        model: mujoco.MjModel,
        geom: mujoco.MjvGeom,
        obj_name: str,
        rgba: np.ndarray = np.array([1, 1, 1, 1]),
        texture_file: Optional[str] = None,
    ):
        super().__init__(stage, model, geom, obj_name, rgba, texture_file)

        self.mesh_config = mesh_config
        self.tendon_parts = self.generate_primitive_mesh()
        self.usd_refs = collections.defaultdict(dict)

        for name, _ in self.tendon_parts.items():
            part_xform_path = f"{self.xform_path}/Mesh_Xform_{name}"
            mesh_path = f"{part_xform_path}/Mesh_{obj_name}"
            usd_xform = UsdGeom.Xform.Define(stage, part_xform_path)
            self.usd_refs[name]["usd_xform"] = usd_xform
            self.usd_refs[name]["usd_mesh"] = UsdGeom.Mesh.Define(stage, mesh_path)
            self.usd_refs[name]["usd_prim"] = stage.GetPrimAtPath(mesh_path)
            # adding ops for each of the part xforms
            self.usd_refs[name]["translate_op"] = usd_xform.AddTranslateOp()
            self.usd_refs[name]["scale_op"] = usd_xform.AddScaleOp()

        # setting mesh geometry properties for each of the parts in the tendon
        part_geometries = self._get_mesh_geometry()
        part_geometry = None
        for name, part_geometry in part_geometries.items():
            self.usd_refs[name]["usd_mesh"].GetPointsAttr().Set(part_geometry["mesh_vert"])
            self.usd_refs[name]["usd_mesh"].GetFaceVertexCountsAttr().Set(
                [3 for _ in range(part_geometry["mesh_facenum"])]
            )
            self.usd_refs[name]["usd_mesh"].GetFaceVertexIndicesAttr().Set(part_geometry["mesh_face"])

        if self.texture_file:
            # setting uv properties for each of the parts in the tendon
            part_uv_geometries = self._get_uv_geometry()
            for name, part_uv_geometry in part_uv_geometries.items():
                self.texcoords = UsdGeom.PrimvarsAPI(self.usd_refs[name]["usd_mesh"]).CreatePrimvar(
                    "st",
                    Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.faceVarying,
                )
                self.texcoords.Set(part_uv_geometry["mesh_texcoord"])
                self.texcoords.SetIndices(Vt.IntArray(list(range(part_geometry["mesh_facenum"] * 3))))
                for _, ref in self.usd_refs.items():
                    self._set_refinement_properties(ref["usd_prim"])
                    self.attach_image_material(ref["usd_mesh"])
        else:
            for _, ref in self.usd_refs.items():
                self._set_refinement_properties(ref["usd_prim"])
                self.attach_solid_material(ref["usd_mesh"])

    def generate_primitive_mesh(self):
        """Generates the tendon mesh using primitives."""
        mesh_parts = {}
        tex_type = self.model.tex_type[self.geom.texid] if self.texture_file else None
        for part_config in self.mesh_config:
            mesh_name, prim_mesh = shapes_module.mesh_factory(part_config, tex_type)
            prim_mesh.translate(-prim_mesh.get_center())
            mesh_parts[mesh_name] = prim_mesh
        return mesh_parts

    def _get_uv_geometry(self):
        part_uv_geometries = collections.defaultdict(dict)
        for name, mesh in self.tendon_parts.items():
            assert mesh.triangle_uvs is not None
            mesh_texcoord = np.array(mesh.triangle_uvs)
            mesh_facetexcoord = np.asarray(mesh.triangles)
            part_uv_geometries[name] = {
                "mesh_texcoord": mesh_texcoord,
                "mesh_facetexcoord": mesh_facetexcoord,
            }
        return part_uv_geometries

    def _get_mesh_geometry(self):
        part_geometries = collections.defaultdict(dict)
        for name, mesh in self.tendon_parts.items():
            # get mesh geometry
            mesh_vert = np.asarray(mesh.vertices)
            mesh_face = np.asarray(mesh.triangles)
            part_geometries[name] = {
                "mesh_vert": mesh_vert,
                "mesh_face": mesh_face,
                "mesh_facenum": len(mesh_face),
            }
        return part_geometries

    def update(
        self,
        pos: np.ndarray,
        mat: np.ndarray,
        visible: bool,
        frame: Optional[int] = None,
        scale: Optional[np.ndarray] = None,
    ):
        """Updates the position and orientation of an object in the scene."""
        super().update(pos, mat, visible, frame, scale)
        for name in self.tendon_parts:
            if "left" in name:
                translate = [0, 0, -scale[2] - (scale[0] / 2)]
                utils_module.set_attr(
                    attr=self.usd_refs[name]["translate_op"],
                    value=Gf.Vec3f(translate), 
                    frame=frame
                )
            elif "right" in name:
                translate = [0, 0, scale[2] + (scale[0] / 2)]
                utils_module.set_attr(
                    attr=self.usd_refs[name]["translate_op"],
                    value=Gf.Vec3f(translate), 
                    frame=frame
                )

    def update_scale(self, scale: np.ndarray, frame: Optional[int]):
        """Updates the scale of the tendon."""
        for name in self.tendon_parts:
            if "cylinder" in name:
                utils_module.set_attr(attr=self.usd_refs[name]["scale_op"], value=Gf.Vec3f(scale.tolist()), frame=frame)
            else:
                hemisphere_scale = scale.tolist()
                hemisphere_scale[2] = hemisphere_scale[0]
                utils_module.set_attr(
                    attr=self.usd_refs[name]["scale_op"], value=Gf.Vec3f(hemisphere_scale), frame=frame
                )