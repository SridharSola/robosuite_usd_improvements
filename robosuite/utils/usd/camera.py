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
"""Camera handling for USD exporter."""

import numpy as np

# TODO: b/288149332 - Remove once USD Python Binding works well with pytype.
# pytype: disable=module-attr
from pxr import Gf, Usd, UsdGeom

import robosuite.utils.usd.utils as utils_module


class USDCamera:
    """Class that handles the cameras in the USD scene"""

    def __init__(self, stage: Usd.Stage, camera_name: str, parent_path: str = "/World"):
        self.stage = stage
        self.camera_name = camera_name
        self.parent_path = parent_path

        # Create camera under proper parent
        xform_path = f"{parent_path}/Camera_Xform_{camera_name}"
        camera_path = f"{xform_path}/Camera_{camera_name}"
        
        # Create or get parent prim if it doesn't exist
        parent_prim = self.stage.GetPrimAtPath(parent_path)
        if not parent_prim:
            parent_prim = UsdGeom.Xform.Define(stage, parent_path)

        self.usd_xform = UsdGeom.Xform.Define(stage, xform_path)
        self.usd_camera = UsdGeom.Camera.Define(stage, camera_path)
        self.usd_prim = stage.GetPrimAtPath(camera_path)

        # defining ops required by update function
        self.transform_op = self.usd_xform.AddTransformOp()

        self.usd_camera.CreateFocalLengthAttr().Set(12)
        self.usd_camera.CreateFocusDistanceAttr().Set(400)

        self.usd_camera.GetHorizontalApertureAttr().Set(17)
        self.usd_camera.GetVerticalApertureAttr().Set(12)

        self.usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(1e-4, 1e6))

    def update(self, cam_pos: np.ndarray, cam_mat: np.ndarray, frame: int):
        """Updates the position and orientation of the camera in the scene."""
        if self.parent_path != "/World":
            # Get parent world transform
            parent_prim = self.stage.GetPrimAtPath(self.parent_path)
            if parent_prim:
                parent_xform = UsdGeom.Xformable(parent_prim)
                # Get transform at current frame
                parent_matrix = parent_xform.GetLocalTransformation(frame)
                
                # Convert to numpy for easier manipulation
                parent_matrix_np = np.array(parent_matrix).reshape(4,4)
                
                # Create camera transform matrix
                cam_transform = utils_module.create_transform_matrix(
                    rotation_matrix=cam_mat, 
                    translation_vector=cam_pos
                ).T
                
                # Calculate relative transform
                relative_transform = np.linalg.inv(parent_matrix_np) @ cam_transform
                
                # Convert back to USD matrix
                self.transform_op.Set(
                    Gf.Matrix4d(relative_transform.tolist()), 
                    frame
                )
        else:
            # World camera - use absolute transform
            transformation_mat = utils_module.create_transform_matrix(
                rotation_matrix=cam_mat, 
                translation_vector=cam_pos
            ).T
            self.transform_op.Set(Gf.Matrix4d(transformation_mat.tolist()), frame)
