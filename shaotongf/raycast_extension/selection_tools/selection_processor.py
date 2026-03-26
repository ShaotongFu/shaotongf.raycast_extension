import asyncio
import time
from typing import Optional, Tuple, List, Dict, Any

import omni
import omni.usd
import omni.kit.viewport.utility as vp_util
import omni.ui as ui
from omni.ui import scene as sc

import carb.events

import numpy as np
import warp as wp
wp.init()

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
from omni.kit.viewport.utility import get_active_viewport_window


@wp.kernel
def mark_tris_in_ndc_rect(
    points: wp.array(dtype=wp.vec3),
    faceVertexCounts_prefix: wp.array(dtype=wp.int32),
    faceVertexCounts: wp.array(dtype=wp.int32),
    faceVertexIndices: wp.array(dtype=wp.int32),
    out_mask: wp.array(dtype=wp.int32),
    local_to_world: wp.mat44,
    world_to_ndc: wp.mat44,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
):
    tid = wp.tid()

    i0 = faceVertexCounts_prefix[tid]
    count = faceVertexCounts[tid]

    out_mask[tid] = 0

    for k in range(count):
        p = points[faceVertexIndices[i0 + k]]
        p4 = wp.vec4(p[0], p[1], p[2], 1.0)

        world4 = local_to_world * p4
        clip4 = world_to_ndc * world4

        if wp.abs(clip4[3]) > 1.0e-8:
            x = clip4[0] / clip4[3]
            y = clip4[1] / clip4[3]

            if min_x < x and x < max_x and min_y < y and y < max_y:
                out_mask[tid] = 1
                return


class MeshOnlySelectionFilter:
    def __init__(self):
        self._ctx = omni.usd.get_context()
        self.selected = []
        self.face_ids_by_path = {}
        self.ndc_selector = SimpleBoxSelectNdc()

        self._cached_preview_material = None
        self._cached_hidden_material = None

    async def run_once(self):
        stage = self._ctx.get_stage()
        if stage is None:
            print("No USD stage.")
            return False

        result = await self.ndc_selector.run_once()
        if not result:
            return False

        ndc_rect = result["ndc_rect"]
        paths = result["prim_paths"]

        print("final ndc_rect =", ndc_rect)
        #print("selected prims =", paths)

        self.selected = paths
        return self._identify_indices_from_ndc(stage, ndc_rect, paths)

    def _identify_indices_from_ndc(self, stage, ndc_rect: Tuple[float, float, float, float], paths):
        self.face_ids_by_path = {}

        if not paths:
            print("No selected prims.")
            return False

        min_x, min_y, max_x, max_y = ndc_rect
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api is None:
            print("No active viewport.")
            return False

        t = viewport_api.time
        processed_any = False
        t0 = time.perf_counter()

        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue

            if prim.IsA(UsdGeom.Subset):
                continue

            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)

            pts = mesh.GetPointsAttr().Get(t)
            fvc = mesh.GetFaceVertexCountsAttr().Get(t)
            fvi = mesh.GetFaceVertexIndicesAttr().Get(t)

            if pts is None or fvc is None or fvi is None:
                continue

            pts_np = np.asarray(pts, dtype=np.float32)
            if pts_np.ndim != 2 or pts_np.shape[1] != 3:
                continue

            fvc_np = np.asarray(fvc, dtype=np.int32)
            fvi_np = np.asarray(fvi, dtype=np.int32)

            if fvc_np.size == 0 or fvi_np.size == 0:
                continue

            processed_any = True

            points = wp.array(pts_np, dtype=wp.vec3f, device="cuda:0")
            faceVertexCounts = wp.array(fvc_np, dtype=wp.int32, device="cuda:0")
            faceVertexCounts_prefix = wp.array(
                np.concatenate(([0], np.cumsum(fvc_np)[:-1])),
                dtype=wp.int32,
                device="cuda:0",
            )
            faceVertexIndices = wp.array(fvi_np, dtype=wp.int32, device="cuda:0")
            mask = wp.zeros(faceVertexCounts_prefix.size, dtype=wp.int32, device="cuda:0")

            local_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)
            world_to_ndc = viewport_api.world_to_ndc

            local_to_world_np = self.gfmat4d_to_np(local_to_world).T
            world_to_ndc_np = self.gfmat4d_to_np(world_to_ndc).T

            local_to_world_wp = wp.mat44(*local_to_world_np.reshape(-1).tolist())
            world_to_ndc_wp = wp.mat44(*world_to_ndc_np.reshape(-1).tolist())

            wp.launch(
                mark_tris_in_ndc_rect,
                dim=faceVertexCounts_prefix.size,
                inputs=[
                    points,
                    faceVertexCounts_prefix,
                    faceVertexCounts,
                    faceVertexIndices,
                    mask,
                    local_to_world_wp,
                    world_to_ndc_wp,
                    float(min_x),
                    float(min_y),
                    float(max_x),
                    float(max_y),
                ],
                device="cuda:0",
            )

            hit_faces = np.nonzero(mask.numpy())[0]
            if hit_faces.size > 0:
                self.face_ids_by_path[str(path)] = hit_faces

        t1 = time.perf_counter()
        print(f"GPU part took: {(t1 - t0) * 1000:.3f} ms")

        if not processed_any:
            return False

        mtl = self.get_or_create_hidden_omnisurface_material(
            stage,
            mtl_path="/World/Looks/HiddenOmniSurface",
        )

        for prim_path, face_ids in self.face_ids_by_path.items():
            if face_ids is None or len(face_ids) == 0:
                continue

            subset = self.ensure_geom_subset_for_faces(
                stage,
                mesh_prim_path=prim_path,
                face_indices=face_ids,
                subset_name="BoxSelectSubset",
                family_name="materialBind",
            )
            self.bind_material_to_geom_subset(subset, mtl)

        return True

    def gfmat4d_to_np(self, m: Gf.Matrix4d):
        return np.array([
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ], dtype=np.float32)

    def get_or_create_preview_material(
        self,
        stage: Usd.Stage,
        mtl_path: str = "/World/Looks/BoxSelectPreview",
        color=(1.0, 0.0, 0.0),
        opacity: float = 1.0,
    ):
        if self._cached_preview_material is not None:
            prim = self._cached_preview_material.GetPrim()
            if prim and prim.IsValid():
                return self._cached_preview_material

        mtl_sdf = Sdf.Path(mtl_path)
        mtl = UsdShade.Material.Define(stage, mtl_sdf)
        shader = UsdShade.Shader.Define(stage, mtl_sdf.AppendChild("Shader"))

        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        self._cached_preview_material = mtl
        return mtl

    def get_or_create_hidden_omnisurface_material(
        self,
        stage: Usd.Stage,
        mtl_path: str = "/World/Looks/HiddenOmniSurface",
    ):
        if self._cached_hidden_material is not None:
                prim = self._cached_hidden_material.GetPrim()
                if prim and prim.IsValid():
                    return self._cached_hidden_material

        mtl_sdf = Sdf.Path(mtl_path)
        mtl = UsdShade.Material.Define(stage, mtl_sdf)
        shader = UsdShade.Shader.Define(stage, mtl_sdf.AppendChild("Shader"))

        shader.CreateIdAttr("OmniSurfaceLiteBase")

        shader.CreateInput("diffuse_reflection_weight", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("specular_reflection_weight", Sdf.ValueTypeNames.Float).Set(0.0)

        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("geometry_opacity", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("geometry_opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.0)

        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        self._cached_hidden_material = mtl
        return mtl

    def ensure_geom_subset_for_faces(
        self,
        stage: Usd.Stage,
        mesh_prim_path: str,
        face_indices,
        subset_name: str = "BoxSelectSubset",
        family_name: str = "materialBind",
    ):
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Prim is not a UsdGeom.Mesh: {mesh_prim_path}")

        subset_path = Sdf.Path(mesh_prim_path).AppendChild(subset_name)
        existing = stage.GetPrimAtPath(subset_path)
        if existing and existing.IsValid():
            subset = UsdGeom.Subset(existing)
        else:
            subset = UsdGeom.Subset.Define(stage, subset_path)

        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateIndicesAttr().Set(sorted({int(i) for i in face_indices}))
        subset.CreateFamilyNameAttr().Set(family_name)
        return subset

    def bind_material_to_geom_subset(self, subset: UsdGeom.Subset, material: UsdShade.Material):
        stage = omni.usd.get_context().get_stage()
        material_path = "/World/Looks/OmniSurface"
        material_prim = stage.GetPrimAtPath(material_path)
        
        material = UsdShade.Material(material_prim)
        binding_api = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim())
        binding_api.Bind(material)
        
        # binding_api = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim())
        # binding_api.Bind(material)


    def remove_all_box_select_subsets(
        self,
        stage: Optional[Usd.Stage] = None,
        subset_name: str = "BoxSelectSubset",
    ) -> int:
        if stage is None:
            stage = self._ctx.get_stage()

        if stage is None:
            return 0

        paths_to_remove = []

        for prim in stage.Traverse():
            if not prim or not prim.IsValid():
                continue

            if prim.GetName() != subset_name:
                continue

            if not prim.IsA(UsdGeom.Subset):
                continue

            paths_to_remove.append(prim.GetPath())

        for path in paths_to_remove:
            stage.RemovePrim(path)

        return len(paths_to_remove)



class SimpleBoxSelectNdc:
    def __init__(self):
        self._ctx = omni.usd.get_context()
        self._viewport_window = None
        self._scene_view = None

        self._drag_done_evt: Optional[asyncio.Event] = None
        self._selection_done_evt: Optional[asyncio.Event] = None
        self._stage_sub = None

        self._start: Optional[Tuple[float, float]] = None
        self._current: Optional[Tuple[float, float]] = None
        self._result_rect: Optional[Tuple[float, float, float, float]] = None
        self._selected_paths: List[str] = []

    async def run_once(self) -> Dict[str, Any]:
        self.start()

        self._drag_done_evt = asyncio.Event()
        await self._drag_done_evt.wait()

        self._selection_done_evt = asyncio.Event()
        stream = self._ctx.get_stage_event_stream()
        self._stage_sub = stream.create_subscription_to_pop(
            self._on_stage_event,
            name="box_select_ndc_wait_selection_once",
        )

        await self._selection_done_evt.wait()
        self.stop()

        return {
            "ndc_rect": self._result_rect,
            "prim_paths": self._selected_paths,
        }

    def _make_rect(self, p0, p1):
        return (
            min(float(p0[0]), float(p1[0])),
            min(float(p0[1]), float(p1[1])),
            max(float(p0[0]), float(p1[0])),
            max(float(p0[1]), float(p1[1])),
        )

    def start(self):
        self._viewport_window = get_active_viewport_window()
        if not self._viewport_window:
            raise RuntimeError("No active viewport window found!")

        self._start = None
        self._current = None
        self._result_rect = None
        self._selected_paths = []
        self._scene_view = None

        def on_drag_began(sender):
            try:
                ndc = sender.gesture_payload.mouse
                self._start = (float(ndc[0]), float(ndc[1]))
                self._current = self._start
            except Exception as e:
                print(f"on_drag_began error: {e}")

        def on_drag_changed(sender):
            try:
                if self._start is None:
                    return
                ndc = sender.gesture_payload.mouse
                self._current = (float(ndc[0]), float(ndc[1]))
            except Exception as e:
                print(f"on_drag_changed error: {e}")

        def on_drag_ended(sender):
            try:
                if self._start is None:
                    return
                ndc = sender.gesture_payload.mouse
                self._current = (float(ndc[0]), float(ndc[1]))
                self._result_rect = self._make_rect(self._start, self._current)

                if self._drag_done_evt:
                    self._drag_done_evt.set()
            except Exception as e:
                print(f"on_drag_ended error: {e}")

        with self._viewport_window.get_frame("box_select_ndc"):
            with ui.ZStack():
                self._scene_view = sc.SceneView()

                with self._scene_view.scene:
                    sc.Screen(
                        gestures=[
                            sc.DragGesture(
                                name="box_select_drag",
                                on_began_fn=on_drag_began,
                                on_changed_fn=on_drag_changed,
                                on_ended_fn=on_drag_ended,
                            )
                        ]
                    )

        self._viewport_window.viewport_api.add_scene_view(self._scene_view)
        print("SimpleBoxSelectNdc armed")

    def _on_stage_event(self, event: carb.events.IEvent):
        if event.type != int(omni.usd.StageEventType.SELECTION_CHANGED):
            return

        sel = self._ctx.get_selection()
        self._selected_paths = list(sel.get_selected_prim_paths())

        if self._stage_sub is not None:
            self._stage_sub.unsubscribe()
            self._stage_sub = None

        if self._selection_done_evt is not None:
            self._selection_done_evt.set()

    def stop(self):
        if self._stage_sub is not None:
            self._stage_sub.unsubscribe()
            self._stage_sub = None

        if self._scene_view and self._viewport_window:
            try:
                self._viewport_window.viewport_api.remove_scene_view(self._scene_view)
            except Exception as e:
                print(f"remove_scene_view error: {e}")

        self._scene_view = None
        self._viewport_window = None


class NdcBoxTool(SimpleBoxSelectNdc):
    pass
