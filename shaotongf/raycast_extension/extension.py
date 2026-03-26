import asyncio
from itertools import count
from json import tool
import os

import omni.ext
import omni.ui as ui
import omni.usd

from pxr import Usd, UsdGeom, UsdShade, Sdf


class MyExtension(omni.ext.IExt):
    def on_startup(self, _ext_id):
        print("[shaotongf.raycast_extension] Extension startup")

        from .selection_tools.selection_processor import MeshOnlySelectionFilter
        self.selector = MeshOnlySelectionFilter()


        self._window = ui.Window("Raycast Extension", width=300, height=200)
        with self._window.frame:
            with ui.VStack():
                self._label = ui.Label("Drag in viewport, release to commit selection")

                def on_start_batch():
                    async def _job():
                        ok = await self.selector.run_once()
                        print("done =", ok)
                    asyncio.ensure_future(_job())


                def on_remove_geom_subsets():
                        self.selector.remove_all_box_select_subsets()
                        

                ui.Button("box_select_warp", clicked_fn=on_start_batch)
                ui.Button("remove_geom_subsets", clicked_fn=on_remove_geom_subsets)
                

    def on_shutdown(self):
        print("[shaotongf.raycast_extension] Extension shutdown")