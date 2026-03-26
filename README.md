# shaotongf.raycast_extension
This is an Omniverse extension that allows users to select mesh faces by dragging the mouse. When used with an Opacity material, it can automatically hide the selected faces, making it easier to observe the internal structure of the model.

<img width="1535" height="763" alt="bc516564-1f62-4cbe-a3d7-8bd8d341d1de" src="https://github.com/user-attachments/assets/d03313a5-bb9d-4f4e-afe5-80a1976b73ef" />

# Installation
1. Clone the repository.
```bash
git clone https://github.com/ShaotongFu/shaotongf.raycast_extension.git
```

2. Copy the extension to the `exts` path of your Omniverse project. The reference path is:
```bash
<Your_OV_Project_Path>\\_build\\windows-x86_64\\release\\exts
```

3. Add the extension to your app's `dependencies`. The app is defined in a `.kit` file, which is usually stored at:
```bash
<Your_OV_Project_Path>\\_build\\windows-x86_64\\release\\apps
```

```toml
[dependencies]
"shaotongf.raycast_extension" = {}
```

4. Use the `repo.bat launch` command to launch the app. The Raycast_Extension panel will open automatically.
<img width="3840" height="2295" alt="image" src="https://github.com/user-attachments/assets/cbb86541-211c-4215-acf5-be88d2a72a6c" />

# Usage
1. In Omniverse, create an Opacity material via `Create -> Material -> OmniSurface`. Select this material in the Stage, then enable `Opacity` under `Shader -> Geometry` in the Properties panel. Set `Opacity` to `0`. Do not change the default material name.
<img width="3828" height="2198" alt="image" src="https://github.com/user-attachments/assets/3c00936b-8a3a-4645-beaf-6074ee4553cd" />

2. Open a USD model.

3. Click `box_select_warp` at extension panel to start drag selection. Drag the mouse, and the faces inside the selected box region will be hidden automatically. You can hide multiple regions by clicking `box_select_warp` multiple times.

4. Click `remove_geom_subsets` to restore all hidden faces and display them again.

# Limitations
This extension is incompatible with Kit 110.
