# RadiomicViz Viewer — Build Spec

## Goal

Build a browser-based interactive 3D viewer for RadiomicViz extraction results. Must work both locally and over SSH (via VS Code port forwarding). Uses **Niivue.js** for WebGL-based NIfTI rendering served through a lightweight **Flask** server.

## Why browser-based

- Works over SSH with VS Code port forwarding (no X11/OpenGL on cluster)
- Works locally by opening a browser tab
- Niivue.js handles 3D volume rendering, orthogonal slices, colormaps via WebGL on the user's machine
- No Qt/OpenGL dependencies on the server side

## Why Flask

- The Python server's only job is serving NIfTI files and one HTML page
- All interactivity (slicing, colormaps, overlays) happens client-side in Niivue.js + vanilla JS
- Flask is one small dependency — no Bokeh, no Tornado, no Panel widget system
- Easy to understand, debug, and extend

## Architecture

```
User runs: result.view() or `radiomicviz view ...`
    → Flask server starts on localhost:PORT
    → Serves NIfTI files at /data/<filename>
    → Serves viewer HTML page at /
    → Browser loads page, Niivue.js fetches NIfTIs, renders via WebGL
    → UI controls (dropdowns, sliders) are plain HTML/JS talking to Niivue
```

## New files to create

```
src/radiomicviz/
├── viewer/
│   ├── __init__.py          # public API: launch_viewer(), launch_viewer_from_result()
│   ├── app.py               # Flask app: routes for serving NIfTIs + HTML
│   ├── templates/
│   │   └── viewer.html      # HTML page with Niivue.js + controls (Jinja2 template)
│   └── static/              # (optional, for local Niivue bundle if not using CDN)
```

## User-facing API

### Python API (add `.view()` to ExtractionResult in result.py)

```python
result = extract("t1.nii.gz", "mask.nii.gz", preset="mri-habitat", mode="voxelwise")

# Launch viewer in browser
result.view()                          # opens browser, serves on random port
result.view(port=8888)                 # specific port (useful for SSH forwarding)
result.view(features=["original_firstorder_Mean", "original_glcm_Contrast"])  # subset
```

### Also support viewing from files (no ExtractionResult needed)

```python
from radiomicviz.viewer import launch_viewer

# View raw image + mask
launch_viewer(image="t1.nii.gz", mask="mask.nii.gz")

# View image + mask + feature overlay NIfTIs from a previous extraction
launch_viewer(
    image="t1.nii.gz",
    mask="mask.nii.gz",
    overlays=["./nifti_out/original_firstorder_Mean.nii.gz",
              "./nifti_out/original_glcm_Contrast.nii.gz"],
)

# View a 4D feature map NIfTI
launch_viewer(image="t1.nii.gz", feature_4d="maps.nii.gz")
```

### CLI

```bash
# View image + mask
radiomicviz view --image t1.nii.gz --mask mask.nii.gz

# View with feature overlays
radiomicviz view \
    --image t1.nii.gz \
    --mask mask.nii.gz \
    --overlays ./nifti_out/original_firstorder_Mean.nii.gz \
    --overlays ./nifti_out/original_glcm_Contrast.nii.gz

# View 4D feature maps
radiomicviz view --image t1.nii.gz --feature-4d maps.nii.gz

# Specify port (for SSH forwarding)
radiomicviz view --image t1.nii.gz --mask mask.nii.gz --port 8888

# Don't auto-open browser (useful on headless cluster)
radiomicviz view --image t1.nii.gz --mask mask.nii.gz --port 8888 --no-browser
```

## Viewer UI features (in the browser)

All UI controls are plain HTML + vanilla JS calling Niivue's API. No Python widget framework needed.

### Must have
- **Orthogonal slice views**: axial, coronal, sagittal (Niivue does this natively)
- **3D volume rendering**: toggle on/off (Niivue supports this)
- **Mask overlay**: show/hide the ROI mask on top of the image, adjustable opacity
- **Feature map overlay**: dropdown to select which feature map to overlay (populated from /api/overlays endpoint)
- **Colormap selector**: dropdown for overlay colormap (viridis, hot, cool, winter, etc. — Niivue has these built in)
- **Opacity slider**: HTML range input controlling overlay transparency
- **Crosshair**: click to navigate slices, show voxel coordinates + intensity values (Niivue native)

### Nice to have (don't block on these)
- **Feature value tooltip**: hover/click shows the overlay value at that voxel
- **Slice slider**: range input to scrub through slices
- **Screenshot button**: Niivue has a `.saveScene()` method
- **Multi-overlay**: show two feature maps side by side for comparison

## Flask app routes

```
GET /                           → viewer.html (Jinja2 template, passes volume manifest)
GET /data/<filename>            → serves NIfTI file (image, mask, or overlay)
GET /api/volumes                → JSON manifest: {image: "image.nii.gz", mask: "mask.nii.gz", overlays: ["feat1.nii.gz", ...]}
```

That's it. Three routes. The `/api/volumes` endpoint tells the JS which files are available so the dropdown can be populated dynamically.

## How the data flows

### From ExtractionResult.view()

1. `result.view()` is called
2. Create a temp directory (using `tempfile.mkdtemp()`)
3. Symlink or copy NIfTI files into it:
   - The original image: symlink from `result.metadata.image_path`
   - The mask: save from `result.mask_nii` (it's already a nibabel object)
   - Feature maps (voxelwise mode): save each from `result.feature_maps` dict
   - Feature maps (ROI mode): generate choropleth NIfTIs using existing `to_nifti()` logic
4. Create Flask app pointing at that temp directory
5. Start server in a background thread
6. Open browser (unless `--no-browser`)
7. Print "Viewer running at http://localhost:PORT — press Ctrl+C to stop"
8. Block on main thread until Ctrl+C, then clean up temp dir

### From launch_viewer() or CLI

1. User provides file paths directly
2. Flask app serves files from their original locations (no temp dir needed)
3. Same viewer.html page

## Integration points with existing code

### result.py — add `.view()` method

```python
def view(self, port: int = 0, features: list[str] | None = None,
         open_browser: bool = True) -> None:
    """Launch interactive browser viewer for this result."""
    from radiomicviz.viewer import launch_viewer_from_result
    launch_viewer_from_result(self, port=port, features=features,
                              open_browser=open_browser)
```

### cli.py — add `view` command

```python
@cli.command()
@click.option("-i", "--image", required=True, type=click.Path(exists=True))
@click.option("-m", "--mask", default=None, type=click.Path(exists=True))
@click.option("--overlays", multiple=True, type=click.Path(exists=True),
              help="Feature map NIfTI(s) to load as overlays")
@click.option("--feature-4d", default=None, type=click.Path(exists=True),
              help="4D NIfTI with stacked feature maps")
@click.option("--port", type=int, default=0, help="Port (0 = auto)")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser")
def view(image, mask, overlays, feature_4d, port, no_browser):
    """Launch interactive viewer in browser."""
    from radiomicviz.viewer import launch_viewer
    launch_viewer(image=image, mask=mask, overlays=list(overlays),
                  feature_4d=feature_4d, port=port,
                  open_browser=not no_browser)
```

### __init__.py — add to exports

```python
from radiomicviz.viewer import launch_viewer
# add "launch_viewer" to __all__
```

### pyproject.toml — update viewer dependencies

```toml
viewer = [
    "flask>=2.3",
]
```

That's it. nibabel is already a core dependency. Niivue.js is loaded from CDN in the HTML. Flask is the only new dependency.

## Niivue.js integration notes

- Niivue repo: https://github.com/niivue/niivue
- CDN: `https://unpkg.com/@niivue/niivue/dist/niivue.umd.js`
- Niivue needs NIfTI files served over HTTP (it fetches them via URL)
- Niivue is initialized in JS with volume URLs and config

Basic Niivue setup in the template:

```html
<script src="https://unpkg.com/@niivue/niivue/dist/niivue.umd.js"></script>
<canvas id="gl" width="800" height="600"></canvas>
<script>
const nv = new niivue.Niivue({backColor: [0, 0, 0, 1]});
nv.attachToCanvas(document.getElementById("gl"));

// Load volumes from Flask endpoints
nv.loadVolumes([
    {url: "/data/image.nii.gz", colormap: "gray"},
    {url: "/data/mask.nii.gz", colormap: "red", opacity: 0.3},
]);

// To add/swap an overlay later (from dropdown change):
// nv.loadVolumes([...base, {url: "/data/overlay_name.nii.gz", colormap: "viridis", opacity: 0.5}]);
</script>
```

Key Niivue API methods you'll need:
- `nv.loadVolumes(volumeList)` — load/reload volumes
- `nv.setOpacity(volumeIndex, opacity)` — change overlay opacity
- `nv.setColormap(volumeIndex, colormapName)` — change colormap
- `nv.setSliceType(nv.sliceTypeMultiplanar)` — orthogonal views
- `nv.setSliceType(nv.sliceTypeRender)` — 3D rendering
- `nv.saveScene("screenshot.png")` — save screenshot
- Colormaps available: "gray", "hot", "cool", "winter", "viridis", "plasma", "inferno", "red", "green", "blue", etc.

## Testing approach

- Unit test: `launch_viewer` constructs Flask app correctly, routes return expected content types
- Integration test: start server in thread, HTTP GET /api/volumes returns valid JSON, GET /data/image.nii.gz returns bytes
- Manual test: run with real NIfTI data, check browser renders correctly

## Implementation order

1. **app.py** — Flask app with the 3 routes, accepting a data directory path
2. **viewer.html** — Niivue canvas + basic controls (overlay dropdown, colormap dropdown, opacity slider, 3D toggle)
3. **__init__.py (viewer)** — `launch_viewer()` and `launch_viewer_from_result()` functions
4. **Wire up** — add `.view()` to result.py, add `view` command to cli.py, update __init__.py and pyproject.toml
5. **Test** — manual test with real data, then write automated tests

## What NOT to build

- No desktop GUI (no Qt, no napari) — browser only
- No Jupyter widget (yet) — just a standalone server that opens in browser
- No authentication or multi-user support
- No editing/annotation tools — view only
- No Python-side rendering — all rendering is Niivue.js in the browser