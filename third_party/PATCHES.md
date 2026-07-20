# Patches to vendored AutoNUE public-code

CLAUDE.md section 7 requires the AutoNUE tooling to be vendored under `third_party/autonue/`
with "any minimal Python 3 compatibility patches" and "the patch recorded" here.

Upstream: https://github.com/AutoNUE/public-code at commit
`5d9a93beb176b03dd32c79ce050d0fbddc9acd00` (2021-02-15). See
`third_party/autonue/PROVENANCE.txt` for what was vendored and why.

Every patch below was driven by a failure observed on this device, not by style. `anue_labels.py`
and `annotation.py` are **unmodified**: `build-lut` resolves the 8-class collapse by name against
that label table, so it has to remain upstream's word verbatim.

The spec anticipated "Python 2 era" syntax problems. There were none: the code already uses
`from __future__ import print_function` and Python 3 syntax throughout. The real breakages are
API rot (Pillow removed a symbol), dead imports of uninstalled packages, an import-order bug,
and three latent NameErrors.

## Summary

| # | File | Why |
|---|------|-----|
| 1 | `preperation/json2labelImg.py` | `from PIL import PILLOW_VERSION` removed in Pillow 9.0, so upstream `sys.exit(-1)`s before doing any work |
| 2 | `preperation/json2labelImg.py` | `tqdm.write()` called but tqdm never imported: the unknown-label diagnostic raised NameError |
| 3 | `preperation/createLabels.py` | `imageio` and `numpngw` imported at module scope but never used; both absent here, making the script unimportable |
| 4 | `preperation/createLabels.py` | `helpers/` added to `sys.path` inside `main()`, but module-scope imports need it first |
| 5 | `preperation/createLabels.py` | `tqdm.writeError()` does not exist; an empty input set raised AttributeError, then divided by zero |
| 6 | `preperation/createLabels.py` | `--color` branch formatted undefined `f` (parameter is `fn`), raising NameError that hid the real error |
| 7 | `preperation/json2instanceImg.py` | Same as patch 1. `createLabels.py` imports this module at scope, so its probe killed the process even though DRIVYX never generates instance masks |

## Detail

**Patch 1 and 7 (Pillow).** `PILLOW_VERSION` was removed in Pillow 9.0 (this device has 10.2.0).
Upstream's probe is a guard that PIL is really Pillow; it now always raises and calls
`sys.exit(-1)`. Verified on device:

```
$ python -c "from PIL import PILLOW_VERSION"
ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'
```

The `import PIL.Image` immediately after already proves Pillow is present, so the probe is
removed rather than rewritten against `PIL.__version__`. Without this, `gen-masks` produces
nothing and prints only "Please install the module 'Pillow'" on a machine where Pillow is
installed.

**Patch 2 (tqdm).** `createLabelImage()` handles an unrecognised polygon label by calling
`tqdm.write("Something wrong in: " + inJson)`, but the module imports no tqdm. The intended
diagnostic therefore raises `NameError`. DRIVYX depends on exactly this path failing loudly and
legibly (CLAUDE.md rule 30: unresolved label names must abort with a precise message), so the
import is added rather than the call removed.

**Patch 3 (dead imports).** `from imageio import imread, imsave` and `from numpngw import
write_png` sit at module scope. Neither symbol appears anywhere else in the file (only `pandas`
is used, in the `--semisup_da` branch DRIVYX does not take). Neither package is installed, so
the module raised `ModuleNotFoundError` on import. Removed, rather than adding two dependencies
to satisfy dead code.

**Patch 4 (import order).** `createLabels.py` does `from json2labelImg import json2labelImg` at
module scope; that module immediately does `from anue_labels import name2label`, which lives in
`helpers/`. Upstream appends `helpers/` to `sys.path` inside `main()`, which runs long after the
failed import. The script therefore only worked if the caller pre-seeded `PYTHONPATH`. The path
setup is hoisted above the imports so the script is self-contained and `data/masks.py` need not
depend on an undocumented environment convention.

**Patch 5 (empty input).** `tqdm.writeError` is not a tqdm API (`tqdm.write` is). With no input
files upstream raises `AttributeError`, and the very next line computes
`progress * 100 / len(files)`, a division by zero. Both mask the actual problem. Replaced with a
`SystemExit` naming the directory that was searched.

**Patch 6 (undefined name).** The `--color` branch's error handler formats `f`, but the enclosing
function's parameter is `fn`. A colour-conversion failure would raise `NameError` from inside the
handler and discard the real exception. DRIVYX does not pass `--color`, but the fix is one
identifier and removes a trap.

## Verification

The patched tree was exercised on real IDD data before being accepted:

```
$ python -c "import createLabels"          # patches 1,3,4,7
createLabels imports OK

$ json2labelImg('/mnt/nvme/idd/seg/gtFine/val/215/frame0291_gtFine_polygons.json',
                out, 'level3Id')           # patches 1,2
mask shape : (1080, 1920) dtype: uint8
unique ids : [0, 1, 3, 5, 6, 20, 21, 24, 25, 255]
```

## Full diff against upstream

```diff
--- upstream/preperation/createLabels.py
+++ vendored/preperation/createLabels.py
@@ -6,10 +6,18 @@
 import os
 import glob
 import sys
-#from scipy.misc import imread, imsave
-from imageio import imread, imsave
+# DRIVYX patch 3: `from imageio import imread, imsave` and `from numpngw import write_png`
+# were imported at module scope but are never used in this file. Both packages are absent
+# here, so the unused imports made the script unimportable. Removed rather than installing
+# two dependencies for dead code.
 import numpy as np
-from numpngw import write_png
+
+# DRIVYX patch 4: upstream appends helpers/ to sys.path inside main(), but the two imports
+# below run at module scope and transitively import anue_labels from helpers/. The append
+# was therefore too late and the script only worked if the caller pre-seeded PYTHONPATH.
+# Hoisting the path setup above the imports makes the script self-contained.
+sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
+sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
 from json2labelImg import json2labelImg
 from json2instanceImg import json2instanceImg
@@ -57,7 +65,10 @@
         try:
             json2labelImg(fn, dst, 'color')
         except:
-            tqdm.write("Failed to convert: {}".format(f))
+            # DRIVYX patch 6: upstream formatted `f`, undefined in this scope (the parameter
+            # is `fn`), so a colour-conversion failure raised NameError and hid the real
+            # exception.
+            tqdm.write("Failed to convert: {}".format(fn))
             raise
 
     # if args.panoptic and args.instance:
@@ -124,8 +135,12 @@
 
     #print('args.semisup_da', args.semisup_da, len(files))
     if not files:
-        tqdm.writeError(
-            "Did not find any files. Please consult the README.")
+        # DRIVYX patch 5: upstream called tqdm.writeError(), which does not exist (tqdm has
+        # write()), so an empty input set raised AttributeError and then hit a division by
+        # len(files) below. Report the real problem and stop.
+        raise SystemExit(
+            "Did not find any *_polygons.json under {}. Nothing to convert.".format(
+                os.path.join(args.datadir, "gtFine")))
 
     # a bit verbose
     tqdm.write(
--- upstream/preperation/json2labelImg.py
+++ vendored/preperation/json2labelImg.py
@@ -8,17 +8,17 @@
 import sys
 import getopt
 
+# DRIVYX patch 2: createLabelImage() calls tqdm.write() on an unknown label but upstream
+# never imports tqdm, turning the intended diagnostic into a NameError.
+from tqdm import tqdm
+
 import numpy
 
 # Image processing
-# Check if PIL is actually Pillow as expected
-try:
-    from PIL import PILLOW_VERSION
-except:
-    print("Please install the module 'Pillow' for image processing, e.g.")
-    print("pip install pillow")
-    sys.exit(-1)
-
+# DRIVYX patch 1: upstream probed `from PIL import PILLOW_VERSION` to check that PIL is
+# Pillow. PILLOW_VERSION was removed in Pillow 9.0, so on any modern Pillow that probe
+# raises and upstream exits(-1) before rasterising anything. The import below already
+# proves Pillow is installed, so the probe is dropped rather than rewritten.
 try:
     import PIL.Image as Image
     import PIL.ImageDraw as ImageDraw
--- upstream/preperation/json2instanceImg.py
+++ vendored/preperation/json2instanceImg.py
@@ -41,13 +41,9 @@
 from tqdm import tqdm
 
 # Image processing
-# Check if PIL is actually Pillow as expected
-try:
-    from PIL import PILLOW_VERSION
-except:
-    print("Please install the module 'Pillow' for image processing, e.g.")
-    print("pip install pillow")
-    sys.exit(-1)
+# DRIVYX patch 7: identical to patch 1, in the other rasteriser. createLabels.py imports
+# this module at scope, so its broken PILLOW_VERSION probe exited the process before any
+# semantic mask could be produced, even though DRIVYX never generates instance masks.
 
 try:
     import PIL.Image as Image
```
